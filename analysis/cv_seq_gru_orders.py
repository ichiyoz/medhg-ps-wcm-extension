"""Sequential model over the ORDER trajectory (GRU), + care-team block.

Order-substrate sibling of cv_seq_gru.py. Instead of the care-unit path
(A3), each encounter is the time-ordered sequence of ORDERS placed during
the episode (ED->discharge if via ED, else admit->discharge), tokenized to
grouped therapeutic/order classes (~80 tokens). Consecutive identical
tokens are collapsed into runs (data.collapse_order_runs) so the sequence
carries state changes, with RepeatCount as intensity.

Providers enter as an encounter-level care-team block (Option B:
build_provider_team_features) -- team composition + primary-surgeon volume
-- NOT per-order attribution (ordering-provider is unreliable).

Fold-honest 5-fold CV. Comparators on identical splits:

  tab            : HGB on tabular + CPT
  tab+prov       : + care-team block            (does provider add over case-mix?)
  gru_only       : order-GRU last hidden -> linear head (sequence alone)
  tab+gru        : + order-GRU encounter embedding
  tab+gru+prov   : + both                        (deployable candidate)

Reads A1 + A2 + A4 + order_sequence + labels (no units).

    PYTHONPATH=. python analysis/cv_seq_gru_orders.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (apply_preprocess, fit_preprocess, load_raw,
                           add_calendar_features, load_order_sequence,
                           collapse_order_runs, build_provider_team_features)
from medhg_ps.deploy import _load_cpt_map
from medhg_ps.train import set_seed, _resolve_device

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
MAX_EPOCHS, PATIENCE = (15, 4) if SMOKE else (120, 12)
MAXLEN = 128                      # keep the LAST MAXLEN runs (pre-discharge = most
                                 # readmit-relevant); tune from collapsed lengths
EMB_DIM, HID, NUMF = 24, 48, 5   # step feats: log-repeat, log-gap, pos, sin/cos tod
HGB = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=42)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)


# ---- assemble frame + per-encounter order sequences -----------------------
print(f"[ordgru] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
cpt_map = _load_cpt_map()
raw = load_raw()                                 # units-optional; A3/A5 unused here
enc_nodupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
merged = (raw.encounters.merge(enc_nodupes, on="LogID", how="inner")
          .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]], on="LogID", how="inner")
          .reset_index(drop=True))
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)

# provider care-team block (Option B)
prov = build_provider_team_features(raw.enc_prov_edges, raw.prov_attrs, raw.encounters)
merged = merged.merge(prov, on="LogID", how="left")
for c in C.PROVIDER_FEATURE_COLUMNS:
    merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
             + [c for c in C.CALENDAR_FEATURE_COLUMNS if c in merged.columns])
prov_cols = [c for c in C.PROVIDER_FEATURE_COLUMNS if c in merged.columns]
y_all = merged["ReadmittedWithin30Days"].astype(int).values
N = len(merged)
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)

# order token sequences: collapse runs, build a vocab, encode LAST MAXLEN runs
orders = collapse_order_runs(load_order_sequence())
orders["LogID"] = orders["LogID"].astype(str)
orders["RunStart"] = pd.to_datetime(orders["RunStart"], errors="coerce")
orders = orders.sort_values(["LogID", "SeqInEncounter"])
vocab = orders["OrderGroup"].value_counts().index.tolist()   # all present tokens
T2I = {t: i for i, t in enumerate(vocab)}
PAD = len(vocab)

row_of = {l: i for i, l in enumerate(merged["LogID"])}
seq_idx = np.full((N, MAXLEN), PAD, dtype=np.int64)
seq_num = np.zeros((N, MAXLEN, NUMF), dtype=np.float32)
lengths = np.ones(N, dtype=np.int64)             # no-order rows read one PAD step
for lid, grp in orders.groupby("LogID", sort=False):
    r = row_of.get(lid)
    if r is None:
        continue
    grp = grp.tail(MAXLEN)                        # keep most recent runs
    toks = grp["OrderGroup"].tolist()
    reps = grp["RepeatCount"].tolist()
    gaps = grp["MinutesFromPrev"].tolist()
    times = grp["RunStart"].tolist()
    L = len(toks); lengths[r] = max(L, 1)
    for j in range(L):
        seq_idx[r, j] = T2I.get(toks[j], PAD)
        gap = gaps[j]; gap = 0.0 if pd.isna(gap) else max(float(gap), 0.0)
        t = times[j]; hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        seq_num[r, j] = [np.log1p(max(float(reps[j]), 0.0)),
                         np.log1p(gap),
                         (j + 1) / MAXLEN,
                         np.sin(2 * np.pi * hour / 24.0),
                         np.cos(2 * np.pi * hour / 24.0)]

print(f"[ordgru] cohort={N:,}  base={y_all.mean()*100:.2f}%  vocab={len(vocab)}  "
      f"median seq len={int(np.median(lengths))}  max={int(lengths.max())}", flush=True)


class SeqGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(PAD + 1, EMB_DIM, padding_idx=PAD)
        self.gru = nn.GRU(EMB_DIM + NUMF, HID, batch_first=True)
        self.head = nn.Linear(HID, 2)

    def encode(self, idx, num, lens):
        x = torch.cat([self.emb(idx), num], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]

    def forward(self, idx, num, lens):
        return self.head(self.encode(idx, num, lens))


def train_gru(tr, va):
    set_seed(SEED)
    net = SeqGRU().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    npos, nneg = float((y_all[tr] == 1).sum()), float((y_all[tr] == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)],
                     dtype=torch.float32, device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    Xi = torch.tensor(seq_idx, device=dev); Xn = torch.tensor(seq_num, device=dev)
    Ln = torch.tensor(lengths, device=dev); Y = torch.tensor(y_all, device=dev)
    best, state, pat = -1.0, None, PATIENCE
    rng = np.random.default_rng(SEED)
    for ep in range(MAX_EPOCHS):
        net.train(); perm = rng.permutation(tr)
        for s in range(0, len(perm), 4096):
            b = perm[s:s + 4096]
            opt.zero_grad()
            loss = lf(net(Xi[b], Xn[b], Ln[b]), Y[b])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.softmax(net(Xi[va], Xn[va], Ln[va]), -1)[:, 1].cpu().numpy()
        try: vauc = roc_auc_score(y_all[va], pv)
        except ValueError: vauc = 0.5
        if vauc > best:
            best, state, pat = vauc, {k: v.detach().cpu().clone()
                                      for k, v in net.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        emb = net.encode(Xi, Xn, Ln).cpu().numpy()
        praw = torch.softmax(net(Xi, Xn, Ln), -1)[:, 1].cpu().numpy()
    return emb, praw, best


def design(tr, with_prov, with_gru, gru_emb):
    Xtab, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    blocks = [Xtab, ohe.transform(cpt_arr)]
    if with_prov:
        blocks.append(merged[prov_cols].values.astype(float))
    if with_gru:
        sc = StandardScaler().fit(gru_emb[tr]); blocks.append(sc.transform(gru_emb))
    X = np.hstack(blocks)
    return StandardScaler().fit(X[tr]).transform(X)


def tree(X, tr, te):
    m = HGB().fit(X[tr], y_all[tr]); p = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y_all[te], p), average_precision_score(y_all[te], p)


def tree_calibrated(X, tr, te):
    est = CalibratedClassifierCV(HGB(), method="isotonic", cv=3).fit(X[tr], y_all[tr])
    p = est.predict_proba(X[te])[:, 1]
    return (roc_auc_score(y_all[te], p), average_precision_score(y_all[te], p),
            brier_score_loss(y_all[te], p))


def main():
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    tags = ["tab", "tab+prov", "gru_only", "tab+gru", "tab+gru+prov"]
    R = {m: {"au": [], "ap": []} for m in tags}
    R["tab+gru+prov"]["br"] = []
    for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y_all)):
        rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
        nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]
        emb, praw, vb = train_gru(tr, va)

        for tag, wp, wg in [("tab", False, False), ("tab+prov", True, False),
                            ("tab+gru", False, True), ("tab+gru+prov", True, True)]:
            X = design(tr, wp, wg, emb)
            if tag == "tab+gru+prov":
                au, ap, br = tree_calibrated(X, tr, te); R[tag]["br"].append(br)
            else:
                au, ap = tree(X, tr, te)
            R[tag]["au"].append(au); R[tag]["ap"].append(ap)
        R["gru_only"]["au"].append(roc_auc_score(y_all[te], praw[te]))
        R["gru_only"]["ap"].append(average_precision_score(y_all[te], praw[te]))
        print(f"[ordgru] fold {fi+1}/{K} (val {vb:.3f}) | "
              + "  ".join(f"{t} {R[t]['au'][-1]:.3f}" for t in tags), flush=True)

    print(f"\n=== {K}-fold CV | base {y_all.mean()*100:.2f}% ===")
    print(f"  {'model':16s} {'AUROC':>16s} {'AUPRC':>16s}")
    for t in tags:
        au, ap = np.array(R[t]["au"]), np.array(R[t]["ap"])
        print(f"  {t:16s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")
    dp = np.array(R["tab+prov"]["au"]) - np.array(R["tab"]["au"])
    dg = np.array(R["tab+gru"]["au"]) - np.array(R["tab"]["au"])
    dgp = np.array(R["tab+gru+prov"]["au"]) - np.array(R["tab+gru"]["au"])
    print(f"\n  tab+prov     - tab      dAUROC {dp.mean():+.4f}  (folds>0 {int((dp>0).sum())}/{K})")
    print(f"  tab+gru      - tab      dAUROC {dg.mean():+.4f}  (folds>0 {int((dg>0).sum())}/{K})")
    print(f"  tab+gru+prov - tab+gru  dAUROC {dgp.mean():+.4f}  (folds>0 {int((dgp>0).sum())}/{K})")
    br = np.array(R["tab+gru+prov"]["br"])
    print(f"  tab+gru+prov calibrated Brier {br.mean():.4f} +/- {br.std():.4f}")


if __name__ == "__main__":
    main()
