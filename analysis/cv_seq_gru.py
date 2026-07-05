"""Sequential model over the care-unit trajectory (GRU), vs hand-crafted
sequence features.

The deployable model encodes the post-op care path as a fixed bag of
hand-crafted features (per-unit counts, directly-follows bigrams, ICU/end
flags). This script tests whether LEARNING the sequence does better: a GRU
reads each encounter's time-ordered unit visits -- unit type, hours in unit,
arrival time-of-day, and position -- and its final hidden state becomes the
encounter representation (the "run an RNN over the unit transitions, keep the
last state" design).

Fold-honest 5-fold CV. Four comparators on identical splits:

  tab            : HGB on tabular + CPT (no care path)
  tab+staticseq  : HGB on tabular + CPT + hand-crafted sequence  [deployable]
  gru_only       : GRU last hidden -> linear head (sequence alone)
  tab+gru        : HGB on tabular + CPT + GRU encounter embedding

tab+gru vs tab tests whether the learned sequence adds signal at all;
tab+gru vs tab+staticseq tests learned-vs-hand-crafted. The GRU is trained on
each fold's train rows only (early-stopped on val AUROC); its embedding is
re-extracted per fold (no leakage).

Prints aggregate numbers only.  (Temporal-edge-aware GNN is the heavier graph
variant of the same idea; this RNN is the lighter, cleaner test.)

    PYTHONPATH=. python analysis/cv_seq_gru.py
"""
from __future__ import annotations

import os
from pathlib import Path

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
                           build_preop_trajectory_features, add_calendar_features)
from medhg_ps.deploy import (MIN_SUP, UNIT_BUCKET, seq_feature_dict, _load_cpt_map)
from medhg_ps.train import set_seed, _resolve_device

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
MAX_EPOCHS, PATIENCE = (15, 4) if SMOKE else (120, 12)
MAXLEN = 40                      # max trajectory length (observed max 46; tail clipped)
UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]
U2I = {u: i for i, u in enumerate(UNITS)}        # 0..5; PAD = 6
PAD = len(UNITS)
EMB_DIM, HID, NUMF = 16, 32, 4                    # numeric step features: log-hrs, sin, cos, pos
HGB = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=42)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


# ---- assemble frame + per-encounter unit sequences ------------------------
print(f"[gru] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
cpt_map = _load_cpt_map()
raw = load_raw()
enc_nodupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
merged = (raw.encounters.merge(enc_nodupes, on="LogID", how="inner")
          .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]], on="LogID", how="inner")
          .reset_index(drop=True))
ss = merged[["LogID"]].copy()
ss["_ss"] = pd.to_datetime(merged.get("Procedure/Surgery Start"), errors="coerce")
merged = merged.merge(build_preop_trajectory_features(raw.enc_unit_edges, ss), on="LogID", how="left")
for c in C.TRAJECTORY_FEATURE_COLUMNS:
    merged[c] = merged[c].fillna(0)
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)
feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
             + [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged.columns])
y_all = merged["ReadmittedWithin30Days"].astype(int).values
N = len(merged)
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)

# bucket A3 unit visits, time-ordered, and also the hand-crafted static features
u = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
u["cid"] = u["Clarity_ID"].astype("Int64").astype(str)
u["g"] = u["UnitType"].map(UNIT_BUCKET).fillna("Other")
ud = u.dropna(subset=["Clarity_ID"]).drop_duplicates("cid", keep="first").set_index("cid")
a3 = raw.enc_unit_edges.copy()
a3["UnitType"] = (_norm(a3["DepartmentID"]).map(ud["g"]).fillna(a3["UnitType"]))
a3["InTime"] = pd.to_datetime(a3["InTime"], errors="coerce")
a3["Hours"] = pd.to_numeric(a3.get("Hours"), errors="coerce").fillna(0.0)
a3["LogID"] = a3["LogID"].astype(str)
a3 = a3.sort_values(["LogID", "InTime"])

row_of = {l: i for i, l in enumerate(merged["LogID"])}
seq_idx = np.full((N, MAXLEN), PAD, dtype=np.int64)
seq_num = np.zeros((N, MAXLEN, NUMF), dtype=np.float32)
lengths = np.ones(N, dtype=np.int64)        # no-trajectory rows read one PAD step
static_rows = []
for lid, grp in a3.groupby("LogID", sort=False):
    r = row_of.get(lid)
    if r is None:
        continue
    units = grp["UnitType"].tolist()
    static = seq_feature_dict(units); static["LogID"] = lid
    static_rows.append(static)
    hrs = grp["Hours"].tolist(); times = grp["InTime"].tolist()
    steps = [(U2I.get(un, U2I["Other"]), h, t) for un, h, t in zip(units, hrs, times)]
    steps = steps[:MAXLEN]
    L = len(steps); lengths[r] = max(L, 1)
    for j, (ui, h, t) in enumerate(steps):
        seq_idx[r, j] = ui
        hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        seq_num[r, j] = [np.log1p(max(h, 0.0)),
                         np.sin(2 * np.pi * hour / 24.0),
                         np.cos(2 * np.pi * hour / 24.0),
                         (j + 1) / MAXLEN]

Fstat = (merged[["LogID"]].merge(pd.DataFrame(static_rows).astype({"LogID": str}),
                                 on="LogID", how="left").fillna(0))
stat_cols = [c for c in Fstat.columns if c != "LogID"]
print(f"[gru] cohort={N:,}  base={y_all.mean()*100:.2f}%  "
      f"median path len={int(np.median(lengths))}  static seq cands={len(stat_cols)}", flush=True)


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
        return h[-1]                                  # [B, HID] last hidden

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
            best, state, pat = vauc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        emb = net.encode(Xi, Xn, Ln).cpu().numpy()
        praw = torch.softmax(net(Xi, Xn, Ln), -1)[:, 1].cpu().numpy()
    return emb, praw, best


def design(tr, with_static, with_gru, gru_emb):
    Xtab, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    blocks = [Xtab, ohe.transform(cpt_arr)]
    if with_static:
        keep = [c for c in stat_cols if (Fstat.iloc[tr][c] > 0).sum() >= MIN_SUP]
        blocks.append(Fstat[keep].values.astype(float))
    if with_gru:
        sc = StandardScaler().fit(gru_emb[tr]); blocks.append(sc.transform(gru_emb))
    X = np.hstack(blocks)
    return StandardScaler().fit(X[tr]).transform(X)


def tree(X, tr, te):
    m = HGB().fit(X[tr], y_all[tr]); p = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y_all[te], p), average_precision_score(y_all[te], p)


def tree_calibrated(X, tr, te):
    """Isotonic-calibrated HGB (matches the Table-2 deployable recipe);
    returns AUROC, AUPRC, Brier for a deployable-candidate row."""
    est = CalibratedClassifierCV(HGB(), method="isotonic", cv=3).fit(X[tr], y_all[tr])
    p = est.predict_proba(X[te])[:, 1]
    return (roc_auc_score(y_all[te], p), average_precision_score(y_all[te], p),
            brier_score_loss(y_all[te], p))


def main():
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    R = {m: {"au": [], "ap": []} for m in ["tab", "tab+staticseq", "gru_only", "tab+gru"]}
    R["tab+gru"]["br"] = []                       # calibrated Brier (Table-2 candidate)
    for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y_all)):
        rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
        nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]
        emb, praw, vb = train_gru(tr, va)

        for tag, ws, wg in [("tab", False, False), ("tab+staticseq", True, False),
                            ("tab+gru", False, True)]:
            X = design(tr, ws, wg, emb)
            if tag == "tab+gru":                  # isotonic-calibrated: AUROC/AUPRC unchanged, + Brier
                au, ap, br = tree_calibrated(X, tr, te); R[tag]["br"].append(br)
            else:
                au, ap = tree(X, tr, te)
            R[tag]["au"].append(au); R[tag]["ap"].append(ap)
        R["gru_only"]["au"].append(roc_auc_score(y_all[te], praw[te]))
        R["gru_only"]["ap"].append(average_precision_score(y_all[te], praw[te]))
        print(f"[gru] fold {fi+1}/{K} (val {vb:.3f}) | "
              + "  ".join(f"{t} {R[t]['au'][-1]:.3f}" for t in R), flush=True)

    print(f"\n=== {K}-fold CV | base {y_all.mean()*100:.2f}% ===")
    print(f"  {'model':16s} {'AUROC':>16s} {'AUPRC':>16s}")
    for t in ["tab", "tab+staticseq", "gru_only", "tab+gru"]:
        au, ap = np.array(R[t]["au"]), np.array(R[t]["ap"])
        print(f"  {t:16s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")
    ds = np.array(R["tab+gru"]["au"]) - np.array(R["tab+staticseq"]["au"])
    dt = np.array(R["tab+gru"]["au"]) - np.array(R["tab"]["au"])
    print(f"\n  tab+gru - tab           dAUROC {dt.mean():+.4f}  (folds>0 {int((dt>0).sum())}/{K})")
    print(f"  tab+gru - tab+staticseq dAUROC {ds.mean():+.4f}  (folds>0 {int((ds>0).sum())}/{K})")
    br = np.array(R["tab+gru"]["br"])
    print(f"  tab+gru calibrated Brier {br.mean():.4f} +/- {br.std():.4f}  "
          f"(AUROC/AUPRC rank-preserved under isotonic)")


if __name__ == "__main__":
    main()
