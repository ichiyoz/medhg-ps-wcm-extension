"""DICE on order-sequence GRU embeddings.

Per fold the order-sequence GRU is retrained on train rows only (mirrors
cv_seq_gru_orders); the encoder's final hidden state E is extracted for all
rows and standardized (scaler fit on train). DICE (K, d) is fit on E; its
soft memberships (chat) and latent (z) are fed downstream with the tabular
block. All downstream models isotonic-calibrated pooled OOF, RF canonical.

Also runs a final-fit stratification (K=4, GRU + DICE trained on all rows):
per-tier n + readmit rate, high/low ratio, and the top over-represented
OrderGroup tokens per tier (defining phenotype signal).

    PYTHONPATH=. python analysis/cv_dice_orders.py
"""
from __future__ import annotations
import os
os.environ.setdefault("DICE_SURROGATE_SIG", "1")           # bound runtime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (add_calendar_features, apply_preprocess,
                           build_provider_team_features, collapse_order_runs,
                           fit_preprocess, load_order_sequence, load_raw)
from medhg_ps.deploy import _load_cpt_map
from medhg_ps.train import _resolve_device, set_seed
import analysis.dice as dice

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K_FOLDS, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
MAX_EPOCHS, PATIENCE = (10, 3) if SMOKE else (120, 12)
MAXLEN = 128
EMB_DIM, HID, NUMF = 24, 48, 5
DK, DD = 4, 16                                             # fixed DICE K, latent d
NAS_K = (3,) if SMOKE else (3, 4, 5)
NAS_D = (16,) if SMOKE else (16, 32)

def _RF():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                  max_features="sqrt", class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)
def _HGB():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=SEED)
def _LR():
    return LogisticRegression(max_iter=1000, class_weight="balanced")
dev = _resolve_device(C.DEFAULTS_TRAIN.device)

# ---------------------------------------------------------------------
# Frame + per-encounter order sequences (mirrors cv_seq_gru_orders)
# ---------------------------------------------------------------------
print(f"[dord] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
cpt_map = _load_cpt_map()
raw = load_raw()
enc_nodupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
merged = (raw.encounters.merge(enc_nodupes, on="LogID", how="inner")
          .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]], on="LogID", how="inner")
          .reset_index(drop=True))
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)

feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
             + [c for c in C.CALENDAR_FEATURE_COLUMNS if c in merged.columns])
y_all = merged["ReadmittedWithin30Days"].astype(int).values
N = len(merged)
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)
anyz = pd.to_numeric(merged.get("SDOH_Any_Z", 0), errors="coerce").fillna(0).values.astype(int)

# collapsed order runs -> vocab -> per-encounter last-MAXLEN token stream
orders = collapse_order_runs(load_order_sequence())
orders["LogID"] = orders["LogID"].astype(str)
orders["RunStart"] = pd.to_datetime(orders["RunStart"], errors="coerce")
orders = orders.sort_values(["LogID", "SeqInEncounter"])
vocab = orders["OrderGroup"].value_counts().index.tolist()
T2I = {t: i for i, t in enumerate(vocab)}
PAD = len(vocab)

row_of = {l: i for i, l in enumerate(merged["LogID"])}
seq_idx = np.full((N, MAXLEN), PAD, dtype=np.int64)
seq_num = np.zeros((N, MAXLEN, NUMF), dtype=np.float32)
lengths = np.ones(N, dtype=np.int64)
enc_tokens: list = [[] for _ in range(N)]                  # per-row token bag (for phenotype)
for lid, grp in orders.groupby("LogID", sort=False):
    r = row_of.get(lid)
    if r is None:
        continue
    grp = grp.tail(MAXLEN)
    toks = grp["OrderGroup"].tolist()
    reps = grp["RepeatCount"].tolist()
    gaps = grp["MinutesFromPrev"].tolist()
    times = grp["RunStart"].tolist()
    L = len(toks); lengths[r] = max(L, 1)
    enc_tokens[r] = toks
    for j in range(L):
        seq_idx[r, j] = T2I.get(toks[j], PAD)
        gap = gaps[j]; gap = 0.0 if pd.isna(gap) else max(float(gap), 0.0)
        t = times[j]; hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        seq_num[r, j] = [np.log1p(max(float(reps[j]), 0.0)),
                         np.log1p(gap),
                         (j + 1) / MAXLEN,
                         np.sin(2 * np.pi * hour / 24.0),
                         np.cos(2 * np.pi * hour / 24.0)]

print(f"[dord] cohort={N:,}  base={y_all.mean()*100:.2f}%  vocab={len(vocab)}  "
      f"median seq len={int(np.median(lengths))}  max={int(lengths.max())}  "
      f"tab feats={len(feat_cols)}  surrogate_sig={os.environ.get('DICE_SURROGATE_SIG')}",
      flush=True)


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
    return emb


def gru_embeddings(tr, va):
    """Train order-GRU on train rows; return standardized embeddings for ALL
    rows (scaler fit on train)."""
    emb = train_gru(tr, va)
    return StandardScaler().fit(emb[tr]).transform(emb)


def build_xtab(tr):
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    X = np.hstack([Xtab, ohe.transform(cpt_arr)])
    return StandardScaler().fit(X[tr]).transform(X)


def build_v(fit_rows):
    age = pd.to_numeric(merged["AgeYears"], errors="coerce").values.reshape(-1, 1)
    age = np.where(np.isnan(age), np.nanmedian(age[fit_rows]), age)
    age = (age - age[fit_rows].mean()) / (age[fit_rows].std() + 1e-8)
    g = merged[["Gender"]].astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(g.iloc[fit_rows])
    return np.hstack([age, ohe.transform(g)]).astype(np.float32)


def _calibrated_oof_probs(model_factory, X, tr, te):
    est = CalibratedClassifierCV(model_factory(), method="isotonic", cv=3)
    est.fit(X[tr], y_all[tr])
    return est.predict_proba(X[te])[:, 1]


def _bootstrap_ci(y, p, fn, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); n = len(y); vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try: vals[i] = fn(y[idx], p[idx])
        except Exception: vals[i] = np.nan
    return (float(np.nanpercentile(vals, 100 * alpha / 2)),
            float(np.nanpercentile(vals, 100 * (1 - alpha / 2))))


def main():
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    models = ["tab",
              "gru_orders_stack",
              "dice_ords_only_lr",
              "dice_ords_rf",
              "dice_ords_full_rf",
              "dice_ords_hgb",
              "dice_ords_lr"]
    oof = {m: np.full(N, np.nan) for m in models}
    Ksel: list = []; dsel: list = []

    for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y_all)):
        rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
        nv = int(round(VAL_FRAC * N)); val, tr = tr_all[:nv], tr_all[nv:]
        full_tr = np.concatenate([tr, val])

        # 1. order-GRU embeddings (train on tr only, extract for all)
        E = gru_embeddings(tr, val)

        # 2. DICE with per-fold NAS (over K, d) on TRAIN embeddings
        v = build_v(full_tr)
        Kstar, dstar, _ = dice.search_fit(E[tr], y_all[tr], E[val], y_all[val],
                                          v[tr], v[val], Ks=NAS_K, ds=NAS_D)
        Ksel.append(Kstar); dsel.append(dstar)
        dmodel = dice.fit(E[full_tr], y_all[full_tr], Kstar, dstar, v[full_tr])
        chat = dice.cluster_proba(dmodel, E)                  # [N, K]
        z = dice.embed(dmodel, E)                             # [N, d]

        # 3. tabular block + downstream designs
        X = build_xtab(tr)
        XD = np.hstack([chat, X])
        XZ = np.hstack([z, chat, X])
        XE = np.hstack([E, X])

        oof["tab"][te]                 = _calibrated_oof_probs(_RF, X,    full_tr, te)
        oof["gru_orders_stack"][te]    = _calibrated_oof_probs(_RF, XE,   full_tr, te)
        oof["dice_ords_only_lr"][te]   = _calibrated_oof_probs(_LR, chat, full_tr, te)
        oof["dice_ords_rf"][te]        = _calibrated_oof_probs(_RF, XD,   full_tr, te)
        oof["dice_ords_full_rf"][te]   = _calibrated_oof_probs(_RF, XZ,   full_tr, te)
        oof["dice_ords_hgb"][te]       = _calibrated_oof_probs(_HGB, XD,  full_tr, te)
        oof["dice_ords_lr"][te]        = _calibrated_oof_probs(_LR, XD,   full_tr, te)
        print(f"[dord] fold {fi+1}/{K_FOLDS}  K*={Kstar} d*={dstar}", flush=True)

    # ---- results ----
    print(f"\n=== FINAL Table | calibrated pooled OOF | N={N:,} base {y_all.mean()*100:.2f}% "
          f"| NAS K*={Ksel} d*={dsel} ===")
    print(f"  {'model':22s} {'AUROC (95% CI)':>26s} {'AUPRC (95% CI)':>26s} {'Brier':>8s}")
    rows = []
    for m in models:
        p = oof[m]
        au = roc_auc_score(y_all, p); ap = average_precision_score(y_all, p)
        br = brier_score_loss(y_all, p)
        au_lo, au_hi = _bootstrap_ci(y_all, p, roc_auc_score, seed=1)
        ap_lo, ap_hi = _bootstrap_ci(y_all, p, average_precision_score, seed=2)
        rows.append((m, au, au_lo, au_hi, ap, ap_lo, ap_hi, br))
        print(f"  {m:22s}  {au:.3f} ({au_lo:.3f}-{au_hi:.3f})   "
              f"{ap:.3f} ({ap_lo:.3f}-{ap_hi:.3f})  {br:.4f}")
    pd.DataFrame(rows, columns=["model","AUROC","AUROC_lo","AUROC_hi","AUPRC","AUPRC_lo","AUPRC_hi","Brier"])\
        .to_csv("artifacts/newdata/cv_dice_orders_results.csv", index=False)
    np.savez("artifacts/newdata/cv_dice_orders_oof.npz", y=y_all, **oof)

    # ---- reference deltas ----
    RF_TAB_AUROC, RF_TAB_AUPRC = rows[0][1], rows[0][4]
    print(f"\n  ceiling reference: tab RF = {RF_TAB_AUROC:.3f} / {RF_TAB_AUPRC:.3f}")
    print(f"  paired vs RF tab ceiling (per patient, on pooled OOF):")
    for r in rows[1:]:
        print(f"    {r[0]:22s} dAUROC {r[1]-RF_TAB_AUROC:+.4f}   dAUPRC {r[4]-RF_TAB_AUPRC:+.4f}")
    print(f"  prior tabular-DICE dice_gbdt (SDoH):        AUROC 0.728 / AUPRC 0.251")
    print(f"  prior tabular-DICE dice_gbdt (SQL-fixed):   AUROC 0.709 / AUPRC 0.173")

    # ---- stratification: final fit on all rows ----
    print(f"\n=== stratification: DICE(K={DK}) on order-GRU embeddings (final fit, all rows) ===",
          flush=True)
    rng = np.random.default_rng(SEED); allr = np.arange(N).copy(); rng.shuffle(allr)
    nv = int(round(VAL_FRAC * N)); val, tr = allr[:nv], allr[nv:]
    E = gru_embeddings(tr, val)
    fm = dice.fit(E, y_all, DK, DD, build_v(np.arange(N)))
    hard = dice.cluster_proba(fm, E).argmax(1)
    tiers = sorted(((int((hard == k).sum()),
                     float(y_all[hard == k].mean()) if (hard == k).any() else float("nan"), k)
                    for k in range(DK)), key=lambda t: (t[1] if t[1] == t[1] else 9))
    for n_k, r, k in tiers:
        za = anyz[hard == k].mean() * 100 if n_k else float("nan")
        print(f"  cluster {k}: n={n_k:5d}  readmit={r*100:5.2f}%   SDOH_Any_Z in tier={za:4.1f}%")
    valid = [r for n_k, r, _ in tiers if r == r and r > 0 and n_k]
    if len(valid) >= 2:
        print(f"  high/low risk ratio = {max(valid)/min(valid):.2f}")
    r1, r0 = y_all[anyz == 1].mean(), y_all[anyz == 0].mean()
    print(f"  [anchor] SDOH_Any_Z flagged readmit {r1*100:.1f}% vs {r0*100:.1f}% (RR {r1/r0:.1f})")
    print(f"  (prior DICE-on-graph K=4 tiers: 2.5% / 5.4% / 12.9% / (empty))")
    print(f"  (prior tabular-DICE K=4 tiers: 5.1 / 13.1 / 13.3 / 26.1%)")

    # ---- phenotype-defining tokens per tier ----
    print(f"\n=== top over-represented OrderGroup tokens per tier (lift vs cohort) ===")
    cohort_freq: dict[str, float] = {}
    for toks in enc_tokens:
        for t in set(toks):
            cohort_freq[t] = cohort_freq.get(t, 0) + 1
    cohort_pct = {t: v / N for t, v in cohort_freq.items()}
    for n_k, r, k in tiers:
        if not n_k:
            continue
        mask = hard == k
        f: dict[str, float] = {}
        for i in np.where(mask)[0]:
            for t in set(enc_tokens[i]):
                f[t] = f.get(t, 0) + 1
        n_tier = int(mask.sum())
        lifts = [(t, (f[t] / n_tier) / cohort_pct[t], f[t])
                 for t in f if cohort_pct.get(t, 0) > 0 and f[t] >= 20]
        lifts.sort(key=lambda z: -z[1])
        print(f"  cluster {k}  (n={n_k}, readmit {r*100:.1f}%):")
        for t, lift, cnt in lifts[:5]:
            print(f"     {t:40s} lift={lift:4.1f}  n_in_tier={cnt}")


if __name__ == "__main__":
    main()
