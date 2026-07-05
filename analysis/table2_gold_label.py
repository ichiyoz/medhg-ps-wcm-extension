"""FINAL Table 2 on the NSQIP-updated GOLD label.

Near-clone of analysis/table2_orders_final.py. Only differences:
  - Loads /Users/yiyezhang/Downloads/medhg_ps_data/bulk_features_with_label_gold.parquet
    and uses `ReadmittedWithin30Days_gold` as the label (NSQIP override where
    matched; WCM elsewhere).
  - Writes to artifacts/newdata/table2_gold_{results.csv,dca.{csv,json,png},oof.npz}.

Same protocol as table2_orders_final.py:
  - 5-fold StratifiedKFold(seed=42)
  - Xtab: fit_preprocess/apply_preprocess (train-only) + one-hot CPT + StandardScaler
  - RF (500 trees, min_samples_leaf=10, sqrt, class_weight='balanced')
  - Isotonic-calibrated (CalibratedClassifierCV, method='isotonic', cv=3), pooled OOF
  - Bootstrap 95% CIs via medhg_ps.evaluate._bootstrap_ci (n_boot=2000)

Rows:
  1) rf_clin              -- RF on Xtab (baseline)
  2) rf_gru_orders        -- RF on [Xtab + order-GRU embedding] (deployable)
  3) rf_gru_orders_prov   -- RF on [Xtab + order-GRU embedding + prov block]
  4) rf_prov              -- RF on [Xtab + prov block only]

    PYTHONPATH=. python analysis/table2_gold_label.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (apply_preprocess, fit_preprocess, load_raw,
                           add_calendar_features, load_order_sequence,
                           collapse_order_runs, build_provider_team_features)
from medhg_ps.deploy import _load_cpt_map
from medhg_ps.evaluate import _bootstrap_ci
from medhg_ps.train import set_seed, _resolve_device

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
MAX_EPOCHS, PATIENCE = (15, 4) if SMOKE else (120, 12)
MAXLEN = 128
EMB_DIM, HID, NUMF = 24, 48, 5
N_BOOT = 200 if SMOKE else 2000
OUT_DIR = Path(C.PROJECT_ROOT) / "artifacts" / "newdata"
OUT_DIR.mkdir(parents=True, exist_ok=True)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)

GOLD_PARQUET = Path("/Users/yiyezhang/Downloads/medhg_ps_data/bulk_features_with_label_gold.parquet")


def RF():
    """Canonical RF (matches analysis/run_rf.py wrapper)."""
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# assemble frame + order sequences + provider team block
# ---------------------------------------------------------------------------
print(f"[t2g] {'SMOKE ' if SMOKE else ''}loading (gold label)...", flush=True)
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

# ---- OVERRIDE label with gold-label parquet ----
gold = pd.read_parquet(GOLD_PARQUET)[["LogID", "ReadmittedWithin30Days_gold", "label_source"]].copy()
gold["LogID"] = gold["LogID"].astype(str)
n_before = len(merged)
merged = merged.merge(gold, on="LogID", how="left")
n_missing_gold = int(merged["ReadmittedWithin30Days_gold"].isna().sum())
if n_missing_gold:
    print(f"[t2g] WARNING: {n_missing_gold} rows without gold label -- falling back to WCM label for those",
          flush=True)
    merged["ReadmittedWithin30Days_gold"] = merged["ReadmittedWithin30Days_gold"].fillna(
        merged["ReadmittedWithin30Days"])
    merged["label_source"] = merged["label_source"].fillna("WCM_fallback")
merged["y_wcm"] = merged["ReadmittedWithin30Days"].astype(int)
merged["ReadmittedWithin30Days"] = merged["ReadmittedWithin30Days_gold"].astype(int)  # now GOLD

prov = build_provider_team_features(raw.enc_prov_edges, raw.prov_attrs, raw.encounters)
merged = merged.merge(prov, on="LogID", how="left")
for c in C.PROVIDER_FEATURE_COLUMNS:
    merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
             + [c for c in C.CALENDAR_FEATURE_COLUMNS if c in merged.columns])
prov_cols = [c for c in C.PROVIDER_FEATURE_COLUMNS if c in merged.columns]
y_all = merged["ReadmittedWithin30Days"].astype(int).values
y_wcm = merged["y_wcm"].astype(int).values
N = len(merged)
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)

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
    for j in range(L):
        seq_idx[r, j] = T2I.get(toks[j], PAD)
        gap = gaps[j]; gap = 0.0 if pd.isna(gap) else max(float(gap), 0.0)
        t = times[j]; hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        seq_num[r, j] = [np.log1p(max(float(reps[j]), 0.0)),
                         np.log1p(gap),
                         (j + 1) / MAXLEN,
                         np.sin(2 * np.pi * hour / 24.0),
                         np.cos(2 * np.pi * hour / 24.0)]

n_gold_source = int((merged["label_source"] == "NSQIP").sum())
n_flip = int(((y_all != y_wcm)).sum())
print(f"[t2g] cohort={N:,}  base(gold)={y_all.mean()*100:.3f}%  "
      f"base(WCM)={y_wcm.mean()*100:.3f}%", flush=True)
print(f"[t2g]  NSQIP-sourced labels: {n_gold_source:,}  |  label flips (gold vs WCM): {n_flip}",
      flush=True)
print(f"[t2g]  vocab={len(vocab)}  median seq len={int(np.median(lengths))}  "
      f"tab feats={len(feat_cols)}  prov feats={len(prov_cols)}", flush=True)


class SeqGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(PAD + 1, EMB_DIM, padding_idx=PAD)
        self.gru = nn.GRU(EMB_DIM + NUMF, HID, batch_first=True)
        self.head = nn.Linear(HID, 2)

    def encode(self, idx, num, lens):
        x = torch.cat([self.emb(idx), num], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False)
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
        try:
            vauc = roc_auc_score(y_all[va], pv)
        except ValueError:
            vauc = 0.5
        if vauc > best:
            best, state, pat = vauc, {k: v.detach().cpu().clone()
                                      for k, v in net.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        emb = net.encode(Xi, Xn, Ln).cpu().numpy()
    return emb, best


def build_design(tr, use_prov, use_gru, gru_emb):
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True),
                           id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    blocks = [Xtab, ohe.transform(cpt_arr)]
    if use_prov:
        blocks.append(merged[prov_cols].values.astype(float))
    if use_gru:
        sc = StandardScaler().fit(gru_emb[tr])
        blocks.append(sc.transform(gru_emb))
    X = np.hstack(blocks)
    return StandardScaler().fit(X[tr]).transform(X)


def rf_calibrated_oof(X, tr, te):
    est = CalibratedClassifierCV(RF(), method="isotonic", cv=3).fit(X[tr], y_all[tr])
    return est.predict_proba(X[te])[:, 1]


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
MODELS = ["rf_clin", "rf_gru_orders", "rf_gru_orders_prov", "rf_prov"]
oof = {m: np.full(N, np.nan) for m in MODELS}

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y_all)):
    rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]

    emb, vb = train_gru(tr, va)

    X_clin = build_design(tr, use_prov=False, use_gru=False, gru_emb=emb)
    X_gru  = build_design(tr, use_prov=False, use_gru=True,  gru_emb=emb)
    X_gp   = build_design(tr, use_prov=True,  use_gru=True,  gru_emb=emb)
    X_prov = build_design(tr, use_prov=True,  use_gru=False, gru_emb=emb)

    oof["rf_clin"][te]            = rf_calibrated_oof(X_clin, tr, te)
    oof["rf_gru_orders"][te]      = rf_calibrated_oof(X_gru,  tr, te)
    oof["rf_gru_orders_prov"][te] = rf_calibrated_oof(X_gp,   tr, te)
    oof["rf_prov"][te]            = rf_calibrated_oof(X_prov, tr, te)

    line = f"[t2g] fold {fi+1}/{K}  gru_val={vb:.3f} | "
    for m in MODELS:
        p = oof[m][te]
        line += f" {m} {roc_auc_score(y_all[te], p):.3f}/{average_precision_score(y_all[te], p):.3f} "
    print(line, flush=True)

# ---------------------------------------------------------------------------
# metrics with bootstrap CIs
# ---------------------------------------------------------------------------
assert all(not np.isnan(oof[m]).any() for m in MODELS)
np.savez(OUT_DIR / "table2_gold_oof.npz", y=y_all, y_wcm=y_wcm,
         **{m: oof[m] for m in MODELS})

rows = []
print(f"\n=== FINAL Table 2 -- GOLD LABEL (calibrated pooled OOF; N={N:,}; base "
      f"{y_all.mean()*100:.2f}%; bootstrap n={N_BOOT}) ===")
for m in MODELS:
    p = oof[m]
    au = float(roc_auc_score(y_all, p))
    ap = float(average_precision_score(y_all, p))
    br = float(brier_score_loss(y_all, p))
    au_lo, au_hi = _bootstrap_ci(y_all, p, roc_auc_score,   N_BOOT, 0.05, 0)
    ap_lo, ap_hi = _bootstrap_ci(y_all, p, average_precision_score, N_BOOT, 0.05, 1)
    rows.append(dict(model=m, auroc=au, auroc_lo=au_lo, auroc_hi=au_hi,
                     auprc=ap, auprc_lo=ap_lo, auprc_hi=ap_hi, brier=br))
    print(f"  {m:22s} AUROC {au:.3f} ({au_lo:.3f}-{au_hi:.3f})  "
          f"AUPRC {ap:.3f} ({ap_lo:.3f}-{ap_hi:.3f})  Brier {br:.4f}")

# paired per-fold deltas vs rf_clin
print("\n=== paired deltas vs rf_clin (per fold) ===")
skf2 = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
folds = list(skf2.split(np.zeros(N), y_all))
for m in ["rf_gru_orders", "rf_gru_orders_prov", "rf_prov"]:
    dau, dap = [], []
    for _, te in folds:
        a0 = roc_auc_score(y_all[te], oof["rf_clin"][te])
        p0 = average_precision_score(y_all[te], oof["rf_clin"][te])
        a1 = roc_auc_score(y_all[te], oof[m][te])
        p1 = average_precision_score(y_all[te], oof[m][te])
        dau.append(a1 - a0); dap.append(p1 - p0)
    dau = np.array(dau); dap = np.array(dap)
    print(f"  {m:22s} dAUROC {dau.mean():+.4f} (folds>0 {int((dau>0).sum())}/{K})   "
          f"dAUPRC {dap.mean():+.4f} (folds>0 {int((dap>0).sum())}/{K})")

# also rf_gru_orders_prov vs rf_gru_orders
dau, dap = [], []
for _, te in folds:
    a0 = roc_auc_score(y_all[te], oof["rf_gru_orders"][te])
    p0 = average_precision_score(y_all[te], oof["rf_gru_orders"][te])
    a1 = roc_auc_score(y_all[te], oof["rf_gru_orders_prov"][te])
    p1 = average_precision_score(y_all[te], oof["rf_gru_orders_prov"][te])
    dau.append(a1 - a0); dap.append(p1 - p0)
dau = np.array(dau); dap = np.array(dap)
print(f"  rf_gru_orders_prov - rf_gru_orders  "
      f"dAUROC {dau.mean():+.4f} (folds>0 {int((dau>0).sum())}/{K})   "
      f"dAUPRC {dap.mean():+.4f} (folds>0 {int((dap>0).sum())}/{K})")

pd.DataFrame(rows).to_csv(OUT_DIR / "table2_gold_results.csv", index=False)

# ---------------------------------------------------------------------------
# DCA on the same OOF panels
# ---------------------------------------------------------------------------
THRESHOLDS = np.round(np.arange(0.02, 0.301, 0.01), 4)
prev = float(y_all.mean())


def nb(p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y_all == 1)).sum())
    fp = int(((pred == 1) & (y_all == 0)).sum())
    return tp / N - (fp / N) * (thr / (1 - thr))


treat_all = np.array([prev - (1 - prev) * (t / (1 - t)) for t in THRESHOLDS])
curves = {m: np.array([nb(oof[m], t) for t in THRESHOLDS]) for m in MODELS}

nb_at = {}
for t in [0.05, 0.10, 0.15, 0.20]:
    j = int(np.argmin(np.abs(THRESHOLDS - t)))
    nb_at[f"{t:.2f}"] = dict(treat_all=float(treat_all[j]),
                             **{m: float(curves[m][j]) for m in MODELS})

dca_rows = []
for i, t in enumerate(THRESHOLDS):
    dca_rows.append(dict(threshold=float(t), treat_all=float(treat_all[i]),
                         treat_none=0.0,
                         **{m: float(curves[m][i]) for m in MODELS}))
pd.DataFrame(dca_rows).to_csv(OUT_DIR / "table2_gold_dca.csv", index=False)
with open(OUT_DIR / "table2_gold_dca.json", "w") as f:
    json.dump(dict(N=N, prevalence=prev, nb_at=nb_at,
                   label="ReadmittedWithin30Days_gold",
                   nsqip_sourced_labels=n_gold_source,
                   flips_vs_wcm=n_flip), f, indent=2)

print("\n=== DCA net benefit at key thresholds (gold label) ===")
print(f"  {'p_t':>5} {'treat_all':>10} " + " ".join(f"{m:>22}" for m in MODELS))
for t in [0.05, 0.10, 0.15, 0.20]:
    j = int(np.argmin(np.abs(THRESHOLDS - t)))
    print(f"  {t:>5.2f} {treat_all[j]:>10.4f} "
          + " ".join(f"{curves[m][j]:>22.4f}" for m in MODELS))

# DCA plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {"rf_clin": "#1f77b4", "rf_gru_orders": "#2ca02c",
              "rf_gru_orders_prov": "#d62728", "rf_prov": "#9467bd"}
    for m in MODELS:
        ax.plot(THRESHOLDS, curves[m], color=colors[m], lw=1.8, label=m)
    ax.plot(THRESHOLDS, treat_all, "k--", lw=1.0, label="treat-all")
    ax.axhline(0, color="k", ls=":", lw=1.0, label="treat-none")
    ax.axvline(prev, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title(f"DCA — Table 2, GOLD label (N={N}, base {prev*100:.2f}%)")
    ax.set_xlim(0.02, 0.30)
    lo = min(curves[m].min() for m in MODELS) - 0.01
    ax.set_ylim(min(lo, treat_all.min() - 0.005), 0.06)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "table2_gold_dca.png", dpi=180)
    print(f"[t2g] wrote {OUT_DIR/'table2_gold_dca.png'}")
except Exception as e:
    print(f"[t2g] DCA plot skipped: {e}")

print(f"\n[t2g] wrote:"
      f"\n  {OUT_DIR/'table2_gold_results.csv'}"
      f"\n  {OUT_DIR/'table2_gold_dca.csv'}"
      f"\n  {OUT_DIR/'table2_gold_dca.json'}"
      f"\n  {OUT_DIR/'table2_gold_oof.npz'}")
