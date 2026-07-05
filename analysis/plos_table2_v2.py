"""PLOS Table 2 v2 — adds MedHG-PS raw ie-HGCN + F1/precision/recall metrics.

Same 5 models as analysis/plos_table2.py (OOF cached at
artifacts/newdata/plos_table2_oof.npz) plus a 6th model: raw MedHG-PS
ie-HGCN classifier trained on the A2-only (encounter+provider)
heterograph. A3 units excluded to avoid trajectory leakage into PLOS.
Same 9 leaky features dropped. Same 5-fold seed 42 and val_frac 0.10.

For every model we now also report the operating-point metrics at the
F1-optimal threshold on the pooled OOF: F1, precision, recall,
specificity, flag rate. Bootstrap 95% CIs (n=2000) computed with a fixed
threshold from the full sample (documented as such).
"""
from __future__ import annotations
import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

import medhg_ps.config as C
import medhg_ps.data as D
from medhg_ps.data import (apply_preprocess, fit_preprocess, load_raw,
                           add_calendar_features, build_provider_team_features)
from medhg_ps.deploy import _load_cpt_map
from medhg_ps.evaluate import _bootstrap_ci
from medhg_ps.train import set_seed, _resolve_device

SEED = 42
K = 5
VAL_FRAC = 0.10
HID = 48
MAX_EPOCHS = 200
PATIENCE = 15
N_BOOT = 2000

OUT_DIR = Path(C.PROJECT_ROOT) / "artifacts" / "newdata"
OUT_DIR.mkdir(parents=True, exist_ok=True)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)

LEAKY_FEATS = {
    "Discharge Disposition",
    "# of Cardiac Arrest Requiring CPR",
    "# of Stroke/Cerebral Vascular Acccident (CVA)",
    "# of Postop Unplanned Intubation",
    "preop_los_acute_hr", "preop_los_intensive_hr", "preop_los_intermediate_hr",
    "preop_transfer_count", "preop_n_units",
}


# ------------------------------------------------------------------ prep
print("[plos-v2] loading...", flush=True)
cpt_map = _load_cpt_map()
raw = load_raw()
enc_nodupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
merged = (raw.encounters.merge(enc_nodupes, on="LogID", how="inner")
          .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]],
                 on="LogID", how="inner").reset_index(drop=True))
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)

# PLOS from A3
a3 = D._read_table(C.UNIT_EDGES_PARQUET, C.A3_UNIT_EDGES_COLUMNS).copy()
a3["LogID"] = a3["LogID"].astype(str)
a3["InTime"] = pd.to_datetime(a3["InTime"], errors="coerce")
a3["OutTime"] = pd.to_datetime(a3["OutTime"], errors="coerce")
los = a3.groupby("LogID").apply(
    lambda g: (g["OutTime"].max() - g["InTime"].min()).total_seconds() / 86400
).rename("los_days").reset_index()
merged = merged.merge(los, on="LogID", how="left")
cutoff = float(merged["los_days"].quantile(0.75))
merged["plos"] = (merged["los_days"] > cutoff).astype("Int64")
merged = merged.loc[merged["los_days"].notna()].reset_index(drop=True)

prov = build_provider_team_features(raw.enc_prov_edges, raw.prov_attrs,
                                    raw.encounters)
merged = merged.merge(prov, on="LogID", how="left")
for c in C.PROVIDER_FEATURE_COLUMNS:
    merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

feat_cols_full = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
                  + [c for c in C.CALENDAR_FEATURE_COLUMNS if c in merged.columns])
feat_cols = [c for c in feat_cols_full if c not in LEAKY_FEATS]
y_all = merged["plos"].astype(int).values
N = len(merged)
print(f"[plos-v2] N={N}, base={y_all.mean()*100:.2f}%, cutoff={cutoff:.2f} d,"
      f" {len(feat_cols)} clean feats", flush=True)


# ------------------------------------------------------------------ MedHG-PS A2
def train_medhgps_a2(tr, va, te):
    """Raw MedHG-PS ie-HGCN-style classifier on (encounter, provider) with A2
    edges. Trains a 2-layer bidirectional heterograph net + softmax head.
    Returns per-encounter positive-class probability."""
    import dgl
    edges = raw.enc_prov_edges.copy()
    edges["LogID"] = edges["LogID"].astype(str)
    edges["ProvID"] = edges["ProvID"].astype(str)
    enc_id = {l: i for i, l in enumerate(merged["LogID"])}
    prov_ids = sorted(edges["ProvID"].dropna().unique().tolist())
    pid = {p: i for i, p in enumerate(prov_ids)}
    e_src, e_dst = [], []
    for l, p in zip(edges["LogID"], edges["ProvID"]):
        i = enc_id.get(l); j = pid.get(p)
        if i is None or j is None or pd.isna(p): continue
        e_src.append(i); e_dst.append(j)
    if not e_src:
        return np.full(N, np.nan)
    g = dgl.heterograph({
        ("encounter", "treated_by", "provider"): (torch.tensor(e_src), torch.tensor(e_dst)),
        ("provider", "treats", "encounter"):     (torch.tensor(e_dst), torch.tensor(e_src)),
    }, num_nodes_dict={"encounter": N, "provider": len(prov_ids)}).to(dev)

    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    X_enc = apply_preprocess(merged[feat_cols], st).astype(np.float32)
    enc_feat = torch.tensor(X_enc, device=dev)
    prov_feat = torch.zeros(len(prov_ids), enc_feat.shape[1], device=dev)

    class HG(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin_e = nn.Linear(d, HID)
            self.lin_p = nn.Linear(d, HID)
            self.lin2  = nn.Linear(HID, HID)
            self.head  = nn.Linear(HID, 2)
        def forward(self, g, e_feat, p_feat):
            e = torch.relu(self.lin_e(e_feat))
            p = torch.relu(self.lin_p(p_feat))
            g.nodes["encounter"].data["h"] = e
            g.nodes["provider"].data["h"]  = p
            g.multi_update_all({
                "treated_by": (dgl.function.copy_u("h", "m"), dgl.function.mean("m", "h2")),
                "treats":     (dgl.function.copy_u("h", "m"), dgl.function.mean("m", "h2")),
            }, "sum")
            e = torch.relu(self.lin2(g.nodes["encounter"].data.get("h2", e)))
            return self.head(e)

    set_seed(SEED)
    net = HG(enc_feat.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    npos, nneg = float((y_all[tr] == 1).sum()), float((y_all[tr] == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)],
                     dtype=torch.float32, device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    Y = torch.tensor(y_all, device=dev, dtype=torch.long)
    tr_t = torch.tensor(tr, device=dev, dtype=torch.long)
    va_t = torch.tensor(va, device=dev, dtype=torch.long)

    best, state, pat = -1.0, None, PATIENCE
    for ep in range(MAX_EPOCHS):
        net.train(); opt.zero_grad()
        logits = net(g, enc_feat, prov_feat)
        loss = lf(logits[tr_t], Y[tr_t])
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            lv = net(g, enc_feat, prov_feat)
            pv = torch.softmax(lv[va_t], -1)[:, 1].cpu().numpy()
        try: vauc = roc_auc_score(y_all[va], pv)
        except ValueError: vauc = 0.5
        if vauc > best:
            best = vauc; pat = PATIENCE
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            pat -= 1
            if pat <= 0: break

    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        logits = net(g, enc_feat, prov_feat)
        p_all = torch.softmax(logits, -1)[:, 1].cpu().numpy()

    # per-fold isotonic calibration on the val slice, applied to test
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_all[va], y_all[va])
    return iso.transform(p_all[te]), best


# ------------------------------------------------------------------ run GNN
print("[plos-v2] training MedHG-PS raw ie-HGCN (A2-only)...", flush=True)
p_medhg = np.full(N, np.nan)
skf = StratifiedKFold(K, shuffle=True, random_state=SEED)
for fold, (tr_all, te) in enumerate(skf.split(np.zeros(N), y_all)):
    rng = np.random.default_rng(SEED + fold)
    tr_all = tr_all.copy(); rng.shuffle(tr_all)
    n_val = int(round(VAL_FRAC * N))
    val, tr = tr_all[:n_val], tr_all[n_val:]
    p_te, vauc = train_medhgps_a2(tr, val, te)
    p_medhg[te] = p_te
    print(f"[plos-v2] fold {fold+1}/{K}  val-AUROC={vauc:.3f}  "
          f"test-AUROC={roc_auc_score(y_all[te], p_te):.3f}", flush=True)


# ------------------------------------------------------------------ load cached
cached = np.load(OUT_DIR / "plos_table2_oof.npz")
assert np.array_equal(cached["y"], y_all), "y mismatch between cache and this run"
OOF = {
    "rf_clin":            cached["rf_clin"],
    "rf_prov":            cached["rf_prov"],
    "rf_gru_orders":      cached["rf_gru_orders"],
    "rf_gru_orders_prov": cached["rf_gru_orders_prov"],
    "rf_gnn_emb":         cached["rf_gnn_emb"],
    "medhgps_raw":        p_medhg,
}


# ------------------------------------------------------------------ metrics
def opt_threshold_by_f1(y, p):
    thrs = np.unique(np.round(p, 3))
    thrs = np.concatenate([[0.0], thrs, [1.0]])
    best_thr, best_f1 = 0.5, -1.0
    for t in thrs:
        yhat = (p >= t).astype(int)
        if yhat.sum() == 0 or yhat.sum() == len(yhat): continue
        f1 = f1_score(y, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr

def thr_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    f1  = f1_score(y, yhat, zero_division=0)
    pre = precision_score(y, yhat, zero_division=0)
    rec = recall_score(y, yhat, zero_division=0)
    spe = tn / max(tn + fp, 1)
    flag = (tp + fp) / len(y)
    return dict(f1=f1, precision=pre, recall=rec, specificity=spe, flag=flag,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))

def boot_ci_at_thr(y, p, thr, seed_base=100, n_boot=N_BOOT):
    def _f1(y, p): return f1_score(y, (p >= thr).astype(int), zero_division=0)
    def _pre(y, p): return precision_score(y, (p >= thr).astype(int), zero_division=0)
    def _rec(y, p): return recall_score(y, (p >= thr).astype(int), zero_division=0)
    f1_lo, f1_hi = _bootstrap_ci(y, p, _f1, n_boot=n_boot, seed=seed_base)
    pre_lo, pre_hi = _bootstrap_ci(y, p, _pre, n_boot=n_boot, seed=seed_base + 1)
    rec_lo, rec_hi = _bootstrap_ci(y, p, _rec, n_boot=n_boot, seed=seed_base + 2)
    return (f1_lo, f1_hi), (pre_lo, pre_hi), (rec_lo, rec_hi)


rows = []
for name, p in OOF.items():
    valid = ~np.isnan(p)
    yv, pv = y_all[valid], p[valid]
    au = roc_auc_score(yv, pv)
    ap = average_precision_score(yv, pv)
    br = brier_score_loss(yv, pv)
    au_ci = _bootstrap_ci(yv, pv, roc_auc_score, n_boot=N_BOOT, seed=0)
    ap_ci = _bootstrap_ci(yv, pv, average_precision_score, n_boot=N_BOOT, seed=1)
    thr = opt_threshold_by_f1(yv, pv)
    m = thr_metrics(yv, pv, thr)
    f1_ci, pre_ci, rec_ci = boot_ci_at_thr(yv, pv, thr, seed_base=100, n_boot=N_BOOT)
    rows.append(dict(
        model=name,
        AUROC=au, AUROC_lo=au_ci[0], AUROC_hi=au_ci[1],
        AUPRC=ap, AUPRC_lo=ap_ci[0], AUPRC_hi=ap_ci[1],
        Brier=br,
        opt_thr=thr,
        F1=m["f1"], F1_lo=f1_ci[0], F1_hi=f1_ci[1],
        precision=m["precision"], precision_lo=pre_ci[0], precision_hi=pre_ci[1],
        recall=m["recall"], recall_lo=rec_ci[0], recall_hi=rec_ci[1],
        specificity=m["specificity"], flag_rate=m["flag"],
        tp=m["tp"], fp=m["fp"], tn=m["tn"], fn=m["fn"],
        n=int(valid.sum()),
    ))

df = pd.DataFrame(rows)
csv_out = OUT_DIR / "plos_table2_v2_results.csv"
df.to_csv(csv_out, index=False)

# ------------------------------------------------------------------ report
print("\n=== PLOS Table 2 v2 (N={}, base={:.2f}%, F1-optimal threshold) ==="
      .format(N, y_all.mean()*100))
hdr = f"{'model':22s} {'AUROC (95% CI)':>22s} {'AUPRC (95% CI)':>22s} {'Brier':>7s}  {'thr':>5s}  {'F1 (95% CI)':>22s} {'Prec':>7s} {'Rec':>7s} {'Spec':>7s} {'Flag%':>6s}"
print(hdr)
for r in rows:
    print(f"{r['model']:22s} "
          f"{r['AUROC']:.3f} ({r['AUROC_lo']:.3f}-{r['AUROC_hi']:.3f})  "
          f"{r['AUPRC']:.3f} ({r['AUPRC_lo']:.3f}-{r['AUPRC_hi']:.3f})  "
          f"{r['Brier']:.4f}  "
          f"{r['opt_thr']:.3f}  "
          f"{r['F1']:.3f} ({r['F1_lo']:.3f}-{r['F1_hi']:.3f})  "
          f"{r['precision']:.3f} {r['recall']:.3f} {r['specificity']:.3f} "
          f"{r['flag_rate']*100:5.1f}%")

print(f"\n[plos-v2] wrote {csv_out}", flush=True)
print("DONE", flush=True)
