"""Swap the PLOS Table 3 sequence-only row from PRE-SURGERY ORDERS to
PRE-SURGERY CARE-UNIT SEQUENCE. Reuses the six non-sequence rows already
cached in artifacts/newdata/plos_table3_oof.npz (rf_clin, gnn_prov_rf,
hypergraph_prov_rf, sim_edges_rf, temporal_prov_rf, allfeat_rel_rf) and
retrains only the GRU sequence-only row on pre-surgery A3 unit-stays.

Pre-surgery filter matches plos_table23_units.py:
  surgery_start = min A3.InTime where UnitType == 'OR' per LogID,
                  fallback SurgeryDate midnight
  keep A3 rows where OutTime < surgery_start
  vocab = InstitutionType:UnitType

Same protocol as the rest of Table 3: 5-fold StratifiedKFold seed 42,
isotonic-calibrated pooled OOF, class-weighted CE, Adam, early stopping on
val AUROC, bootstrap n=2000 CIs.

Outputs overwrite artifacts/newdata/plos_table3_oof.npz and
plos_table3_results.csv with the units-swap.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

import medhg_ps.config as C
import medhg_ps.data as D
from medhg_ps.data import (add_calendar_features, apply_preprocess,
                           build_provider_team_features, fit_preprocess,
                           load_raw)
from medhg_ps.deploy import _load_cpt_map
from medhg_ps.evaluate import _bootstrap_ci
from medhg_ps.train import _resolve_device, set_seed

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
MAX_EPOCHS, PATIENCE = (10, 3) if SMOKE else (60, 8)
MAXLEN = 128
EMB_DIM, HID, NUMF = 24, 48, 5
N_BOOT = 200 if SMOKE else 2000
OUT_DIR = Path(C.PROJECT_ROOT) / "artifacts" / "newdata"
dev = _resolve_device(C.DEFAULTS_TRAIN.device)


# ---------------------------------------------------------------------------
# Load raw + derive PLOS (same as plos_table3.py header)
# ---------------------------------------------------------------------------
print("[t3u] loading raw + deriving PLOS...", flush=True)
cpt_map = _load_cpt_map()
raw = load_raw()
enc_nodupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
merged = (raw.encounters.merge(enc_nodupes, on="LogID", how="inner")
          .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]],
                 on="LogID", how="inner")
          .reset_index(drop=True))
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)

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
y_all = merged["plos"].astype(int).values
N = len(merged); base = y_all.mean()
print(f"[t3u] PLOS cutoff={cutoff:.2f} days  N={N:,}  base={base*100:.2f}%",
      flush=True)


# ---------------------------------------------------------------------------
# Pre-surgery A3 unit sequence (mirrors plos_table23_units.py)
# ---------------------------------------------------------------------------
a3["UnitType"] = a3["UnitType"].astype(str)
a3["InstitutionType"] = a3["InstitutionType"].astype(str)
or_start = (a3[a3["UnitType"].str.upper() == "OR"]
            .groupby("LogID")["InTime"].min())
sd_fallback = pd.to_datetime(merged.set_index("LogID")["SurgeryDate"],
                             errors="coerce")
surgery_start = or_start.reindex(merged["LogID"]).fillna(
    sd_fallback.reindex(merged["LogID"]))
sd_map_units = dict(zip(merged["LogID"], surgery_start.values))
a3["SurgeryStart"] = a3["LogID"].map(sd_map_units)
mask_pre = (a3["OutTime"].notna()
            & a3["SurgeryStart"].notna()
            & (a3["OutTime"] < a3["SurgeryStart"]))
a3_pre = a3.loc[mask_pre].copy()
n_pre_total = len(a3_pre)
print(f"[t3u] pre-surgery A3 unit-stays retained: {n_pre_total:,} of "
      f"{len(a3):,} ({n_pre_total/len(a3)*100:.1f}%)", flush=True)
per_case = a3_pre.groupby("LogID").size()
per_case_full = pd.Series(0, index=merged["LogID"])
per_case_full.update(per_case)
n_any = int((per_case_full > 0).sum())
med_len_any = int(per_case.median()) if len(per_case) else 0
print(f"[t3u] encounters with any pre-surgery unit-stays: {n_any}/{N} "
      f"({n_any/N*100:.1f}%); median seq len among those = {med_len_any}",
      flush=True)

a3_pre["Token"] = (a3_pre["InstitutionType"].str[:8].fillna("Unk")
                   + ":" + a3_pre["UnitType"].str[:8].fillna("Unk"))
a3_pre = a3_pre.sort_values(["LogID", "InTime"])
vocab = a3_pre["Token"].value_counts().index.tolist() if len(a3_pre) else []
T2I = {t: i for i, t in enumerate(vocab)}
PAD = max(len(vocab), 1)
print(f"[t3u] pre-op unit vocab={len(vocab)}", flush=True)

row_of = {l: i for i, l in enumerate(merged["LogID"])}
seq_idx = np.full((N, MAXLEN), PAD, dtype=np.int64)
seq_num = np.zeros((N, MAXLEN, NUMF), dtype=np.float32)
lengths = np.ones(N, dtype=np.int64)
for lid, grp in a3_pre.groupby("LogID", sort=False):
    r = row_of.get(lid)
    if r is None: continue
    grp = grp.tail(MAXLEN)
    toks = grp["Token"].tolist()
    hours = grp["Hours"].tolist()
    times = grp["InTime"].tolist()
    outs = grp["OutTime"].tolist()
    L = len(toks); lengths[r] = max(L, 1)
    for j in range(L):
        seq_idx[r, j] = T2I.get(toks[j], PAD)
        h = 0.0 if pd.isna(hours[j]) else max(float(hours[j]), 0.0)
        t = times[j]; hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        gap = 0.0
        if j > 0 and pd.notna(outs[j-1]) and pd.notna(times[j]):
            gap = max((times[j] - outs[j-1]).total_seconds() / 60.0, 0.0)
        seq_num[r, j] = [np.log1p(h),
                         np.log1p(gap),
                         (j + 1) / MAXLEN,
                         np.sin(2 * np.pi * hour / 24.0),
                         np.cos(2 * np.pi * hour / 24.0)]


# ---------------------------------------------------------------------------
# GRU + train
# ---------------------------------------------------------------------------
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


def train_gru_only(tr, va):
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
        pt = torch.softmax(net(Xi, Xn, Ln), -1)[:, 1].cpu().numpy()
    return pt, best


# ---------------------------------------------------------------------------
# 5-fold: retrain gru_units_only per fold, isotonic-calibrate on val split
# of train, predict on test fold
# ---------------------------------------------------------------------------
print("[t3u] 5-fold gru_units_only training...", flush=True)
skf = StratifiedKFold(K, shuffle=True, random_state=SEED)
p_units = np.full(N, np.nan)
rng = np.random.default_rng(SEED)
for fold, (tr_all, te) in enumerate(skf.split(np.zeros(N), y_all)):
    n_val = max(1, int(round(len(tr_all) * VAL_FRAC)))
    _perm = rng.permutation(tr_all)
    va = _perm[:n_val]; tr = _perm[n_val:]
    p_full, vauc = train_gru_only(tr, va)
    # isotonic-calibrate on val
    try:
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_full[va], y_all[va])
        p_units[te] = iso.transform(p_full[te])
    except Exception:
        p_units[te] = p_full[te]
    print(f"[t3u]   fold {fold+1}/{K}  val AUROC={vauc:.3f}  "
          f"test AUROC={roc_auc_score(y_all[te], p_units[te]):.3f}",
          flush=True)
assert not np.isnan(p_units).any()


# ---------------------------------------------------------------------------
# Merge with cached non-seq rows (from artifacts/newdata/plos_table3_orders_oof.npz)
# ---------------------------------------------------------------------------
prior = np.load(OUT_DIR / "plos_table3_orders_oof.npz")
KEEP = ["y", "rf_clin", "gnn_prov_rf", "hypergraph_prov_rf",
        "sim_edges_rf", "temporal_prov_rf", "allfeat_rel_rf"]
assert np.array_equal(prior["y"].astype(int), y_all), (
    "cached y differs from current y_all -- cohort mismatch")
new_oof = {k: prior[k] for k in KEEP}
new_oof["gru_units_only"] = p_units
np.savez(OUT_DIR / "plos_table3_oof.npz", **new_oof)
print(f"[t3u] wrote {OUT_DIR/'plos_table3_oof.npz'} with keys {list(new_oof)}",
      flush=True)


# ---------------------------------------------------------------------------
# Metrics: AUROC/AUPRC (CI), Brier, F1-opt thr + F1/prec/recall/spec/flag% (CI on F1)
# ---------------------------------------------------------------------------
MODEL_ORDER = ["rf_clin", "gnn_prov_rf", "hypergraph_prov_rf",
               "gru_units_only", "sim_edges_rf", "temporal_prov_rf",
               "allfeat_rel_rf"]

def _f1_opt_thr(y, p, n=201):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    j = int(np.nanargmax(f1[:-1]))
    return float(thr[j])

def _threshold_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tp = int(((yhat==1)&(y==1)).sum()); fp = int(((yhat==1)&(y==0)).sum())
    fn = int(((yhat==0)&(y==1)).sum()); tn = int(((yhat==0)&(y==0)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1); spec = tn/max(tn+fp,1)
    f1 = 2*prec*rec/max(prec+rec,1e-12); flag = (tp+fp)/len(y)
    return f1, prec, rec, spec, flag

def _boot_f1(y, p, thr, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed); n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yhat = (p[idx] >= thr).astype(int); yt = y[idx]
        tp = ((yhat==1)&(yt==1)).sum(); fp = ((yhat==1)&(yt==0)).sum()
        fn = ((yhat==0)&(yt==1)).sum()
        prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        vals.append(2*prec*rec/max(prec+rec,1e-12))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

rows = []
for m in MODEL_ORDER:
    p = new_oof[m]
    au = roc_auc_score(y_all, p)
    ap = average_precision_score(y_all, p)
    br = brier_score_loss(y_all, p)
    au_lo, au_hi = _bootstrap_ci(y_all, p, roc_auc_score, n_boot=N_BOOT, seed=0)
    ap_lo, ap_hi = _bootstrap_ci(y_all, p, average_precision_score, n_boot=N_BOOT, seed=1)
    thr = _f1_opt_thr(y_all, p)
    f1, prec, rec, spec, flag = _threshold_metrics(y_all, p, thr)
    f1_lo, f1_hi = _boot_f1(y_all, p, thr, n_boot=N_BOOT, seed=2)
    rows.append(dict(model=m, auroc=au, auroc_lo=au_lo, auroc_hi=au_hi,
                     auprc=ap, auprc_lo=ap_lo, auprc_hi=ap_hi, brier=br,
                     opt_thr=thr, f1=f1, precision=prec, recall=rec,
                     specificity=spec, flag_rate=flag,
                     f1_lo=f1_lo, f1_hi=f1_hi))

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "plos_table3_results.csv", index=False)
print(f"[t3u] wrote {OUT_DIR/'plos_table3_results.csv'}", flush=True)
print("\n=== PLOS Table 3 (units-swap) ===")
for r in rows:
    print(f"  {r['model']:22s}  AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f})  "
          f"AUPRC {r['auprc']:.3f} ({r['auprc_lo']:.3f}-{r['auprc_hi']:.3f})  "
          f"Brier {r['brier']:.4f}  F1 {r['f1']:.3f}  Prec {r['precision']:.3f}  "
          f"Rec {r['recall']:.3f}  Spec {r['specificity']:.3f}  Flag {r['flag_rate']*100:.1f}%")
print("DONE", flush=True)
