"""Table 2 (manuscript-ready) on the PLOS outcome (top-25% length of stay).

Same 5-model panel and protocol as analysis/table2_orders_final.py:
  - 5-fold StratifiedKFold(seed=42)
  - Xtab: fit_preprocess/apply_preprocess (train-only) + one-hot CPT +
    StandardScaler (train-only)
  - RF (500 trees, min_samples_leaf=10, sqrt, class_weight='balanced')
  - Isotonic-calibrated (CalibratedClassifierCV, method='isotonic', cv=3),
    pooled OOF: each patient scored once on the fold that held them out
  - Bootstrap 95% CIs via medhg_ps.evaluate._bootstrap_ci (n_boot=2000)

PLOS-specific:
  - Outcome: total encounter LOS derived from A3 unit trajectory (max OutTime
    - min InTime). PLOS = top-25% (cutoff ~2.30 days).
  - N = 13,899 (encounters with derivable LOS).
  - CRITICAL leaky-feature drops:
      * Discharge Disposition (post-outcome)
      * # of Cardiac Arrest / CVA / Postop Unplanned Intubation (postop events)
      * preop_los_acute_hr, preop_los_intensive_hr, preop_los_intermediate_hr
        (pre-op portion of total LOS = mechanically part of outcome)
      * preop_transfer_count, preop_n_units (encounter trajectory)
  - Order-sequence GRU: filter to PRE-SURGERY orders only
    (OrderTime < A1.SurgeryDate midnight). 85% of encounters have zero
    pre-surgery orders under this filter (median = 0), so the embedding
    largely encodes "was there a pre-op admission" -- still partly leaky
    for PLOS since pre-op LOS contributes to total LOS. Reported with the
    caveat.
  - GNN embedding: heterograph restricted to A2 (encounter-provider)
    only, not A3 (unit trajectory), to avoid leaking trajectory into
    the GNN encoder.
"""
from __future__ import annotations

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
import medhg_ps.data as D
from medhg_ps.data import (apply_preprocess, fit_preprocess, load_raw,
                           add_calendar_features, load_order_sequence,
                           collapse_order_runs, build_provider_team_features)
from medhg_ps.deploy import _load_cpt_map
from medhg_ps.evaluate import _bootstrap_ci
from medhg_ps.train import set_seed, _resolve_device

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
MAX_EPOCHS, PATIENCE = (10, 3) if SMOKE else (80, 10)
MAXLEN = 128
EMB_DIM, HID, NUMF = 24, 48, 5
N_BOOT = 200 if SMOKE else 2000
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


def RF():
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# assemble frame + derive PLOS from A3 + provider block + pre-surgery orders
# ---------------------------------------------------------------------------
print(f"[plos] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
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

# Derive LOS + PLOS from A3
a3 = D._read_table(C.UNIT_EDGES_PARQUET, C.A3_UNIT_EDGES_COLUMNS).copy()
a3["LogID"] = a3["LogID"].astype(str)
a3["InTime"]  = pd.to_datetime(a3["InTime"], errors="coerce")
a3["OutTime"] = pd.to_datetime(a3["OutTime"], errors="coerce")
los = a3.groupby("LogID").apply(
    lambda g: (g["OutTime"].max() - g["InTime"].min()).total_seconds() / 86400
).rename("los_days").reset_index()
merged = merged.merge(los, on="LogID", how="left")
cutoff = float(merged["los_days"].quantile(0.75))
merged["plos"] = (merged["los_days"] > cutoff).astype("Int64")
before = len(merged)
merged = merged.loc[merged["los_days"].notna()].reset_index(drop=True)
print(f"[plos] PLOS cutoff = {cutoff:.2f} days; dropped {before - len(merged)} "
      f"encounters with no derivable LOS; N={len(merged)}", flush=True)

# Provider block
prov = build_provider_team_features(raw.enc_prov_edges, raw.prov_attrs,
                                    raw.encounters)
merged = merged.merge(prov, on="LogID", how="left")
for c in C.PROVIDER_FEATURE_COLUMNS:
    merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

feat_cols_full = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
                  + [c for c in C.CALENDAR_FEATURE_COLUMNS if c in merged.columns])
feat_cols = [c for c in feat_cols_full if c not in LEAKY_FEATS]
prov_cols = [c for c in C.PROVIDER_FEATURE_COLUMNS if c in merged.columns]
dropped = [c for c in feat_cols_full if c in LEAKY_FEATS]
print(f"[plos] dropped {len(dropped)} leaky feats: {dropped}", flush=True)
print(f"[plos] using {len(feat_cols)} clean feats + {len(prov_cols)} prov feats "
      f"+ 1-hot CPT", flush=True)

y_all = merged["plos"].astype(int).values
N = len(merged)
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)

# -------- Pre-surgery UNIT-SEQUENCE only (A3 trajectory truncated) ---------
# Surgery start = InTime of first OR row in A3 per LogID; fallback to
# SurgeryDate at midnight. Then filter A3 to unit-stays that ENDED strictly
# before surgery start (OutTime < surgery_start). This retains only pre-op
# admission trajectory (which units the patient passed through pre-surgery),
# without leaking post-op trajectory that mechanically defines LOS.
a3_full = a3.copy()  # a3 already loaded above with datetime columns
a3_full["UnitType"] = a3_full["UnitType"].astype(str)
a3_full["InstitutionType"] = a3_full["InstitutionType"].astype(str)
or_start = (a3_full[a3_full["UnitType"].str.upper() == "OR"]
            .groupby("LogID")["InTime"].min())
sd_fallback = pd.to_datetime(merged.set_index("LogID")["SurgeryDate"],
                             errors="coerce")
surgery_start = or_start.reindex(merged["LogID"]).fillna(
    sd_fallback.reindex(merged["LogID"]))
sd_map_units = dict(zip(merged["LogID"], surgery_start.values))
a3_full["SurgeryStart"] = a3_full["LogID"].map(sd_map_units)
mask_pre = (a3_full["OutTime"].notna()
            & a3_full["SurgeryStart"].notna()
            & (a3_full["OutTime"] < a3_full["SurgeryStart"]))
n_pre_total = int(mask_pre.sum())
a3_pre = a3_full.loc[mask_pre].copy()
print(f"[plos] pre-surgery A3 unit-stays retained: {n_pre_total:,} of "
      f"{len(a3_full):,} ({n_pre_total/len(a3_full)*100:.1f}%)", flush=True)
per_case = a3_pre.groupby("LogID").size()
per_case_full = pd.Series(0, index=merged["LogID"])
per_case_full.update(per_case)
n_any = int((per_case_full > 0).sum())
med_len_any = int(per_case.median()) if len(per_case) else 0
print(f"[plos] encounters with any pre-surgery unit-stays: "
      f"{n_any}/{N} ({n_any/N*100:.1f}%); median pre-surgery seq len "
      f"among those with any = {med_len_any}", flush=True)

# vocabulary = InstitutionType + UnitType tuple (richer than UnitType alone)
a3_pre["Token"] = (a3_pre["InstitutionType"].str[:8].fillna("Unk")
                   + ":" + a3_pre["UnitType"].str[:8].fillna("Unk"))
a3_pre = a3_pre.sort_values(["LogID", "InTime"])
vocab = a3_pre["Token"].value_counts().index.tolist() if len(a3_pre) else []
T2I = {t: i for i, t in enumerate(vocab)}
PAD = max(len(vocab), 1)

row_of = {l: i for i, l in enumerate(merged["LogID"])}
seq_idx = np.full((N, MAXLEN), PAD, dtype=np.int64)
seq_num = np.zeros((N, MAXLEN, NUMF), dtype=np.float32)
lengths = np.ones(N, dtype=np.int64)
for lid, grp in a3_pre.groupby("LogID", sort=False):
    r = row_of.get(lid)
    if r is None: continue
    grp = grp.tail(MAXLEN)
    toks  = grp["Token"].tolist()
    hours = grp["Hours"].tolist()
    times = grp["InTime"].tolist()
    outs  = grp["OutTime"].tolist()
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

print(f"[plos] cohort={N:,}  PLOS base={y_all.mean()*100:.2f}%  "
      f"pre-op unit vocab={len(vocab)}  median seq len (any-case)={int(np.median(lengths))}",
      flush=True)


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
    return emb, best


# --------------------------- GNN embedding on A2 only -----------------------
# Build a simple heterograph over encounter+provider nodes (from A2 edges)
# and train an ie-HGCN-style embedder per fold. A3 (units) intentionally
# excluded to avoid trajectory leakage into GNN.

def build_gnn_emb_a2_only(tr, va):
    """Simple 2-layer heterogeneous encoder over (encounter, provider) with
    A2 edges. Returns a per-encounter embedding [N, HID]."""
    try:
        import dgl
    except Exception:
        return np.zeros((N, HID), dtype=np.float32), 0.5
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
        return np.zeros((N, HID), dtype=np.float32), 0.5
    g = dgl.heterograph({
        ("encounter", "treated_by", "provider"): (torch.tensor(e_src), torch.tensor(e_dst)),
        ("provider", "treats", "encounter"):     (torch.tensor(e_dst), torch.tensor(e_src)),
    }, num_nodes_dict={"encounter": N, "provider": len(prov_ids)}).to(dev)
    # simple encounter/provider features
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    enc_feat = torch.tensor(apply_preprocess(merged[feat_cols], st).astype(np.float32),
                            device=dev)
    prov_feat = torch.zeros(len(prov_ids), enc_feat.shape[1], device=dev)
    # 2 mean-aggregation layers via bidirectional message passing
    class HG(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin_e = nn.Linear(d, HID); self.lin_p = nn.Linear(d, HID)
            self.lin2  = nn.Linear(HID, HID)
            self.head  = nn.Linear(HID, 2)
        def forward(self, g, e_feat, p_feat):
            e = torch.relu(self.lin_e(e_feat))
            p = torch.relu(self.lin_p(p_feat))
            g.nodes["encounter"].data["h"] = e
            g.nodes["provider"].data["h"]  = p
            g.multi_update_all({
                "treated_by": (dgl.function.copy_u("h","m"), dgl.function.mean("m","h2")),
                "treats":     (dgl.function.copy_u("h","m"), dgl.function.mean("m","h2")),
            }, "sum")
            e = torch.relu(self.lin2(g.nodes["encounter"].data.get("h2", e)))
            return e, self.head(e)
    net = HG(enc_feat.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    npos, nneg = float((y_all[tr] == 1).sum()), float((y_all[tr] == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)],
                     dtype=torch.float32, device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    Y = torch.tensor(y_all, device=dev)
    tr_t = torch.tensor(tr, device=dev, dtype=torch.long)
    va_t = torch.tensor(va, device=dev, dtype=torch.long)
    best, state, pat = -1.0, None, PATIENCE
    for ep in range(min(MAX_EPOCHS, 30)):
        net.train(); opt.zero_grad()
        _, logits = net(g, enc_feat, prov_feat)
        loss = lf(logits[tr_t], Y[tr_t])
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            _, lv = net(g, enc_feat, prov_feat)
            pv = torch.softmax(lv[va_t], -1)[:, 1].cpu().numpy()
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
        e_emb, _ = net(g, enc_feat, prov_feat)
        emb = e_emb.cpu().numpy()
    return emb, best


def build_design(tr, use_prov=False, use_gru=False, use_gnn=False,
                 gru_emb=None, gnn_emb=None):
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True),
                           id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    blocks = [Xtab, ohe.transform(cpt_arr)]
    if use_prov:
        blocks.append(merged[prov_cols].values.astype(float))
    if use_gru and gru_emb is not None:
        sc = StandardScaler().fit(gru_emb[tr])
        blocks.append(sc.transform(gru_emb))
    if use_gnn and gnn_emb is not None:
        sc = StandardScaler().fit(gnn_emb[tr])
        blocks.append(sc.transform(gnn_emb))
    X = np.hstack(blocks)
    return StandardScaler().fit(X[tr]).transform(X)


def rf_calibrated_oof(X, tr, te):
    est = CalibratedClassifierCV(RF(), method="isotonic", cv=3).fit(X[tr], y_all[tr])
    return est.predict_proba(X[te])[:, 1]


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
MODELS = ["rf_clin", "rf_prov", "rf_gru_units", "rf_gru_units_prov", "rf_gnn_emb"]
oof = {m: np.full(N, np.nan) for m in MODELS}

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y_all)):
    rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]

    print(f"[plos] fold {fi+1}/{K} training GRU + GNN...", flush=True)
    gru_emb, vb_gru = train_gru(tr, va)
    gnn_emb, vb_gnn = build_gnn_emb_a2_only(tr, va)

    X_clin  = build_design(tr)
    X_prov  = build_design(tr, use_prov=True)
    X_gru   = build_design(tr, use_gru=True,  gru_emb=gru_emb)
    X_gp    = build_design(tr, use_prov=True, use_gru=True, gru_emb=gru_emb)
    X_gnn   = build_design(tr, use_gnn=True,  gnn_emb=gnn_emb)

    oof["rf_clin"][te]            = rf_calibrated_oof(X_clin, tr, te)
    oof["rf_prov"][te]            = rf_calibrated_oof(X_prov, tr, te)
    oof["rf_gru_units"][te]      = rf_calibrated_oof(X_gru,  tr, te)
    oof["rf_gru_units_prov"][te] = rf_calibrated_oof(X_gp,   tr, te)
    oof["rf_gnn_emb"][te]         = rf_calibrated_oof(X_gnn,  tr, te)

    line = f"[plos] fold {fi+1}/{K}  gru_val={vb_gru:.3f} gnn_val={vb_gnn:.3f} |"
    for m in MODELS:
        p = oof[m][te]
        line += f" {m} {roc_auc_score(y_all[te], p):.3f}/{average_precision_score(y_all[te], p):.3f} "
    print(line, flush=True)

# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
assert all(not np.isnan(oof[m]).any() for m in MODELS)
np.savez(OUT_DIR / "plos_units_oof.npz", y=y_all, **{m: oof[m] for m in MODELS})

rows = []
print(f"\n=== PLOS Table 2 (calibrated pooled OOF; N={N:,}; base "
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

print("\n=== paired deltas vs rf_clin (per fold) ===")
skf2 = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
folds = list(skf2.split(np.zeros(N), y_all))
for m in ["rf_prov", "rf_gru_units", "rf_gru_units_prov", "rf_gnn_emb"]:
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

# -------- F1-optimal threshold metrics + bootstrap CIs -----------------
def _threshold_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    flag = (tp + fp) / max(len(y), 1)
    return f1, prec, rec, spec, flag

def _f1_opt_thr(y, p, n=201):
    ths = np.linspace(0.01, 0.99, n)
    f1s = [_threshold_metrics(y, p, t)[0] for t in ths]
    return float(ths[int(np.argmax(f1s))])

def _boot_f1(y, p, thr, n_boot, seed):
    rng = np.random.default_rng(seed)
    vs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        vs.append(_threshold_metrics(y[idx], p[idx], thr)[0])
    lo, hi = np.quantile(vs, [0.025, 0.975])
    return float(lo), float(hi)

print(f"\n=== PLOS Table 2 threshold metrics (F1-optimal threshold on pooled OOF) ===")
thr_rows = []
for r in rows:
    m = r["model"]; p = oof[m]
    thr = _f1_opt_thr(y_all, p)
    f1, prec, rec, spec, flag = _threshold_metrics(y_all, p, thr)
    f1_lo, f1_hi = _boot_f1(y_all, p, thr, N_BOOT, 42)
    r.update(dict(opt_thr=thr, f1=f1, f1_lo=f1_lo, f1_hi=f1_hi,
                  precision=prec, recall=rec, specificity=spec, flag_rate=flag))
    thr_rows.append(r)
    print(f"  {m:22s} thr {thr:.3f}  F1 {f1:.3f} ({f1_lo:.3f}-{f1_hi:.3f})  "
          f"prec {prec:.3f}  rec {rec:.3f}  spec {spec:.3f}  flag% {flag*100:.1f}")

pd.DataFrame(thr_rows).to_csv(OUT_DIR / "plos_units_results.csv", index=False)
print(f"\n[plos] saved OOF -> {OUT_DIR/'plos_units_oof.npz'}")
print(f"[plos] saved results -> {OUT_DIR/'plos_units_results.csv'}")
