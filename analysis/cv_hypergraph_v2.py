"""Thorough hypergraph test: tuned HGNN x hub-robust constructions (5-fold CV).

Addresses the shallow first pass. Three axes:
  - TUNING: per fold, train several HGNN configs and select the best on
    validation AUROC (fold-honest tuning).
  - CONSTRUCTION: provider+unit hyperedges with a size CAP -- drop hyperedges
    with > cap members (the high-volume hubs that dilute signal). caps tested:
    None (full), 500, 50, 10. Self-loops always kept.
  - both compared against the tabular baselines on identical folds.

Reports, per construction, the tuned HGNN (raw head and +RF embedding) vs the
best tabular model. Writes nothing.

    PYTHONPATH=. python analysis/cv_hypergraph_v2.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (load_raw, fit_preprocess, apply_preprocess,
                           build_preop_trajectory_features, add_calendar_features)
from medhg_ps.train import set_seed, _resolve_device

K, SEED, VAL_FRAC = 5, 42, 0.10
MAX_EPOCHS, PATIENCE = 300, 25
CAPS = [None, 500, 50, 10]                                   # hyperedge size caps
CONFIGS = [dict(hid=32, dropout=0.3, lr=0.01),              # tuned per fold by val AUROC
           dict(hid=64, dropout=0.5, lr=0.02),
           dict(hid=64, dropout=0.3, lr=0.005)]

print("[hg2] loading + assembling...", flush=True)
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
y = merged["ReadmittedWithin30Days"].astype(int).values
N = len(merged)
row_of = {lid: i for i, lid in enumerate(merged["LogID"])}
dev = _resolve_device(C.DEFAULTS_TRAIN.device)
print(f"[hg2] cohort={N:,}  base={y.mean()*100:.2f}%  feats={len(feat_cols)}", flush=True)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


# collect candidate groups once (list of node-index arrays)
groups = []
for df, key in [(raw.enc_prov_edges, "ProvID"), (raw.enc_unit_edges, "DepartmentID")]:
    g = df[["LogID", key]].copy()
    g["LogID"] = _norm(g["LogID"]); g[key] = _norm(g[key])
    g["r"] = g["LogID"].map(row_of)
    g = g.dropna(subset=["r", key])
    for _, sub in g.groupby(key):
        rows = np.unique(sub["r"].astype(int).values)
        if len(rows) > 1:
            groups.append(rows)
sizes = np.array([len(r) for r in groups])
print(f"[hg2] {len(groups):,} provider/unit groups; size median={int(np.median(sizes))} "
      f"max={sizes.max()} >500={(sizes>500).sum()} >50={(sizes>50).sum()}", flush=True)


class HyperOp:
    """Sparse HGNN propagation operator for a given hyperedge size cap."""
    def __init__(self, cap):
        ni, ei, eid = [], [], 0
        for rows in groups:
            if cap is not None and len(rows) > cap:
                continue
            ni.extend(rows.tolist()); ei.extend([eid] * len(rows)); eid += 1
        kept = eid
        ni.extend(range(N)); ei.extend(range(eid, eid + N)); eid += N   # self-loops
        idx = torch.tensor([ni, ei], dtype=torch.long)
        H = torch.sparse_coo_tensor(idx, torch.ones(len(ni)), size=(N, eid)).coalesce()
        self.kept = kept
        self.H = H.to(dev); self.Ht = H.transpose(0, 1).coalesce().to(dev)
        self.Dvi = torch.sparse.sum(H, 1).to_dense().clamp(min=1).pow(-0.5).to(dev)
        self.Dei = torch.sparse.sum(H, 0).to_dense().clamp(min=1).pow(-1).to(dev)

    def prop(self, X):
        x = X * self.Dvi[:, None]
        x = torch.sparse.mm(self.Ht, x); x = x * self.Dei[:, None]
        x = torch.sparse.mm(self.H, x)
        return x * self.Dvi[:, None]


class HGNN(nn.Module):
    def __init__(self, op, d_in, hid, dropout):
        super().__init__()
        self.op = op
        self.t1 = nn.Linear(d_in, hid); self.t2 = nn.Linear(hid, hid)
        self.bn = nn.BatchNorm1d(hid); self.cls = nn.Linear(hid, 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, X):
        h = self.op.prop(self.t1(X)); h = self.drop(Fn.relu(self.bn(h)))
        emb = self.op.prop(self.t2(h))
        return self.cls(emb), emb


def train_one(op, X, yt, tr_m, va_m, cfg):
    set_seed(SEED)
    m = HGNN(op, X.shape[1], cfg["hid"], cfg["dropout"]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    npos = float((yt[tr_m] == 1).sum()); nneg = float((yt[tr_m] == 0).sum())
    w = torch.tensor([(npos+nneg)/(2*max(nneg, 1)), (npos+nneg)/(2*max(npos, 1))], device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    best, state, pat = -1.0, None, PATIENCE
    yv = yt[va_m].cpu().numpy(); vam = va_m.cpu().numpy()
    for ep in range(MAX_EPOCHS):
        m.train(); logit, _ = m(X); loss = lf(logit[tr_m], yt[tr_m])
        opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(X)[0], -1)[:, 1].cpu().numpy()
        try: va = roc_auc_score(yv, p[vam])
        except ValueError: va = 0.5
        if va > best: best, state, pat = va, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0: break
    m.load_state_dict(state)
    return m, best


def _pre(X):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer([
        ("c", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ("n", Pipeline([("i", SimpleImputer(strategy="mean")), ("s", StandardScaler())]), num)])


def fit_eval(est, X, tr, te):
    pipe = Pipeline([("p", _pre(X)), ("c", est)]); pipe.fit(X.iloc[tr], y[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


def _rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


# ---- precompute folds: splits, per-fold features, tabular baselines -----
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
yt = torch.tensor(y, dtype=torch.long, device=dev)
folds = []
tab = {"rf": [], "hgb": []}
for tr_all, te in skf.split(np.zeros(N), y):
    rng = np.random.default_rng(SEED + len(folds))
    tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nval = int(round(VAL_FRAC * N)); va, tr = tr_all[:nval], tr_all[nval:]
    tr_m = torch.zeros(N, dtype=torch.bool, device=dev); tr_m[tr] = True
    va_m = torch.zeros(N, dtype=torch.bool, device=dev); va_m[va] = True
    _, st = fit_preprocess(merged[feat_cols].loc[np.isin(np.arange(N), tr)].reset_index(drop=True), id_cols=[])
    X = torch.tensor(apply_preprocess(merged[feat_cols], st), dtype=torch.float32, device=dev)
    folds.append((tr, va, te, tr_m, va_m, X))
    a_r, _ = fit_eval(_rf(), merged[feat_cols], tr, te)
    a_h, _ = fit_eval(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                      l2_regularization=1.0, random_state=42), merged[feat_cols], tr, te)
    tab["rf"].append(a_r); tab["hgb"].append(a_h)
best_tab = np.maximum(tab["rf"], tab["hgb"])
print(f"[hg2] tabular best AUROC per fold: {np.round(best_tab,3).tolist()} "
      f"(mean {best_tab.mean():.3f})\n", flush=True)

# ---- sweep constructions x tuned HGNN -----------------------------------
for cap in CAPS:
    op = HyperOp(cap); tag = "full" if cap is None else f"cap{cap}"
    raw_au, enc_au, raw_ap, enc_ap = [], [], [], []
    for fi, (tr, va, te, tr_m, va_m, X) in enumerate(folds):
        # tune: pick config with best validation AUROC
        best_va, best_m = -1, None
        for cfg in CONFIGS:
            m, vauroc = train_one(op, X, yt, tr_m, va_m, cfg)
            if vauroc > best_va: best_va, best_m = vauroc, m
        best_m.eval()
        with torch.no_grad():
            lo, emb = best_m(X)
            probs = torch.softmax(lo, -1)[:, 1].cpu().numpy(); emb = emb.cpu().numpy()
        raw_au.append(roc_auc_score(y[te], probs[te])); raw_ap.append(average_precision_score(y[te], probs[te]))
        fr = pd.concat([merged[feat_cols].reset_index(drop=True),
                        pd.DataFrame(emb, columns=[f"hg_{i}" for i in range(emb.shape[1])])], axis=1)
        a_e, p_e = fit_eval(_rf(), fr, tr, te)
        enc_au.append(a_e); enc_ap.append(p_e)
    raw_au, enc_au = np.array(raw_au), np.array(enc_au)
    dr, de = raw_au - best_tab, enc_au - best_tab
    print(f"[{tag:7s}] kept {op.kept:5d} hyperedges | "
          f"HGNN_raw {raw_au.mean():.3f}/{np.mean(raw_ap):.3f}  "
          f"HGNN+RF {enc_au.mean():.3f}±{enc_au.std():.3f}/{np.mean(enc_ap):.3f}±{np.std(enc_ap):.3f}  || "
          f"raw-tab {dr.mean():+.3f}({int((dr>0).sum())}/{K})  "
          f"enc-tab {de.mean():+.3f}({int((de>0).sum())}/{K})", flush=True)

print(f"\nbest tabular = {best_tab.mean():.3f} AUROC (RF/HGB). "
      f"A construction 'wins' only if enc-tab or raw-tab > 0 in most folds.")
