"""Attention-based hypergraph network (HGAT) vs tabular (5-fold CV).

The last hypergraph variant: instead of degree-normalized averaging (HGNN),
learn ATTENTION over hyperedge members in two stages -- node->hyperedge and
hyperedge->node (Bai et al. 2021, hypergraph attention). This is the principled
fix for hub dilution: attention can down-weight irrelevant members within a
huge (up to 10,019-patient) hyperedge instead of averaging over all of them.

Per fold we tune (validation-selected over a few configs) and evaluate:
  - hgat_raw : HGAT softmax head
  - hgat_enc : RF on tabular + HGAT encounter embedding
vs rf_tab / hgb_tab, on the full and a hub-pruned (cap-50) construction.

Fold-honest; writes nothing.   PYTHONPATH=. python analysis/cv_hypergraph_attn.py
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
MAX_EPOCHS, PATIENCE = 250, 20
CAPS = [None, 50]
CONFIGS = [dict(hid=32, dropout=0.3, lr=0.01),
           dict(hid=64, dropout=0.5, lr=0.005)]

print("[hga] loading + assembling...", flush=True)
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
print(f"[hga] cohort={N:,}  base={y.mean()*100:.2f}%  feats={len(feat_cols)}", flush=True)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


groups = []
for df, key in [(raw.enc_prov_edges, "ProvID"), (raw.enc_unit_edges, "DepartmentID")]:
    g = df[["LogID", key]].copy()
    g["LogID"] = _norm(g["LogID"]); g[key] = _norm(g[key]); g["r"] = g["LogID"].map(row_of)
    for _, sub in g.dropna(subset=["r", key]).groupby(key):
        rows = np.unique(sub["r"].astype(int).values)
        if len(rows) > 1:
            groups.append(rows)


def incidence(cap):
    ni, ei, eid = [], [], 0
    for rows in groups:
        if cap is not None and len(rows) > cap:
            continue
        ni.extend(rows.tolist()); ei.extend([eid] * len(rows)); eid += 1
    ni.extend(range(N)); ei.extend(range(eid, eid + N)); eid += N        # self-loops
    return (torch.tensor(ni, dtype=torch.long, device=dev),
            torch.tensor(ei, dtype=torch.long, device=dev), eid)


def seg_softmax(s, seg, n):
    m = s.new_full((n,), float("-inf")).scatter_reduce(0, seg, s, reduce="amax", include_self=False)
    s = (s - m[seg]).exp()
    denom = s.new_zeros(n).index_add(0, seg, s)
    return s / (denom[seg] + 1e-16)


class HGATLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.an = nn.Linear(d_out, 1, bias=False)     # node->edge attention
        self.ae = nn.Linear(d_out, 1, bias=False)     # edge->node attention

    def forward(self, X, ni, ei, M):
        Wx = self.W(X)
        xi = Wx[ni]                                   # member node feats
        a1 = seg_softmax(Fn.leaky_relu(self.an(xi).squeeze(-1)), ei, M)
        edge = Wx.new_zeros(M, Wx.shape[1]).index_add(0, ei, a1[:, None] * xi)
        ee = edge[ei]
        a2 = seg_softmax(Fn.leaky_relu(self.ae(ee).squeeze(-1)), ni, X.shape[0])
        return Wx.new_zeros(X.shape[0], Wx.shape[1]).index_add(0, ni, a2[:, None] * ee)


class HGAT(nn.Module):
    def __init__(self, d_in, hid, dropout):
        super().__init__()
        self.l1 = HGATLayer(d_in, hid); self.l2 = HGATLayer(hid, hid)
        self.bn = nn.BatchNorm1d(hid); self.cls = nn.Linear(hid, 2); self.drop = nn.Dropout(dropout)

    def forward(self, X, ni, ei, M):
        h = self.drop(Fn.elu(self.bn(self.l1(X, ni, ei, M))))
        emb = self.l2(h, ni, ei, M)
        return self.cls(emb), emb


def train_one(ni, ei, M, X, yt, tr_m, va_m, cfg):
    set_seed(SEED)
    m = HGAT(X.shape[1], cfg["hid"], cfg["dropout"]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    npos = float((yt[tr_m] == 1).sum()); nneg = float((yt[tr_m] == 0).sum())
    w = torch.tensor([(npos+nneg)/(2*max(nneg, 1)), (npos+nneg)/(2*max(npos, 1))], device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    best, state, pat = -1.0, None, PATIENCE
    yv = yt[va_m].cpu().numpy(); vam = va_m.cpu().numpy()
    for ep in range(MAX_EPOCHS):
        m.train(); logit, _ = m(X, ni, ei, M); loss = lf(logit[tr_m], yt[tr_m])
        opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(X, ni, ei, M)[0], -1)[:, 1].cpu().numpy()
        try: va = roc_auc_score(yv, p[vam])
        except ValueError: va = 0.5
        if va > best: best, state, pat = va, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0: break
    m.load_state_dict(state); return m, best


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


skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
yt = torch.tensor(y, dtype=torch.long, device=dev)
folds, tab = [], {"rf": [], "hgb": []}
for tr_all, te in skf.split(np.zeros(N), y):
    rng = np.random.default_rng(SEED + len(folds))
    tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nval = int(round(VAL_FRAC * N)); va, tr = tr_all[:nval], tr_all[nval:]
    tr_m = torch.zeros(N, dtype=torch.bool, device=dev); tr_m[tr] = True
    va_m = torch.zeros(N, dtype=torch.bool, device=dev); va_m[va] = True
    _, st = fit_preprocess(merged[feat_cols].loc[np.isin(np.arange(N), tr)].reset_index(drop=True), id_cols=[])
    X = torch.tensor(apply_preprocess(merged[feat_cols], st), dtype=torch.float32, device=dev)
    folds.append((tr, va, te, tr_m, va_m, X))
    tab["rf"].append(fit_eval(_rf(), merged[feat_cols], tr, te)[0])
    tab["hgb"].append(fit_eval(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                      l2_regularization=1.0, random_state=42), merged[feat_cols], tr, te)[0])
best_tab = np.maximum(tab["rf"], tab["hgb"])
print(f"[hga] tabular best AUROC mean {best_tab.mean():.3f}\n", flush=True)

for cap in CAPS:
    ni, ei, M = incidence(cap); tag = "full" if cap is None else f"cap{cap}"
    rau, eau, rap, eap = [], [], [], []
    for (tr, va, te, tr_m, va_m, X) in folds:
        bva, bm = -1, None
        for cfg in CONFIGS:
            m, v = train_one(ni, ei, M, X, yt, tr_m, va_m, cfg)
            if v > bva: bva, bm = v, m
        bm.eval()
        with torch.no_grad():
            lo, emb = bm(X, ni, ei, M)
            probs = torch.softmax(lo, -1)[:, 1].cpu().numpy(); emb = emb.cpu().numpy()
        rau.append(roc_auc_score(y[te], probs[te])); rap.append(average_precision_score(y[te], probs[te]))
        fr = pd.concat([merged[feat_cols].reset_index(drop=True),
                        pd.DataFrame(emb, columns=[f"hg_{i}" for i in range(emb.shape[1])])], axis=1)
        a_e, p_e = fit_eval(_rf(), fr, tr, te); eau.append(a_e); eap.append(p_e)
    rau, eau = np.array(rau), np.array(eau)
    print(f"[{tag:6s}] HGAT_raw {rau.mean():.3f}/{np.mean(rap):.3f}  "
          f"HGAT+RF {eau.mean():.3f}±{eau.std():.3f}/{np.mean(eap):.3f}±{np.std(eap):.3f}  || "
          f"raw-tab {(rau-best_tab).mean():+.3f}({int(((rau-best_tab)>0).sum())}/{K})  "
          f"enc-tab {(eau-best_tab).mean():+.3f}({int(((eau-best_tab)>0).sum())}/{K})", flush=True)
print(f"\nbest tabular = {best_tab.mean():.3f} AUROC.")
