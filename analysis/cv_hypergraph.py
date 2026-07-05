"""Does a HYPERGRAPH neural network beat tabular? (5-fold CV)

Tests the higher-order relational hypothesis the pairwise GNN could not exploit.
We build a hypergraph on encounter nodes where each hyperedge groups all
encounters that share a provider (from A2) or a care unit (from A3), plus
self-loops, and run a Feng et al. (2019) HGNN:

    X' = sigma( Dv^-1/2 H W De^-1 H^T Dv^-1/2 (X Theta) )

applied without materializing the NxN operator. Per identical fold we compare:
  - hgnn_raw : HGNN softmax head
  - hgnn_enc : RF on tabular + HGNN encounter embedding
  - rf_tab / hgb_tab : tabular baselines (fair bar)

Fold-honest: encounter preprocessing fit on train only; HGNN retrained per fold;
embeddings re-extracted per fold. Reuses run.py's assembly. Writes nothing.

    PYTHONPATH=. python analysis/cv_hypergraph.py
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
from medhg_ps.data import (
    load_raw, fit_preprocess, apply_preprocess,
    build_preop_trajectory_features, add_calendar_features,
)
from medhg_ps.train import set_seed, _resolve_device

K, SEED, VAL_FRAC = 5, 42, 0.10
HID, DROPOUT, LR, L2 = 32, 0.3, 0.01, 1e-4
MAX_EPOCHS, PATIENCE = 300, 25

# ---- assemble merged_all ONCE (mirrors run.py) ----------------------
print("[hg] loading + assembling...", flush=True)
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

nsqip = [c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
derived = [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged.columns]
feat_cols = nsqip + derived
y = merged["ReadmittedWithin30Days"].astype(int).values
N = len(merged)
row_of = {lid: i for i, lid in enumerate(merged["LogID"])}
print(f"[hg] cohort={N:,}  base={y.mean()*100:.2f}%  feats={len(feat_cols)}", flush=True)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


# ---- build hypergraph incidence H (N nodes x M hyperedges) ----------
# Hyperedges: one per provider (A2), one per unit (A3), plus N self-loops.
node_idx, edge_idx, eid = [], [], 0


def add_group(df, key):
    global eid
    g = df.dropna(subset=[key]).copy()
    g["r"] = g["LogID"].map(row_of)
    g = g.dropna(subset=["r"])
    for _, sub in g.groupby(key):
        rows = sub["r"].astype(int).unique()
        if len(rows) == 0:
            continue
        node_idx.extend(rows.tolist()); edge_idx.extend([eid] * len(rows)); eid += 1


a2 = raw.enc_prov_edges[["LogID", "ProvID"]].copy()
a2["LogID"] = _norm(a2["LogID"]); a2["ProvID"] = _norm(a2["ProvID"])
add_group(a2, "ProvID")
a3 = raw.enc_unit_edges[["LogID", "DepartmentID"]].copy()
a3["LogID"] = _norm(a3["LogID"]); a3["DepartmentID"] = _norm(a3["DepartmentID"])
add_group(a3, "DepartmentID")
n_group_edges = eid
# self-loops: each node its own hyperedge (keeps isolated nodes informative)
node_idx.extend(range(N)); edge_idx.extend(range(eid, eid + N)); eid += N
M = eid
print(f"[hg] hyperedges: {n_group_edges:,} provider/unit + {N:,} self-loops = {M:,}; "
      f"incidences={len(node_idx):,}", flush=True)

idx = torch.tensor([node_idx, edge_idx], dtype=torch.long)
val = torch.ones(len(node_idx), dtype=torch.float32)
H = torch.sparse_coo_tensor(idx, val, size=(N, M)).coalesce()
Ht = H.transpose(0, 1).coalesce()
Dv = torch.sparse.sum(H, dim=1).to_dense().clamp(min=1.0)   # node degree
De = torch.sparse.sum(H, dim=0).to_dense().clamp(min=1.0)   # hyperedge degree
Dv_is = Dv.pow(-0.5)
De_i = De.pow(-1.0)


def propagate(X):
    x = X * Dv_is[:, None]
    x = torch.sparse.mm(Ht, x)
    x = x * De_i[:, None]
    x = torch.sparse.mm(H, x)
    return x * Dv_is[:, None]


class HGNN(nn.Module):
    def __init__(self, d_in, hid):
        super().__init__()
        self.t1 = nn.Linear(d_in, hid)
        self.t2 = nn.Linear(hid, hid)
        self.bn = nn.BatchNorm1d(hid)
        self.cls = nn.Linear(hid, 2)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, X):
        h = propagate(self.t1(X)); h = Fn.relu(self.bn(h)); h = self.drop(h)
        emb = propagate(self.t2(h))            # encounter embedding
        return self.cls(emb), emb


dev = _resolve_device(C.DEFAULTS_TRAIN.device)
H = H.to(dev); Ht = Ht.to(dev); Dv_is = Dv_is.to(dev); De_i = De_i.to(dev)


def train_hgnn(X, ytr_t, tr_m, va_m):
    set_seed(SEED)
    model = HGNN(X.shape[1], HID).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    n_pos = float((ytr_t[tr_m] == 1).sum()); n_neg = float((ytr_t[tr_m] == 0).sum())
    w = torch.tensor([(n_pos + n_neg) / (2 * max(n_neg, 1)),
                      (n_pos + n_neg) / (2 * max(n_pos, 1))], device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    best, best_state, patience = -1.0, None, PATIENCE
    yv = ytr_t[va_m].cpu().numpy()
    for ep in range(MAX_EPOCHS):
        model.train()
        logits, _ = model(X)
        loss = lossf(logits[tr_m], ytr_t[tr_m])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(X)[0], -1)[:, 1].cpu().numpy()
        try:
            va = roc_auc_score(yv, p[va_m.cpu().numpy()])
        except ValueError:
            va = 0.5
        if va > best:
            best, best_state, patience = va, {k: v.detach().cpu().clone()
                                              for k, v in model.state_dict().items()}, PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    return model, ep


def _pre(X):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    return ColumnTransformer([
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ("num", Pipeline([("i", SimpleImputer(strategy="mean")), ("s", StandardScaler())]), num)])


def fit_eval(est, X, tr, te):
    pipe = Pipeline([("pre", _pre(X)), ("clf", est)]); pipe.fit(X.iloc[tr], y[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


def _rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                                  class_weight="balanced", random_state=42, n_jobs=-1)


def _hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=42)


skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
MODELS = ["hgnn_raw", "hgnn_enc", "rf_tab", "hgb_tab"]
res = {m: {"auroc": [], "auprc": []} for m in MODELS}
yt = torch.tensor(y, dtype=torch.long, device=dev)

for fold, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
    rng = np.random.default_rng(SEED + fold)
    tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nval = int(round(VAL_FRAC * N))
    va, tr = tr_all[:nval], tr_all[nval:]
    tr_m = torch.zeros(N, dtype=torch.bool, device=dev); tr_m[tr] = True
    va_m = torch.zeros(N, dtype=torch.bool, device=dev); va_m[va] = True

    fa = merged[feat_cols].copy()
    Xtr_only, st = fit_preprocess(fa.loc[np.isin(np.arange(N), tr)].reset_index(drop=True), id_cols=[])
    X = torch.tensor(apply_preprocess(fa, st), dtype=torch.float32, device=dev)

    model, ep = train_hgnn(X, yt, tr_m, va_m)
    model.eval()
    with torch.no_grad():
        logits, emb = model(X)
        probs = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        emb = emb.cpu().numpy()
    res["hgnn_raw"]["auroc"].append(roc_auc_score(y[te], probs[te]))
    res["hgnn_raw"]["auprc"].append(average_precision_score(y[te], probs[te]))

    emb_df = pd.DataFrame(emb, columns=[f"hg_{i}" for i in range(emb.shape[1])])
    frame = pd.concat([merged[feat_cols].reset_index(drop=True), emb_df], axis=1)
    a_e, p_e = fit_eval(_rf(), frame, tr, te)
    a_r, p_r = fit_eval(_rf(), merged[feat_cols], tr, te)
    a_h, p_h = fit_eval(_hgb(), merged[feat_cols], tr, te)
    res["hgnn_enc"]["auroc"].append(a_e); res["hgnn_enc"]["auprc"].append(p_e)
    res["rf_tab"]["auroc"].append(a_r);   res["rf_tab"]["auprc"].append(p_r)
    res["hgb_tab"]["auroc"].append(a_h);  res["hgb_tab"]["auprc"].append(p_h)
    bt = max(a_r, a_h)
    print(f"[hg] fold {fold+1}/{K} (stop@{ep})  hgnn_raw={res['hgnn_raw']['auroc'][-1]:.3f} "
          f"hgnn_enc={a_e:.3f}  rf={a_r:.3f} hgb={a_h:.3f}  (hgnn_enc-best_tab {a_e-bt:+.3f})", flush=True)

print(f"\n=== {K}-fold CV (mean +/- std) | base {y.mean()*100:.2f}% ===")
print(f"  {'model':10s} {'AUROC':>16s} {'AUPRC':>16s}")
for m in MODELS:
    au, ap = np.array(res[m]["auroc"]), np.array(res[m]["auprc"])
    print(f"  {m:10s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")
bt_au = np.maximum(res["rf_tab"]["auroc"], res["hgb_tab"]["auroc"])
for m in ["hgnn_raw", "hgnn_enc"]:
    d = np.array(res[m]["auroc"]) - bt_au
    print(f"\n  {m} - best_tab:  dAUROC {d.mean():+.4f} +/- {d.std():.4f} "
          f"(folds>0 {int((d>0).sum())}/{K})  per-fold {np.round(d,3).tolist()}")
