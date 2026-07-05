"""Broad conjunctive-hyperedge sweep (5-fold CV).

Exhausts the conjunctive constructions buildable from available attributes:
surgeon (S), unit (U), ASA (A), anesthesia (Anes), age-decile (Age),
patient-type (PT), diabetes (DM), comorbidity-signature (DxSig),
procedure-count (Proc). Each construction is a conjunction key -> hyperedge of
encounters sharing that key (+ self-loops). Tuned HGNN per fold, vs tabular.

(CPT is NOT in the extracts -- surgeon x CPT x dx x unit needs a separate pull;
see sql/Extract_Primary_CPT.sql.)

Prints group-size stats and raw/enc vs tabular for every construction.
Fold-honest; writes nothing.  PYTHONPATH=. python analysis/cv_hypergraph_conj2.py
"""
from collections import defaultdict

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
CONFIGS = [dict(hid=32, dropout=0.3, lr=0.01), dict(hid=64, dropout=0.5, lr=0.02)]
COMORB = ["Diabetes Mellitus", "Hypertension requiring medication", "Heart Failure",
          "History of Severe COPD", "Ascites", "Disseminated Cancer", "Bleeding Disorder",
          "Preop Acute Kidney Injury", "Preop Dialysis", "Ventilator Dependent",
          "Immunosuppressive Therapy", "Current Smoker within 1 year",
          "Preop RBC Transfusions (72h)"]
SPECS = [
    (["surg", "unit"], "S.U"),
    (["surg", "unit", "asa"], "S.U.ASA"),
    (["surg", "unit", "anes"], "S.U.Anes"),
    (["surg", "unit", "dm"], "S.U.DM"),
    (["surg", "unit", "dxsig"], "S.U.DxSig"),
    (["surg", "unit", "age"], "S.U.Age"),
    (["surg", "unit", "proc"], "S.U.Proc"),
    (["surg", "asa", "dxsig"], "S.ASA.DxSig"),
    (["unit", "dxsig"], "U.DxSig"),
    (["asa", "dxsig", "ptype"], "ASA.DxSig.PT"),
    (["surg", "unit", "asa", "dm"], "S.U.ASA.DM"),
]

print("[c2] loading + assembling...", flush=True)
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
print(f"[c2] cohort={N:,}  base={y.mean()*100:.2f}%", flush=True)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def _col(name, default=""):
    return (merged[name].astype(str).tolist() if name in merged.columns else [default] * N)


age_b = pd.qcut(pd.to_numeric(merged.get("AgeYears"), errors="coerce"),
                10, labels=False, duplicates="drop")
present = [c for c in COMORB if c in merged.columns]
attr = {
    "surg": _norm(merged.get("PrimarySurgID", pd.Series([""] * N))).tolist(),
    "asa": _col("ASAClass"), "anes": _col("AnesType"), "ptype": _col("PatientType"),
    "dm": _col("Diabetes Mellitus"), "proc": _col("# of Other Procedures"),
    "age": pd.Series(age_b).astype("Int64").astype(str).tolist(),
    "dxsig": (merged[present].astype(str).agg("|".join, axis=1).tolist() if present else [""] * N),
}
units_of = defaultdict(list)
a3 = raw.enc_unit_edges[["LogID", "DepartmentID"]].copy()
a3["LogID"] = _norm(a3["LogID"]); a3["DepartmentID"] = _norm(a3["DepartmentID"])
for lid, dep in zip(a3["LogID"], a3["DepartmentID"]):
    r = row_of.get(lid)
    if r is not None and dep and dep != "nan":
        units_of[r].append(dep)
units_of = {r: sorted(set(v)) for r, v in units_of.items()}


def build_groups(spec):
    d = defaultdict(list)
    need_surg = "surg" in spec
    for r in range(N):
        if need_surg:
            s = attr["surg"][r]
            if not s or s in ("nan", "None", ""):
                continue
        base = [attr[a][r] for a in spec if a != "unit"]
        if "unit" in spec:
            for u in units_of.get(r, []):
                d[tuple(base + [u])].append(r)
        else:
            d[tuple(base)].append(r)
    return [np.array(sorted(set(v))) for v in d.values() if len(set(v)) > 1]


class HyperOp:
    def __init__(self, groups):
        ni, ei, eid = [], [], 0
        for rows in groups:
            ni.extend(rows.tolist()); ei.extend([eid] * len(rows)); eid += 1
        kept = eid
        ni.extend(range(N)); ei.extend(range(eid, eid + N)); eid += N
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
        self.op = op; self.t1 = nn.Linear(d_in, hid); self.t2 = nn.Linear(hid, hid)
        self.bn = nn.BatchNorm1d(hid); self.cls = nn.Linear(hid, 2); self.drop = nn.Dropout(dropout)

    def forward(self, X):
        h = self.drop(Fn.relu(self.bn(self.op.prop(self.t1(X)))))
        return self.cls(self.op.prop(self.t2(h))), self.op.prop(self.t2(h))


def train_one(op, X, yt, tr_m, va_m, cfg):
    set_seed(SEED)
    m = HGNN(op, X.shape[1], cfg["hid"], cfg["dropout"]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    npos = float((yt[tr_m] == 1).sum()); nneg = float((yt[tr_m] == 0).sum())
    w = torch.tensor([(npos+nneg)/(2*max(nneg, 1)), (npos+nneg)/(2*max(npos, 1))], device=dev)
    lf = nn.CrossEntropyLoss(weight=w); best, state, pat = -1.0, None, PATIENCE
    yv = yt[va_m].cpu().numpy(); vam = va_m.cpu().numpy()
    for ep in range(MAX_EPOCHS):
        m.train(); logit, _ = m(X); loss = lf(logit[tr_m], yt[tr_m])
        opt.zero_grad(); loss.backward(); opt.step(); m.eval()
        with torch.no_grad():
            p = torch.softmax(m(X)[0], -1)[:, 1].cpu().numpy()
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
print(f"[c2] tabular best AUROC mean {best_tab.mean():.3f}\n", flush=True)

for spec, tag in SPECS:
    groups = build_groups(spec)
    sizes = np.array([len(g) for g in groups]) if groups else np.array([0])
    op = HyperOp(groups)
    rau, eau, rap, eap = [], [], [], []
    for (tr, va, te, tr_m, va_m, X) in folds:
        bva, bm = -1, None
        for cfg in CONFIGS:
            m, v = train_one(op, X, yt, tr_m, va_m, cfg)
            if v > bva: bva, bm = v, m
        bm.eval()
        with torch.no_grad():
            lo, emb = bm(X); probs = torch.softmax(lo, -1)[:, 1].cpu().numpy(); emb = emb.cpu().numpy()
        rau.append(roc_auc_score(y[te], probs[te])); rap.append(average_precision_score(y[te], probs[te]))
        fr = pd.concat([merged[feat_cols].reset_index(drop=True),
                        pd.DataFrame(emb, columns=[f"hg_{i}" for i in range(emb.shape[1])])], axis=1)
        a_e, p_e = fit_eval(_rf(), fr, tr, te); eau.append(a_e); eap.append(p_e)
    rau, eau = np.array(rau), np.array(eau)
    print(f"[{tag:13s}] {len(groups):6d} edges (med={int(np.median(sizes))} max={sizes.max()}) | "
          f"raw {rau.mean():.3f} enc {eau.mean():.3f}±{eau.std():.3f}/{np.mean(eap):.3f}±{np.std(eap):.3f} || "
          f"raw-tab {(rau-best_tab).mean():+.3f}({int(((rau-best_tab)>0).sum())}/{K})  "
          f"enc-tab {(eau-best_tab).mean():+.3f}({int(((eau-best_tab)>0).sum())}/{K})", flush=True)
print(f"\nbest tabular = {best_tab.mean():.3f} AUROC.")
