"""'Throw everything at it': relational models on ALL available features.

Encounter node features = tabular (38) + CPT (one-hot, 46) + post-op care-unit
sequence features. Run a hypergraph network (provider+unit hyperedges) on these
enriched nodes, vs tree models (RF / HGB) on the SAME enriched features -- so the
only difference is relational message passing. If the relational model doesn't
beat the tree on identical features, the connections add nothing even at best.

Fold-honest 5-fold CV. Reports relational raw / +RF vs the best enriched tree.
   PYTHONPATH=. python analysis/cv_allfeat_relational.py
"""
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (load_raw, fit_preprocess, apply_preprocess,
                           build_preop_trajectory_features, add_calendar_features)
from medhg_ps.train import set_seed, _resolve_device

K, SEED, VAL_FRAC = 5, 42, 0.10
MAX_EPOCHS, PATIENCE = 300, 25
CONFIGS = [dict(hid=64, dropout=0.3, lr=0.01), dict(hid=128, dropout=0.5, lr=0.02)]
UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]
MIN_SUP = 50
COHORT_COLS = ["LogID", "EncounterCSN", "PAT_ID", "SurgeryDate", "SurgeryYear", "PrimaryCPT", "AgeYears"]


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def load_cpt():
    for d in [str(C.DATA_DIR), "/Users/yiyezhang/Dropbox/Surgery"]:
        for p in glob.glob(os.path.join(d, "*.csv")):
            try:
                h = pd.read_csv(p, nrows=1, header=None, dtype=str)
            except Exception:
                continue
            if str(h.iloc[0, 0]).strip() == "LogID":
                df = pd.read_csv(p, low_memory=False)
                if {"LogID", "PrimaryCPT"} <= set(df.columns):
                    return dict(zip(_norm(df["LogID"]), _norm(df["PrimaryCPT"])))
            elif h.shape[1] == len(COHORT_COLS):
                df = pd.read_csv(p, header=None, names=COHORT_COLS, low_memory=False)
                return dict(zip(_norm(df["LogID"]), _norm(df["PrimaryCPT"])))
    raise FileNotFoundError("cohort CSV with PrimaryCPT not found")


print("[all] loading + assembling...", flush=True)
cpt_map = load_cpt()
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
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)

# --- post-op care-unit SEQUENCE features (full trajectory) -----------
gnnb = {"ICU": "Intensive", "PICU": "Intensive", "NICU": "Intensive", "Med/Surg": "Acute",
        "Procedural Area": "OR", "ED": "ED", "Recovery Area": "Intermediate"}
u = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
u["cid"] = u["Clarity_ID"].astype("Int64").astype(str)
u["g"] = u["UnitType"].map(gnnb).fillna("Other")
ud = u.dropna(subset=["Clarity_ID"]).drop_duplicates("cid", keep="first").set_index("cid")
a3 = raw.enc_unit_edges.copy()
a3["UnitType"] = (a3["DepartmentID"].astype(str).str.replace(r"\.0+$", "", regex=True)
                  .map(ud["g"]).fillna(a3["UnitType"]))
a3["InTime"] = pd.to_datetime(a3["InTime"]); a3["LogID"] = a3["LogID"].astype(str)
seqs = a3.sort_values(["LogID", "InTime"]).groupby("LogID", sort=False)["UnitType"].apply(list)


def collapse(s):
    o = []
    for x in s:
        if not o or o[-1] != x:
            o.append(x)
    return o


rows = []
for lid, rawseq in seqs.items():
    c = collapse(rawseq); f = {"LogID": lid}
    for un in UNITS:
        f[f"cnt_{un}"] = rawseq.count(un)
    f["n_stops"] = len(c); f["n_or"] = c.count("OR"); f["has_ICU"] = int("Intensive" in c)
    f["ICU_after_OR"] = int("Intensive" in c and "OR" in c and c.index("Intensive") > c.index("OR"))
    f[f"end_{c[-1]}"] = 1
    for a, b in zip(c, c[1:]):
        f[f"bg_{a}>{b}"] = f.get(f"bg_{a}>{b}", 0) + 1
    rows.append(f)
Fseq = pd.DataFrame(rows).fillna(0)
Fseq["LogID"] = Fseq["LogID"].astype(str)
Fseq = merged[["LogID"]].merge(Fseq, on="LogID", how="left").fillna(0)
seq_all = [c for c in Fseq.columns if c != "LogID"]
print(f"[all] cohort={N:,} base={y.mean()*100:.2f}% | tabular={len(feat_cols)} "
      f"CPT=1 seq_candidates={len(seq_all)}", flush=True)

# --- provider+unit hypergraph (for the relational model) -------------
ni, ei, eid = [], [], 0
for df, key in [(raw.enc_prov_edges, "ProvID"), (raw.enc_unit_edges, "DepartmentID")]:
    g = df[["LogID", key]].copy()
    g["LogID"] = _norm(g["LogID"]); g[key] = _norm(g[key]); g["r"] = g["LogID"].map(row_of)
    for _, sub in g.dropna(subset=["r", key]).groupby(key):
        r = np.unique(sub["r"].astype(int).values)
        if len(r) > 1:
            ni.extend(r.tolist()); ei.extend([eid] * len(r)); eid += 1
ni.extend(range(N)); ei.extend(range(eid, eid + N)); eid += N
H = torch.sparse_coo_tensor(torch.tensor([ni, ei]), torch.ones(len(ni)), size=(N, eid)).coalesce()
Ht = H.transpose(0, 1).coalesce().to(dev); H = H.to(dev)
Dvi = torch.sparse.sum(H, 1).to_dense().clamp(min=1).pow(-0.5).to(dev)
Dei = torch.sparse.sum(H, 0).to_dense().clamp(min=1).pow(-1).to(dev)


def prop(X):
    x = X * Dvi[:, None]; x = torch.sparse.mm(Ht, x); x = x * Dei[:, None]
    x = torch.sparse.mm(H, x); return x * Dvi[:, None]


class HGNN(nn.Module):
    def __init__(self, d, hid, dp):
        super().__init__()
        self.t1 = nn.Linear(d, hid); self.t2 = nn.Linear(hid, hid)
        self.bn = nn.BatchNorm1d(hid); self.cls = nn.Linear(hid, 2); self.drop = nn.Dropout(dp)

    def forward(self, X):
        h = self.drop(Fn.relu(self.bn(prop(self.t1(X))))); emb = prop(self.t2(h))
        return self.cls(emb), emb


def train_one(X, yt, trm, vam, cfg):
    set_seed(SEED); m = HGNN(X.shape[1], cfg["hid"], cfg["dropout"]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    npos = float((yt[trm] == 1).sum()); nneg = float((yt[trm] == 0).sum())
    w = torch.tensor([(npos+nneg)/(2*max(nneg, 1)), (npos+nneg)/(2*max(npos, 1))], device=dev)
    lf = nn.CrossEntropyLoss(weight=w); best, st, pat = -1, None, PATIENCE
    yv = yt[vam].cpu().numpy(); vm = vam.cpu().numpy()
    for ep in range(MAX_EPOCHS):
        m.train(); lo, _ = m(X); loss = lf(lo[trm], yt[trm]); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(X)[0], -1)[:, 1].cpu().numpy()
        try: va = roc_auc_score(yv, p[vm])
        except ValueError: va = 0.5
        if va > best: best, st, pat = va, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0: break
    m.load_state_dict(st); return m, best


skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
yt = torch.tensor(y, dtype=torch.long, device=dev)
res = {m: {"au": [], "ap": []} for m in ["tree_rf", "tree_hgb", "rel_raw", "rel_enc"]}

for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
    rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]
    trm = torch.zeros(N, dtype=torch.bool, device=dev); trm[tr] = True
    vam = torch.zeros(N, dtype=torch.bool, device=dev); vam[va] = True
    # enriched features, all fit on train only
    Xtab_tr, st = fit_preprocess(merged[feat_cols].loc[np.isin(np.arange(N), tr)].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    Xcpt = ohe.transform(cpt_arr)
    keep = [c for c in seq_all if (Fseq.iloc[tr][c] > 0).sum() >= MIN_SUP]
    Xseq = Fseq[keep].values.astype(float)
    Xall = np.hstack([Xtab, Xcpt, Xseq])
    sc = StandardScaler().fit(Xall[tr]); Xall = sc.transform(Xall)
    Xt = torch.tensor(Xall, dtype=torch.float32, device=dev)

    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                                class_weight="balanced", random_state=42, n_jobs=-1).fit(Xall[tr], y[tr])
    pr = rf.predict_proba(Xall[te])[:, 1]
    res["tree_rf"]["au"].append(roc_auc_score(y[te], pr)); res["tree_rf"]["ap"].append(average_precision_score(y[te], pr))
    hg = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                                        random_state=42).fit(Xall[tr], y[tr])
    ph = hg.predict_proba(Xall[te])[:, 1]
    res["tree_hgb"]["au"].append(roc_auc_score(y[te], ph)); res["tree_hgb"]["ap"].append(average_precision_score(y[te], ph))

    bva, bm = -1, None
    for cfg in CONFIGS:
        m, v = train_one(Xt, yt, trm, vam, cfg)
        if v > bva: bva, bm = v, m
    bm.eval()
    with torch.no_grad():
        lo, emb = bm(Xt); pp = torch.softmax(lo, -1)[:, 1].cpu().numpy(); emb = emb.cpu().numpy()
    res["rel_raw"]["au"].append(roc_auc_score(y[te], pp[te])); res["rel_raw"]["ap"].append(average_precision_score(y[te], pp[te]))
    Xe = np.hstack([Xall, emb])
    rfe = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                                 class_weight="balanced", random_state=42, n_jobs=-1).fit(Xe[tr], y[tr])
    pe = rfe.predict_proba(Xe[te])[:, 1]
    res["rel_enc"]["au"].append(roc_auc_score(y[te], pe)); res["rel_enc"]["ap"].append(average_precision_score(y[te], pe))
    bt = max(res["tree_rf"]["au"][-1], res["tree_hgb"]["au"][-1])
    print(f"[all] fold {fi+1}/{K} | tree_rf {res['tree_rf']['au'][-1]:.3f} tree_hgb {res['tree_hgb']['au'][-1]:.3f} "
          f"rel_raw {res['rel_raw']['au'][-1]:.3f} rel_enc {res['rel_enc']['au'][-1]:.3f} "
          f"(rel_enc-tree {res['rel_enc']['au'][-1]-bt:+.3f})", flush=True)

print(f"\n=== {K}-fold CV (mean +/- std) | base {y.mean()*100:.2f}% | ALL features ===")
for m in ["tree_rf", "tree_hgb", "rel_raw", "rel_enc"]:
    au, ap = np.array(res[m]["au"]), np.array(res[m]["ap"])
    print(f"  {m:9s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")
bt = np.maximum(res["tree_rf"]["au"], res["tree_hgb"]["au"])
for m in ["rel_raw", "rel_enc"]:
    d = np.array(res[m]["au"]) - bt
    print(f"  {m} - best enriched tree: dAUROC {d.mean():+.4f} (folds>0 {int((d>0).sum())}/{K})")
