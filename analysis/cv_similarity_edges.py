"""Patient-similarity edges: do they relieve hub-driven over-smoothing?

The manuscript's mechanism for the GNN's failure is that care-unit (and
procedure) nodes are huge hubs -- a single unit links thousands of unrelated
encounters -- so message passing smooths every encounter toward hub averages.
One proposed remedy is to give encounters a NON-hub path: direct
encounter<->encounter edges between clinically similar patients (kNN on the
tabular features). Aggregating over a handful of similar peers, rather than
only through shared hubs, should counter the dilution.

This runs a relation-typed GNN (R-GCN) on the base encounter-provider-unit
graph, with and without the added similarity relation, fold-honest 5-fold:

  base       : enc<->prov, enc<->unit, self-loop
  base+sim   : + enc<->enc kNN similarity edges (k nearest on tabular features)

For each we report the GNN's own head and its embedding fed to a tree, against
the deployable tree (tabular + CPT + hand-crafted care path). The test is
whether similarity edges let the GNN (a) beat its hub-only self and (b) reach
the tabular ceiling. kNN is built on train-fit standardized features per fold.

Prints aggregate numbers only.

    PYTHONPATH=. python analysis/cv_similarity_edges.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import dgl
from dgl.nn import RelGraphConv
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (apply_preprocess, fit_preprocess, load_raw,
                           build_preop_trajectory_features, add_calendar_features)
from medhg_ps.deploy import MIN_SUP, UNIT_BUCKET, seq_feature_dict, _load_cpt_map
from medhg_ps.train import set_seed, _resolve_device

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
MAX_EPOCHS, PATIENCE = (25, 6) if SMOKE else (150, 15)
TOPK = 50                      # encounter node features (RF-selected)
KNN = 10                       # similarity edges per encounter
HID, NL, DP, LR = 64, 2, 0.3, 5e-3
NREL, NBASES = 6, 4            # 0 e->p 1 p->e 2 e->u 3 u->e 4 sim 5 self
HGB = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=42)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


# ---- assemble frame + static care-path features (tree reference) ----------
print(f"[sim] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
cpt_map = _load_cpt_map()
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
row_of = {l: i for i, l in enumerate(merged["LogID"])}
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)

# hand-crafted static care-path features for the tree reference
u = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
u["cid"] = u["Clarity_ID"].astype("Int64").astype(str)
u["g"] = u["UnitType"].map(UNIT_BUCKET).fillna("Other")
ud = u.dropna(subset=["Clarity_ID"]).drop_duplicates("cid", keep="first").set_index("cid")
a3 = raw.enc_unit_edges.copy()
a3["UnitType"] = _norm(a3["DepartmentID"]).map(ud["g"]).fillna(a3["UnitType"])
a3["InTime"] = pd.to_datetime(a3["InTime"], errors="coerce"); a3["LogID"] = a3["LogID"].astype(str)
seqs = a3.sort_values(["LogID", "InTime"]).groupby("LogID", sort=False)["UnitType"].apply(list)
stat = [dict(seq_feature_dict(s), LogID=l) for l, s in seqs.items()]
Fstat = merged[["LogID"]].merge(pd.DataFrame(stat).astype({"LogID": str}), on="LogID", how="left").fillna(0)
stat_cols = [c for c in Fstat.columns if c != "LogID"]

# provider / unit node id spaces
P_ids = sorted({p for p in _norm(raw.enc_prov_edges["ProvID"]) if p and p != "nan"})
U_ids = sorted({d for d in _norm(raw.enc_unit_edges["DepartmentID"]) if d and d != "nan"})
pidx = {p: N + i for i, p in enumerate(P_ids)}
uidx = {d: N + len(P_ids) + i for i, d in enumerate(U_ids)}
TOT = N + len(P_ids) + len(U_ids)

# static heterograph edges (enc<->prov, enc<->unit); similarity edges added per fold
ep = raw.enc_prov_edges[["LogID", "ProvID"]].copy()
ep["LogID"] = _norm(ep["LogID"]); ep["ProvID"] = _norm(ep["ProvID"])
eu = raw.enc_unit_edges[["LogID", "DepartmentID"]].copy()
eu["LogID"] = _norm(eu["LogID"]); eu["DepartmentID"] = _norm(eu["DepartmentID"])
base_src, base_dst, base_et = [], [], []
for lid, pv in zip(ep["LogID"], ep["ProvID"]):
    r, pp = row_of.get(lid), pidx.get(pv)
    if r is not None and pp is not None:
        base_src += [r, pp]; base_dst += [pp, r]; base_et += [0, 1]
for lid, dp in zip(eu["LogID"], eu["DepartmentID"]):
    r, uu = row_of.get(lid), uidx.get(dp)
    if r is not None and uu is not None:
        base_src += [r, uu]; base_dst += [uu, r]; base_et += [2, 3]
print(f"[sim] nodes={TOT:,} (enc {N:,}, prov {len(P_ids):,}, unit {len(U_ids):,})  "
      f"base edges={len(base_src):,}", flush=True)


class RGCN(nn.Module):
    def __init__(self, ind):
        super().__init__()
        self.drop = nn.Dropout(DP); self.layers = nn.ModuleList()
        dims = [ind] + [HID] * NL
        for i in range(NL):
            self.layers.append(RelGraphConv(dims[i], dims[i + 1], NREL,
                                            regularizer="basis", num_bases=NBASES))
        self.cls = nn.Linear(HID, 2)

    def embed(self, g, x, et):
        h = x
        for conv in self.layers:
            h = self.drop(Fn.relu(conv(g, h, et)))
        return h

    def forward(self, g, x, et):
        return self.cls(self.embed(g, x, et))


def build_graph(Xsel, with_sim):
    src, dst, et = list(base_src), list(base_dst), list(base_et)
    if with_sim:
        nn_ = NearestNeighbors(n_neighbors=KNN + 1).fit(Xsel)
        nbr = nn_.kneighbors(Xsel, return_distance=False)[:, 1:]    # drop self
        for i in range(N):
            for j in nbr[i]:
                src += [i, int(j)]; dst += [int(j), i]; et += [4, 4]
    # self-loops as relation 5
    src += list(range(TOT)); dst += list(range(TOT)); et += [5] * TOT
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=TOT).to(dev)
    et = torch.tensor(et, dtype=torch.long, device=dev)
    return g, et


def train_extract(g, et, Xn, yt, trm, vam):
    set_seed(SEED)
    net = RGCN(Xn.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    npos, nneg = float((yt[trm] == 1).sum()), float((yt[trm] == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)], device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    yv, vm = yt[vam].cpu().numpy(), vam.cpu().numpy()
    best, state, pat = -1.0, None, PATIENCE
    for ep in range(MAX_EPOCHS):
        net.train(); logit = net(g, Xn, et)[:N]; loss = lf(logit[trm], yt[trm])
        opt.zero_grad(); loss.backward(); opt.step(); net.eval()
        with torch.no_grad():
            p = torch.softmax(net(g, Xn, et)[:N], -1)[:, 1].cpu().numpy()
        try: va = roc_auc_score(yv, p[vm])
        except ValueError: va = 0.5
        if va > best:
            best, state, pat = va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        emb = net.embed(g, Xn, et)[:N].cpu().numpy()
        praw = torch.softmax(net(g, Xn, et)[:N], -1)[:, 1].cpu().numpy()
    return emb, praw, best


def node_features(tr):
    Xtab, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    Xall = np.hstack([Xtab, ohe.transform(cpt_arr)])
    sel = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                                 class_weight="balanced", random_state=42, n_jobs=-1).fit(Xall[tr], y[tr])
    top = np.argsort(sel.feature_importances_)[::-1][:TOPK]
    Xsel = StandardScaler().fit(Xall[tr][:, top]).transform(Xall[:, top])
    Xn = np.zeros((TOT, TOPK), dtype=np.float32); Xn[:N] = Xsel
    return torch.tensor(Xn, device=dev), Xsel


def tree_ref(tr, te):
    Xtab, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    keep = [c for c in stat_cols if (Fstat.iloc[tr][c] > 0).sum() >= MIN_SUP]
    X = np.hstack([Xtab, ohe.transform(cpt_arr), Fstat[keep].values.astype(float)])
    X = StandardScaler().fit(X[tr]).transform(X)
    m = HGB().fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


def emb_tree(emb, tr, te):
    sc = StandardScaler().fit(emb[tr]); E = sc.transform(emb)
    m = HGB().fit(E[tr], y[tr]); p = m.predict_proba(E[te])[:, 1]
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


def main():
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    tags = ["tree", "base_raw", "base_emb", "sim_raw", "sim_emb"]
    R = {t: {"au": [], "ap": []} for t in tags}
    for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
        rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
        nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]
        trm = torch.zeros(N, dtype=torch.bool, device=dev); trm[tr] = True
        vam = torch.zeros(N, dtype=torch.bool, device=dev); vam[va] = True

        au, ap = tree_ref(tr, te); R["tree"]["au"].append(au); R["tree"]["ap"].append(ap)
        Xn, Xsel = node_features(tr)
        for tag, with_sim in [("base", False), ("sim", True)]:
            g, et = build_graph(Xsel, with_sim)
            emb, praw, vb = train_extract(g, et, Xn, yt, trm, vam)
            R[f"{tag}_raw"]["au"].append(roc_auc_score(y[te], praw[te]))
            R[f"{tag}_raw"]["ap"].append(average_precision_score(y[te], praw[te]))
            a, p = emb_tree(emb, tr, te)
            R[f"{tag}_emb"]["au"].append(a); R[f"{tag}_emb"]["ap"].append(p)
            print(f"[sim] fold {fi+1}/{K} {tag} (val {vb:.3f}) raw {R[f'{tag}_raw']['au'][-1]:.3f} "
                  f"emb {a:.3f}", flush=True)

    print(f"\n=== {K}-fold CV | base {y.mean()*100:.2f}% ===")
    print(f"  {'model':10s} {'AUROC':>16s} {'AUPRC':>16s}")
    for t in tags:
        au, ap = np.array(R[t]["au"]), np.array(R[t]["ap"])
        print(f"  {t:10s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")
    for a, b in [("sim_raw", "base_raw"), ("sim_emb", "base_emb")]:
        d = np.array(R[a]["au"]) - np.array(R[b]["au"])
        print(f"  {a} - {b}: dAUROC {d.mean():+.4f} (folds>0 {int((d>0).sum())}/{K})")


if __name__ == "__main__":
    main()
