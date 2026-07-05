"""Temporal edge features: does time-stamping the encounter-unit edges help?

The GRU result (cv_seq_gru.py) showed that LEARNING the unit trajectory beats
hand-crafted features. This tests the graph-native form of the same idea: keep
the relational structure but attach temporal features to each encounter-unit
edge -- hours in the unit, time-of-day of arrival, and position in the
trajectory -- and let an edge-conditioned message-passing network read them.

An edge-conditioned MPNN (message = MLP[h_src ; edge_feat]) runs on the
encounter-provider-unit graph; each encounter-unit edge carries its visit's
temporal vector, provider/self edges carry zeros. Fold-honest 5-fold, with and
without the temporal edge features (the latter zeroes them), against the
deployable tree:

  notemp     : edge features zeroed (structure only)
  temp       : edge features = [log hours, sin/cos arrival hour, seq position]

Reports each GNN's head and its embedding fed to a tree. temp vs notemp
isolates the temporal contribution; both vs tree shows the ceiling.

Prints aggregate numbers only.

    PYTHONPATH=. python analysis/cv_temporal_edges.py
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
import dgl.function as dglfn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (apply_preprocess, fit_preprocess, load_raw,
                           build_preop_trajectory_features, add_calendar_features)
from medhg_ps.deploy import MIN_SUP, UNIT_BUCKET, seq_feature_dict, _load_cpt_map
from medhg_ps.train import set_seed, _resolve_device

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
MAX_EPOCHS, PATIENCE = (25, 6) if SMOKE else (150, 15)
TOPK, EF = 50, 4               # node feats; edge-feature dim
HID, NL, DP, LR = 64, 2, 0.3, 5e-3
HGB = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=42)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


print(f"[temp] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
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

# static care-path features for the tree reference
u = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
u["cid"] = u["Clarity_ID"].astype("Int64").astype(str)
u["g"] = u["UnitType"].map(UNIT_BUCKET).fillna("Other")
ud = u.dropna(subset=["Clarity_ID"]).drop_duplicates("cid", keep="first").set_index("cid")
a3 = raw.enc_unit_edges.copy()
a3["UnitType"] = _norm(a3["DepartmentID"]).map(ud["g"]).fillna(a3["UnitType"])
a3["InTime"] = pd.to_datetime(a3["InTime"], errors="coerce")
a3["Hours"] = pd.to_numeric(a3.get("Hours"), errors="coerce").fillna(0.0)
a3["SeqInEncounter"] = pd.to_numeric(a3.get("SeqInEncounter"), errors="coerce")
a3["LogID"] = a3["LogID"].astype(str)
a3 = a3.sort_values(["LogID", "InTime"])
seqs = a3.groupby("LogID", sort=False)["UnitType"].apply(list)
stat = [dict(seq_feature_dict(s), LogID=l) for l, s in seqs.items()]
Fstat = merged[["LogID"]].merge(pd.DataFrame(stat).astype({"LogID": str}), on="LogID", how="left").fillna(0)
stat_cols = [c for c in Fstat.columns if c != "LogID"]

# node id spaces
P_ids = sorted({p for p in _norm(raw.enc_prov_edges["ProvID"]) if p and p != "nan"})
U_ids = sorted({d for d in _norm(raw.enc_unit_edges["DepartmentID"]) if d and d != "nan"})
pidx = {p: N + i for i, p in enumerate(P_ids)}
uidx = {d: N + len(P_ids) + i for i, d in enumerate(U_ids)}
TOT = N + len(P_ids) + len(U_ids)

# ---- build the graph ONCE with edge features on encounter-unit edges -------
src, dst, ef = [], [], []
# enc<->unit edges carry the visit's temporal vector
amax = a3.groupby("LogID")["SeqInEncounter"].transform("max")
a3["_posfrac"] = (a3["SeqInEncounter"] / amax.replace(0, np.nan)).fillna(0.0)
for lid, dep, hrs, t, pf in zip(a3["LogID"], _norm(a3["DepartmentID"]), a3["Hours"],
                                a3["InTime"], a3["_posfrac"]):
    r, uu = row_of.get(lid), uidx.get(dep)
    if r is None or uu is None:
        continue
    hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
    e = [np.log1p(max(float(hrs), 0.0)), np.sin(2 * np.pi * hour / 24.0),
         np.cos(2 * np.pi * hour / 24.0), float(pf)]
    src += [r, uu]; dst += [uu, r]; ef += [e, e]
# enc<->prov edges: zero temporal features
ep = raw.enc_prov_edges[["LogID", "ProvID"]].copy()
ep["LogID"] = _norm(ep["LogID"]); ep["ProvID"] = _norm(ep["ProvID"])
for lid, pv in zip(ep["LogID"], ep["ProvID"]):
    r, pp = row_of.get(lid), pidx.get(pv)
    if r is not None and pp is not None:
        src += [r, pp]; dst += [pp, r]; ef += [[0.0] * EF, [0.0] * EF]
# self loops: zero
src += list(range(TOT)); dst += list(range(TOT)); ef += [[0.0] * EF] * TOT
g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=TOT).to(dev)
EF_ALL = torch.tensor(np.asarray(ef, dtype=np.float32), device=dev)
print(f"[temp] nodes={TOT:,}  edges={g.num_edges():,}  (enc-unit carry temporal ef)", flush=True)


class EdgeMPNN(nn.Module):
    """Message = ReLU(W[h_src ; edge_feat]); mean-aggregate; GRU-style update."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.msg = nn.Linear(in_dim + EF, out_dim)
        self.upd = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, g, h, efeat):
        with g.local_scope():
            g.ndata["h"] = h
            g.edata["ef"] = efeat
            g.apply_edges(lambda e: {"m": Fn.relu(self.msg(
                torch.cat([e.src["h"], e.data["ef"]], dim=-1)))})
            g.update_all(dglfn.copy_e("m", "m"), dglfn.mean("m", "agg"))
            return Fn.relu(self.upd(torch.cat([h, g.ndata["agg"]], dim=-1)))


class TempNet(nn.Module):
    def __init__(self, ind):
        super().__init__()
        self.drop = nn.Dropout(DP)
        self.layers = nn.ModuleList()
        dims = [ind] + [HID] * NL
        for i in range(NL):
            self.layers.append(EdgeMPNN(dims[i], dims[i + 1]))
        self.cls = nn.Linear(HID, 2)

    def embed(self, g, x, efeat):
        h = x
        for conv in self.layers:
            h = self.drop(conv(g, h, efeat))
        return h

    def forward(self, g, x, efeat):
        return self.cls(self.embed(g, x, efeat))


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
    return torch.tensor(Xn, device=dev)


def train_extract(Xn, yt, trm, vam, efeat):
    set_seed(SEED)
    net = TempNet(Xn.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    npos, nneg = float((yt[trm] == 1).sum()), float((yt[trm] == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)], device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    yv, vm = yt[vam].cpu().numpy(), vam.cpu().numpy()
    best, state, pat = -1.0, None, PATIENCE
    for ep in range(MAX_EPOCHS):
        net.train(); logit = net(g, Xn, efeat)[:N]; loss = lf(logit[trm], yt[trm])
        opt.zero_grad(); loss.backward(); opt.step(); net.eval()
        with torch.no_grad():
            p = torch.softmax(net(g, Xn, efeat)[:N], -1)[:, 1].cpu().numpy()
        try: va = roc_auc_score(yv, p[vm])
        except ValueError: va = 0.5
        if va > best:
            best, state, pat = va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        emb = net.embed(g, Xn, efeat)[:N].cpu().numpy()
        praw = torch.softmax(net(g, Xn, efeat)[:N], -1)[:, 1].cpu().numpy()
    return emb, praw, best


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
    E = StandardScaler().fit(emb[tr]).transform(emb)
    m = HGB().fit(E[tr], y[tr]); p = m.predict_proba(E[te])[:, 1]
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


def main():
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    zero_ef = torch.zeros_like(EF_ALL)
    tags = ["tree", "notemp_raw", "notemp_emb", "temp_raw", "temp_emb"]
    R = {t: {"au": [], "ap": []} for t in tags}
    for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
        rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
        nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]
        trm = torch.zeros(N, dtype=torch.bool, device=dev); trm[tr] = True
        vam = torch.zeros(N, dtype=torch.bool, device=dev); vam[va] = True

        au, ap = tree_ref(tr, te); R["tree"]["au"].append(au); R["tree"]["ap"].append(ap)
        Xn = node_features(tr)
        for tag, efeat in [("notemp", zero_ef), ("temp", EF_ALL)]:
            emb, praw, vb = train_extract(Xn, yt, trm, vam, efeat)
            R[f"{tag}_raw"]["au"].append(roc_auc_score(y[te], praw[te]))
            R[f"{tag}_raw"]["ap"].append(average_precision_score(y[te], praw[te]))
            a, p = emb_tree(emb, tr, te)
            R[f"{tag}_emb"]["au"].append(a); R[f"{tag}_emb"]["ap"].append(p)
            print(f"[temp] fold {fi+1}/{K} {tag} (val {vb:.3f}) raw {R[f'{tag}_raw']['au'][-1]:.3f} "
                  f"emb {a:.3f}", flush=True)

    print(f"\n=== {K}-fold CV | base {y.mean()*100:.2f}% ===")
    print(f"  {'model':12s} {'AUROC':>16s} {'AUPRC':>16s}")
    for t in tags:
        au, ap = np.array(R[t]["au"]), np.array(R[t]["ap"])
        print(f"  {t:12s} {au.mean():.3f} +/- {au.std():.3f}    {ap.mean():.3f} +/- {ap.std():.3f}")
    for a, b in [("temp_raw", "notemp_raw"), ("temp_emb", "notemp_emb")]:
        d = np.array(R[a]["au"]) - np.array(R[b]["au"])
        print(f"  {a} - {b}: dAUROC {d.mean():+.4f} (folds>0 {int((d>0).sum())}/{K})")


if __name__ == "__main__":
    main()
