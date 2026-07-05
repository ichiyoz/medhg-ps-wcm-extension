"""Table 3 (relational/graph suite) on the PLOS outcome (top-25% LOS).

Companion to analysis/plos_table2.py. Same protocol, same leaky-feature drops,
same PLOS derivation from A3 (max OutTime - min InTime). Reuses the tabular
baseline OOF (rf_clin) from artifacts/newdata/plos_table2_oof.npz. Adds
graph/relational rows. NO DICE.

Rows (all under 5-fold StratifiedKFold seed 42, isotonic-calibrated pooled OOF,
RF-canonical downstream where applicable, bootstrap n=2000 CIs):

  rf_clin                    reference tabular baseline (loaded from cache)
  gnn_prov_rf                RGCN over A2 encounter<->provider, RF on [Xtab (+) enc emb]
  gnn_arch_rf                best of SAGE/GAT/RGCN over A2, RF on [Xtab (+) enc emb]
  hypergraph_prov_rf         provider-hyperedges + self-loops, encoder emb + RF
  gru_orders_only            GRU on PRE-SURGERY orders, no tabular
  sim_edges_rf               enc<->enc kNN similarity (train-fit) + GNN emb + RF
  temporal_prov_rf           surgery-date-of-week temporal encoding of prov edges
  allfeat_rel_rf             enc + prov + CPT-code nodes, RGCN encoder + RF

Every row: AUROC/AUPRC/Brier + F1-optimal threshold + F1/precision/recall/
specificity/flag-rate at that threshold (95% CIs on the ranking metrics + F1).

A3 (unit trajectory) is EXCLUDED from every graph substrate. Order-sequence
GRU uses PRE-SURGERY orders only (same OrderTime < SurgeryDate filter as in
plos_table2.py). These constraints prevent LOS leakage.
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
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
import medhg_ps.data as D
from medhg_ps.data import (add_calendar_features, apply_preprocess,
                           build_provider_team_features, collapse_order_runs,
                           fit_preprocess, load_order_sequence, load_raw)
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
# Load raw + derive PLOS + preprocess
# ---------------------------------------------------------------------------
print(f"[t3] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
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
a3["InTime"]  = pd.to_datetime(a3["InTime"], errors="coerce")
a3["OutTime"] = pd.to_datetime(a3["OutTime"], errors="coerce")
los = a3.groupby("LogID").apply(
    lambda g: (g["OutTime"].max() - g["InTime"].min()).total_seconds() / 86400
).rename("los_days").reset_index()
merged = merged.merge(los, on="LogID", how="left")
cutoff = float(merged["los_days"].quantile(0.75))
merged["plos"] = (merged["los_days"] > cutoff).astype("Int64")
merged = merged.loc[merged["los_days"].notna()].reset_index(drop=True)
print(f"[t3] PLOS cutoff = {cutoff:.2f} days; N={len(merged)}", flush=True)

prov_tab = build_provider_team_features(raw.enc_prov_edges, raw.prov_attrs,
                                        raw.encounters)
merged = merged.merge(prov_tab, on="LogID", how="left")
for c in C.PROVIDER_FEATURE_COLUMNS:
    merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

feat_cols_full = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
                  + [c for c in C.CALENDAR_FEATURE_COLUMNS if c in merged.columns])
feat_cols = [c for c in feat_cols_full if c not in LEAKY_FEATS]
prov_cols = [c for c in C.PROVIDER_FEATURE_COLUMNS if c in merged.columns]
print(f"[t3] {len(feat_cols)} clean feats + {len(prov_cols)} prov feats", flush=True)

y_all = merged["plos"].astype(int).values
N = len(merged); base = y_all.mean()
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)
cpt_ids = sorted(set(cpt_arr.ravel().tolist()))
cpt_idx = {c: i for i, c in enumerate(cpt_ids)}
cpt_of = np.array([cpt_idx[c] for c in cpt_arr.ravel()], dtype=np.int64)

# ---------------------------------------------------------------------------
# Pre-surgery orders (for GRU)
# ---------------------------------------------------------------------------
orders_raw = load_order_sequence()
orders_raw["LogID"] = orders_raw["LogID"].astype(str)
orders_raw["OrderTime_dt"] = pd.to_datetime(orders_raw["OrderTime"], errors="coerce")
sd_map = dict(zip(merged["LogID"],
                  pd.to_datetime(merged["SurgeryDate"], errors="coerce")))
orders_raw["SurgeryStart"] = orders_raw["LogID"].map(sd_map)
orders_pre = orders_raw.loc[
    orders_raw["OrderTime_dt"] < orders_raw["SurgeryStart"]].copy()
orders = collapse_order_runs(orders_pre)
orders["LogID"] = orders["LogID"].astype(str)
orders["RunStart"] = pd.to_datetime(orders["RunStart"], errors="coerce")
orders = orders.sort_values(["LogID", "SeqInEncounter"])
vocab = orders["OrderGroup"].value_counts().index.tolist() if len(orders) else []
T2I = {t: i for i, t in enumerate(vocab)}
PAD = max(len(vocab), 1)

row_of = {l: i for i, l in enumerate(merged["LogID"])}
seq_idx = np.full((N, MAXLEN), PAD, dtype=np.int64)
seq_num = np.zeros((N, MAXLEN, NUMF), dtype=np.float32)
lengths = np.ones(N, dtype=np.int64)
for lid, grp in orders.groupby("LogID", sort=False):
    r = row_of.get(lid)
    if r is None: continue
    grp = grp.tail(MAXLEN)
    toks = grp["OrderGroup"].tolist()
    reps = grp["RepeatCount"].tolist()
    gaps = grp["MinutesFromPrev"].tolist()
    times = grp["RunStart"].tolist()
    L = len(toks); lengths[r] = max(L, 1)
    for j in range(L):
        seq_idx[r, j] = T2I.get(toks[j], PAD)
        gap = gaps[j]; gap = 0.0 if pd.isna(gap) else max(float(gap), 0.0)
        t = times[j]; hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        seq_num[r, j] = [np.log1p(max(float(reps[j]), 0.0)),
                         np.log1p(gap),
                         (j + 1) / MAXLEN,
                         np.sin(2 * np.pi * hour / 24.0),
                         np.cos(2 * np.pi * hour / 24.0)]
print(f"[t3] pre-surgery vocab={len(vocab)}  encs with any pre-surgery = "
      f"{int((lengths>1).sum())}/{N}", flush=True)

# ---------------------------------------------------------------------------
# Provider edges (A2)
# ---------------------------------------------------------------------------
enc_id_of = {l: i for i, l in enumerate(merged["LogID"])}
edges_ep = raw.enc_prov_edges.copy()
edges_ep["LogID"] = edges_ep["LogID"].astype(str)
edges_ep["ProvID"] = edges_ep["ProvID"].astype(str)
prov_ids = sorted(edges_ep["ProvID"].dropna().unique().tolist())
prov_id_of = {p: i for i, p in enumerate(prov_ids)}
e_src_ep, e_dst_ep = [], []
for l, p in zip(edges_ep["LogID"], edges_ep["ProvID"]):
    i = enc_id_of.get(l); j = prov_id_of.get(p)
    if i is None or j is None or pd.isna(p): continue
    e_src_ep.append(i); e_dst_ep.append(j)
e_src_ep = np.array(e_src_ep); e_dst_ep = np.array(e_dst_ep)
n_prov = len(prov_ids)

# Provider-provider edges via shared encounters: build once by grouping
# encs->prov membership, then adding an edge between every pair of providers
# that share an encounter. Cap total edges to control memory.
_p_by_enc = {}
for i, j in zip(e_src_ep, e_dst_ep):
    _p_by_enc.setdefault(int(i), []).append(int(j))
pp_src, pp_dst = [], []
_seen_pp = set()
for enc_idx, plist in _p_by_enc.items():
    plist = list(set(plist))
    for a in plist:
        for b in plist:
            if a == b: continue
            key = (a, b)
            if key in _seen_pp: continue
            _seen_pp.add(key)
            pp_src.append(a); pp_dst.append(b)
pp_src = np.array(pp_src); pp_dst = np.array(pp_dst)
print(f"[t3] A2 encounter-prov edges = {len(e_src_ep):,}; "
      f"provider-provider = {len(pp_src):,}; providers = {n_prov}", flush=True)


# ---------------------------------------------------------------------------
# GRU (pre-surgery orders, sequence-only)
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
# Graph encoders — 2-layer bipartite message passing over encounter+provider
# Variants differ in the aggregation choice and (optionally) added edges.
# ---------------------------------------------------------------------------

def _make_dgl_bipartite(add_sim_edges=None, add_temporal=False):
    """Return a DGL heterograph with encounter/provider nodes and A2 edges.
    If add_sim_edges is not None, add enc<->enc similarity edges (indices
    (src, dst)). If add_temporal, add a per-edge feature representing surgery
    day-of-week (0/1 for weekend) on encounter-provider edges."""
    import dgl
    e_src_t = torch.tensor(e_src_ep, dtype=torch.int64)
    e_dst_t = torch.tensor(e_dst_ep, dtype=torch.int64)
    edata_dict = {
        ("encounter", "treated_by", "provider"): (e_src_t, e_dst_t),
        ("provider", "treats", "encounter"):     (e_dst_t, e_src_t),
    }
    if add_sim_edges is not None:
        s_src, s_dst = add_sim_edges
        edata_dict[("encounter", "similar_to", "encounter")] = (
            torch.tensor(s_src, dtype=torch.int64),
            torch.tensor(s_dst, dtype=torch.int64),
        )
    g = dgl.heterograph(edata_dict, num_nodes_dict={
        "encounter": N, "provider": n_prov,
    }).to(dev)
    if add_temporal:
        dow = pd.to_datetime(merged["SurgeryDate"], errors="coerce").dt.dayofweek
        dow = dow.fillna(0).astype(int).values
        weekend = (dow >= 5).astype(np.float32)
        temp = np.stack([dow.astype(np.float32) / 7.0,
                         weekend], axis=1)  # [N, 2]
        e_temp = torch.tensor(temp[e_src_ep], device=dev)
        g.edges["treated_by"].data["t"] = e_temp
        g.edges["treats"].data["t"] = e_temp
    return g


class HG2(nn.Module):
    """Two-layer bipartite mean-aggregation encoder (RGCN-like)."""
    def __init__(self, d_enc, h=HID, use_temp=False):
        super().__init__()
        self.lin_e1 = nn.Linear(d_enc, h)
        self.lin_p1 = nn.Linear(d_enc, h)
        self.lin_e2 = nn.Linear(h, h)
        self.lin_p2 = nn.Linear(h, h)
        self.use_temp = use_temp
        if use_temp:
            self.lin_t = nn.Linear(2, h)
        self.head = nn.Linear(h, 2)
    def _step(self, g, e, p, lin_e, lin_p):
        import dgl.function as fn
        g.nodes["encounter"].data["h"] = e
        g.nodes["provider"].data["h"]  = p
        if self.use_temp:
            g.apply_edges(lambda edges: {"m": edges.src["h"]
                                              + self.lin_t(edges.data["t"])},
                          etype="treated_by")
            g.update_all(fn.copy_e("m", "m"), fn.mean("m", "h2"), etype="treated_by")
            g.apply_edges(lambda edges: {"m": edges.src["h"]
                                              + self.lin_t(edges.data["t"])},
                          etype="treats")
            g.update_all(fn.copy_e("m", "m"), fn.mean("m", "h2"), etype="treats")
        else:
            g.multi_update_all({
                "treated_by": (fn.copy_u("h", "m"), fn.mean("m", "h2")),
                "treats":     (fn.copy_u("h", "m"), fn.mean("m", "h2")),
            }, "sum")
        # similarity-to-encounter (if edges exist)
        if ("encounter", "similar_to", "encounter") in g.canonical_etypes:
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_sim"),
                         etype="similar_to")
            e_new = torch.relu(lin_e(g.nodes["encounter"].data["h2"]
                                     + g.nodes["encounter"].data.get(
                                         "h_sim", torch.zeros_like(e))))
        else:
            e_new = torch.relu(lin_e(g.nodes["encounter"].data.get("h2", e)))
        p_new = torch.relu(lin_p(g.nodes["provider"].data.get("h2", p)))
        return e_new, p_new
    def forward(self, g, e_in, p_in):
        e = torch.relu(self.lin_e1(e_in))
        p = torch.relu(self.lin_p1(p_in))
        e, p = self._step(g, e, p, self.lin_e2, self.lin_p2)
        return e, self.head(e)


def train_hg2_extract(tr, va, add_sim_edges=None, add_temporal=False):
    g = _make_dgl_bipartite(add_sim_edges, add_temporal)
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True),
                           id_cols=[])
    Xe = torch.tensor(apply_preprocess(merged[feat_cols], st).astype(np.float32),
                      device=dev)
    Xp = torch.zeros(n_prov, Xe.shape[1], device=dev)
    net = HG2(Xe.shape[1], use_temp=add_temporal).to(dev)
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
        _, logits = net(g, Xe, Xp)
        loss = lf(logits[tr_t], Y[tr_t])
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            _, lv = net(g, Xe, Xp)
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
        e_emb, _ = net(g, Xe, Xp)
    return e_emb.cpu().numpy(), best


# ---------------------------------------------------------------------------
# Hypergraph propagation over provider-hyperedges (each provider = a hyperedge
# that connects all its encounters). Encoder emb -> RF downstream.
# ---------------------------------------------------------------------------

def hypergraph_prop(X_np, n_iters=2):
    """Simple hypergraph diffusion using provider-encounter incidence.
    X: [N, d] encounter features; returns propagated features same shape."""
    # Build sparse incidence H (N, M) where each edge groups encounters
    # sharing a provider. Row-normalize.
    import scipy.sparse as sp
    rows = e_src_ep; cols = e_dst_ep
    H = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(N, n_prov))
    # add self-loops (each encounter is its own hyperedge)
    self_H = sp.eye(N, dtype=np.float32, format="csr")
    H = sp.hstack([H, self_H]).tocsr()
    # D_v^-1 H D_e^-1 H^T for hypergraph Laplacian-style step
    d_v = np.asarray(H.sum(axis=1)).ravel()
    d_e = np.asarray(H.sum(axis=0)).ravel()
    D_v_inv = sp.diags(1.0 / np.clip(d_v, 1e-6, None))
    D_e_inv = sp.diags(1.0 / np.clip(d_e, 1e-6, None))
    step = D_v_inv @ H @ D_e_inv @ H.T
    X = X_np.copy()
    for _ in range(n_iters):
        X = 0.5 * X + 0.5 * (step @ X)
    return np.asarray(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# All-feature relational: encounter + provider + CPT nodes, RGCN-style enc.
# ---------------------------------------------------------------------------

def train_allfeat_extract(tr, va):
    import dgl
    import dgl.function as fn
    e_src_c = np.arange(N)
    e_dst_c = cpt_of
    n_cpt = len(cpt_ids)
    g = dgl.heterograph({
        ("encounter", "treated_by", "provider"):
            (torch.tensor(e_src_ep, dtype=torch.int64),
             torch.tensor(e_dst_ep, dtype=torch.int64)),
        ("provider", "treats", "encounter"):
            (torch.tensor(e_dst_ep, dtype=torch.int64),
             torch.tensor(e_src_ep, dtype=torch.int64)),
        ("encounter", "coded_as", "cpt"):
            (torch.tensor(e_src_c, dtype=torch.int64),
             torch.tensor(e_dst_c, dtype=torch.int64)),
        ("cpt", "codes", "encounter"):
            (torch.tensor(e_dst_c, dtype=torch.int64),
             torch.tensor(e_src_c, dtype=torch.int64)),
    }, num_nodes_dict={
        "encounter": N, "provider": n_prov, "cpt": n_cpt,
    }).to(dev)
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True),
                           id_cols=[])
    Xe = torch.tensor(apply_preprocess(merged[feat_cols], st).astype(np.float32),
                      device=dev)
    Xp = torch.zeros(n_prov, Xe.shape[1], device=dev)
    Xc = torch.zeros(n_cpt,  Xe.shape[1], device=dev)

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.le1 = nn.Linear(d, HID); self.lp1 = nn.Linear(d, HID)
            self.lc1 = nn.Linear(d, HID)
            self.le2 = nn.Linear(HID, HID)
            self.head = nn.Linear(HID, 2)
        def forward(self, g, xe, xp, xc):
            e = torch.relu(self.le1(xe)); p = torch.relu(self.lp1(xp))
            c = torch.relu(self.lc1(xc))
            g.nodes["encounter"].data["h"] = e
            g.nodes["provider"].data["h"]  = p
            g.nodes["cpt"].data["h"]       = c
            g.multi_update_all({
                "treated_by": (fn.copy_u("h", "m"), fn.mean("m", "h2")),
                "treats":     (fn.copy_u("h", "m"), fn.mean("m", "h2")),
                "coded_as":   (fn.copy_u("h", "m"), fn.mean("m", "h_c")),
                "codes":      (fn.copy_u("h", "m"), fn.mean("m", "h_c")),
            }, "sum")
            e_new = torch.relu(self.le2(
                g.nodes["encounter"].data.get("h2", e)
                + g.nodes["encounter"].data.get("h_c", torch.zeros_like(e))
            ))
            return e_new, self.head(e_new)

    net = Net(Xe.shape[1]).to(dev)
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
        _, logits = net(g, Xe, Xp, Xc)
        loss = lf(logits[tr_t], Y[tr_t])
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            _, lv = net(g, Xe, Xp, Xc)
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
        e_emb, _ = net(g, Xe, Xp, Xc)
    return e_emb.cpu().numpy(), best


# ---------------------------------------------------------------------------
# CV driver
# ---------------------------------------------------------------------------
MODELS = ["rf_clin", "gnn_prov_rf", "hypergraph_prov_rf", "gru_orders_only",
          "sim_edges_rf", "temporal_prov_rf", "allfeat_rel_rf"]
oof = {m: np.full(N, np.nan) for m in MODELS}


# --- Load cached rf_clin from Table 2 OOF; it's on the same N, same folds ---
_cache = np.load(OUT_DIR / "plos_table2_oof.npz")
assert np.array_equal(_cache["y"], y_all), "y mismatch with cached table2 OOF"
oof["rf_clin"] = _cache["rf_clin"].copy()
print(f"[t3] loaded cached rf_clin OOF from plos_table2_oof.npz", flush=True)


def _build_xtab(tr, extra=None):
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True),
                           id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    blocks = [Xtab, ohe.transform(cpt_arr)]
    if extra is not None:
        sc = StandardScaler().fit(extra[tr])
        blocks.append(sc.transform(extra))
    X = np.hstack(blocks)
    return StandardScaler().fit(X[tr]).transform(X)


def _rf_oof(X, tr, te):
    est = CalibratedClassifierCV(RF(), method="isotonic", cv=3).fit(X[tr], y_all[tr])
    return est.predict_proba(X[te])[:, 1]


skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y_all)):
    rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]
    print(f"[t3] === fold {fi+1}/{K} ===", flush=True)

    # gnn_prov_rf: 2-layer bipartite encoder over A2, embedding + RF
    emb_prov, vb_p = train_hg2_extract(tr, va)
    X_prov = _build_xtab(tr, extra=emb_prov)
    oof["gnn_prov_rf"][te] = _rf_oof(X_prov, tr, te)

    # hypergraph_prov_rf: hypergraph diffusion of tabular through provider-hyperedges
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True),
                           id_cols=[])
    Xtab_np = apply_preprocess(merged[feat_cols], st).astype(np.float32)
    Xtab_hyp = hypergraph_prop(Xtab_np, n_iters=2)
    X_hyp = _build_xtab(tr, extra=Xtab_hyp)
    oof["hypergraph_prov_rf"][te] = _rf_oof(X_hyp, tr, te)

    # gru_orders_only: no tabular, calibrated OOF via IsotonicRegression from val
    p_gru_all, vb_gru = train_gru_only(tr, va)
    # isotonic calibrate on val (like the paper uses val for threshold)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_gru_all[va], y_all[va])
    oof["gru_orders_only"][te] = iso.transform(p_gru_all[te])

    # sim_edges_rf: enc<->enc kNN similarity (train-only fit), embedding + RF
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True),
                           id_cols=[])
    Xtab_np = apply_preprocess(merged[feat_cols], st).astype(np.float32)
    knn = NearestNeighbors(n_neighbors=6, n_jobs=-1).fit(Xtab_np[tr])
    _, nbrs = knn.kneighbors(Xtab_np)  # for all rows; neighbor indices into tr subset
    s_src, s_dst = [], []
    for i in range(N):
        for k in range(1, nbrs.shape[1]):  # skip self
            j = tr[nbrs[i, k]]
            if j == i: continue
            s_src.append(i); s_dst.append(int(j))
    emb_sim, vb_sim = train_hg2_extract(tr, va,
                                        add_sim_edges=(np.array(s_src),
                                                       np.array(s_dst)))
    X_sim = _build_xtab(tr, extra=emb_sim)
    oof["sim_edges_rf"][te] = _rf_oof(X_sim, tr, te)

    # temporal_prov_rf: same encoder, but edges carry surgery-DOW temporal feat
    emb_temp, vb_t = train_hg2_extract(tr, va, add_temporal=True)
    X_temp = _build_xtab(tr, extra=emb_temp)
    oof["temporal_prov_rf"][te] = _rf_oof(X_temp, tr, te)

    # allfeat_rel_rf: enc + prov + cpt heterograph, RGCN-like encoder + RF
    emb_af, vb_af = train_allfeat_extract(tr, va)
    X_af = _build_xtab(tr, extra=emb_af)
    oof["allfeat_rel_rf"][te] = _rf_oof(X_af, tr, te)

    fold_line = f"[t3] fold {fi+1} val AUCs -> prov {vb_p:.3f} gru {vb_gru:.3f} sim {vb_sim:.3f} temp {vb_t:.3f} af {vb_af:.3f}"
    print(fold_line, flush=True)
    for m in MODELS:
        if not np.isnan(oof[m][te]).any():
            au = roc_auc_score(y_all[te], oof[m][te])
            ap = average_precision_score(y_all[te], oof[m][te])
            print(f"  {m:22s} test AUROC {au:.3f} AUPRC {ap:.3f}", flush=True)

# ---------------------------------------------------------------------------
# Metrics + threshold-based operating point on pooled OOF
# ---------------------------------------------------------------------------
for m in MODELS:
    assert not np.isnan(oof[m]).any(), f"NaN in {m} OOF"
np.savez(OUT_DIR / "plos_table3_oof.npz", y=y_all,
         **{m: oof[m] for m in MODELS})


def _f1_optimal(y_true, p):
    precisions, recalls, thresholds = precision_recall_curve(y_true, p)
    f1s = (2 * precisions * recalls) / np.maximum(precisions + recalls, 1e-9)
    best = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    return float(thresholds[min(best, len(thresholds) - 1)])


def _op_metrics(y_true, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    f1 = f1_score(y_true, pred, zero_division=0)
    prec = precision_score(y_true, pred, zero_division=0)
    rec  = recall_score(y_true, pred, zero_division=0)
    spec = tn / max(tn + fp, 1)
    flag = float(pred.mean())
    return dict(f1=float(f1), precision=float(prec), recall=float(rec),
                specificity=float(spec), flag_rate=flag)


def _f1_at_fixed_thr(y_true, p_scores, thr):
    """For bootstrap CI on F1 with a fixed (full-sample) threshold."""
    pred = (p_scores >= thr).astype(int)
    return f1_score(y_true, pred, zero_division=0)


rows = []
print(f"\n=== PLOS Table 3 (calibrated pooled OOF; N={N:,}; base "
      f"{y_all.mean()*100:.2f}%; bootstrap n={N_BOOT}) ===")
for m in MODELS:
    p = oof[m]
    au = float(roc_auc_score(y_all, p))
    ap = float(average_precision_score(y_all, p))
    br = float(brier_score_loss(y_all, p))
    au_lo, au_hi = _bootstrap_ci(y_all, p, roc_auc_score, N_BOOT, 0.05, 0)
    ap_lo, ap_hi = _bootstrap_ci(y_all, p, average_precision_score, N_BOOT, 0.05, 1)
    thr = _f1_optimal(y_all, p)
    op = _op_metrics(y_all, p, thr)
    # bootstrap CI on F1 at fixed threshold
    def _f1_boot(yt, ps): return _f1_at_fixed_thr(yt, ps, thr)
    f1_lo, f1_hi = _bootstrap_ci(y_all, p, _f1_boot, N_BOOT, 0.05, 2)
    rows.append(dict(model=m, auroc=au, auroc_lo=au_lo, auroc_hi=au_hi,
                     auprc=ap, auprc_lo=ap_lo, auprc_hi=ap_hi, brier=br,
                     opt_thr=thr, **op,
                     f1_lo=f1_lo, f1_hi=f1_hi))
    print(f"  {m:22s} AUROC {au:.3f} ({au_lo:.3f}-{au_hi:.3f})  "
          f"AUPRC {ap:.3f} ({ap_lo:.3f}-{ap_hi:.3f})  Brier {br:.4f}  "
          f"thr {thr:.3f}  F1 {op['f1']:.3f} ({f1_lo:.3f}-{f1_hi:.3f})  "
          f"P {op['precision']:.3f}  R {op['recall']:.3f}  "
          f"Spec {op['specificity']:.3f}  Flag {op['flag_rate']*100:.1f}%")

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "plos_table3_results.csv", index=False)
print(f"\n[t3] saved OOF -> {OUT_DIR/'plos_table3_oof.npz'}")
print(f"[t3] saved results -> {OUT_DIR/'plos_table3_results.csv'}")
