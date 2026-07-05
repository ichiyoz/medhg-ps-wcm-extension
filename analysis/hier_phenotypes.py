"""Hierarchical outcome-then-sequence phenotyping (Variant B).

For EACH substrate (care-unit sequence, order sequence):

  Step 1  Supervised sequence embedding    -- train GRU end-to-end on all rows
                                              (BCE, class-weighted). Extract
                                              final hidden state H_enc.
  Step 2  OOF risk score p_risk             -- 5-fold StratifiedKFold(seed 42).
                                              GRU trained per fold; hold-out
                                              softmax(pos) is the leak-free
                                              per-patient risk.
  Step 3  Outcome tiers (1D SMD partition) -- sort by p_risk; K_max=6 equal
                                              -frequency bins; greedy merge the
                                              ADJACENT pair with smallest SMD on
                                              the gold outcome until:
                                                (a) all pairwise SMD >= 0.30,
                                                (b) every bin has n >= 100,
                                                (c) K_tier >= 2.
                                              Falls back to target 0.20 if no
                                              solution at 0.30; reports failure
                                              otherwise.
  Step 4  Within-tier k-means               -- standardize H_enc within the
                                              tier; try K_sub in {1..4}; pick
                                              K_sub maximizing silhouette with
                                              all sub-sizes >= 50 (else K_sub=1).
  Step 5  Characterize (tier x sub)          -- n / readmit / substrate signature
                                              (top over-represented tokens or
                                              units) / clinical signature.
  Step 6  Cross-substrate concordance        -- Cohen kappa on tiers; Jaccard on
                                              high-tier membership; divergent
                                              patient description.

Outputs:
    artifacts/newdata/hier_phenotypes.log
    artifacts/newdata/hier_phenotypes_care.csv       LogID,tier,sub_phenotype
    artifacts/newdata/hier_phenotypes_order.csv      LogID,tier,sub_phenotype
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (apply_preprocess, fit_preprocess, load_raw,
                            add_calendar_features, build_preop_trajectory_features,
                            load_order_sequence, collapse_order_runs)
from medhg_ps.deploy import _load_cpt_map, UNIT_BUCKET
from medhg_ps.train import set_seed, _resolve_device

SEED = 42
dev = _resolve_device(C.DEFAULTS_TRAIN.device)
N_SPLITS = 5
MIN_TIER = 100                    # min patients per outcome tier
MIN_SUB  = 50                     # min patients per sub-phenotype
SMD_TARGET = 0.30                 # primary target
SMD_RELAX  = 0.20                 # fallback if primary infeasible

OUTDIR = Path("/Users/yiyezhang/Library/CloudStorage/Dropbox/Surgery/medhg-ps/artifacts/newdata")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# 0. LOAD data (gold label)
# ==========================================================================
print("[hier] loading raw + gold label...", flush=True)
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
merged = merged.merge(build_preop_trajectory_features(raw.enc_unit_edges, ss),
                     on="LogID", how="left")
for c in C.TRAJECTORY_FEATURE_COLUMNS:
    merged[c] = merged[c].fillna(0)
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)

# gold label
gold = pd.read_parquet(
    "/Users/yiyezhang/Downloads/medhg_ps_data/bulk_features_with_label_gold.parquet")[
    ["LogID","ReadmittedWithin30Days_gold"]]
gold["LogID"] = gold["LogID"].astype(str)
merged = merged.merge(gold, on="LogID", how="left")
y = merged["ReadmittedWithin30Days_gold"].astype(int).values
N = len(merged); base_rate = y.mean()
print(f"[hier] cohort N={N}  gold base rate {base_rate*100:.2f}%", flush=True)

log_ids = merged["LogID"].values


# ==========================================================================
# 1. CARE-UNIT SEQUENCE substrate: build indexed tensors
# ==========================================================================
print("\n[hier] building care-unit sequence tensors...", flush=True)
UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]
U2I = {u: i for i, u in enumerate(UNITS)}
PAD_U = len(UNITS)
MAXLEN_U = 40
EMB_DIM_U, HID_U, NUMF_U = 16, 32, 4

def _norm(s): return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)

try:
    u_ref = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
    u_ref["cid"] = u_ref["Clarity_ID"].astype("Int64").astype(str)
    u_ref["g"] = u_ref["UnitType"].map(UNIT_BUCKET).fillna("Other")
    ud = u_ref.dropna(subset=["Clarity_ID"]).drop_duplicates("cid", keep="first").set_index("cid")
except Exception as e:
    print(f"[hier] Unit_Names.xlsx unusable ({e}); using A3 UnitType as-is")
    ud = None

a3 = raw.enc_unit_edges.copy()
if ud is not None:
    a3["UnitType"] = _norm(a3["DepartmentID"]).map(ud["g"]).fillna(a3["UnitType"])
a3["InTime"] = pd.to_datetime(a3["InTime"], errors="coerce")
a3["Hours"] = pd.to_numeric(a3.get("Hours"), errors="coerce").fillna(0.0)
a3["LogID"] = a3["LogID"].astype(str)
a3 = a3.sort_values(["LogID", "InTime"])

row_of = {l: i for i, l in enumerate(merged["LogID"])}
seq_idx_u = np.full((N, MAXLEN_U), PAD_U, dtype=np.int64)
seq_num_u = np.zeros((N, MAXLEN_U, NUMF_U), dtype=np.float32)
lengths_u = np.ones(N, dtype=np.int64)
per_enc_units = {}
for lid, grp in a3.groupby("LogID", sort=False):
    r = row_of.get(lid)
    if r is None: continue
    units = grp["UnitType"].tolist()
    per_enc_units[lid] = units
    hrs = grp["Hours"].tolist(); times = grp["InTime"].tolist()
    steps = [(U2I.get(un, U2I["Other"]), h, t) for un, h, t in zip(units, hrs, times)]
    steps = steps[:MAXLEN_U]
    L = len(steps); lengths_u[r] = max(L, 1)
    for j, (ui, h, t) in enumerate(steps):
        seq_idx_u[r, j] = ui
        hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        seq_num_u[r, j] = [np.log1p(max(h, 0.0)),
                          np.sin(2 * np.pi * hour / 24.0),
                          np.cos(2 * np.pi * hour / 24.0),
                          (j + 1) / MAXLEN_U]


class UnitGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(PAD_U + 1, EMB_DIM_U, padding_idx=PAD_U)
        self.gru = nn.GRU(EMB_DIM_U + NUMF_U, HID_U, batch_first=True)
        self.head = nn.Linear(HID_U, 2)
    def encode(self, idx, num, lens):
        x = torch.cat([self.emb(idx), num], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]
    def forward(self, idx, num, lens):
        return self.head(self.encode(idx, num, lens))


# ==========================================================================
# 2. ORDER SEQUENCE substrate: build tensors
# ==========================================================================
print("\n[hier] building order sequence tensors...", flush=True)
MAXLEN_O = 128
EMB_DIM_O, HID_O, NUMF_O = 24, 48, 5

orders = collapse_order_runs(load_order_sequence())
orders["LogID"] = orders["LogID"].astype(str)
orders["RunStart"] = pd.to_datetime(orders["RunStart"], errors="coerce")
orders = orders.sort_values(["LogID", "SeqInEncounter"])
vocab = orders["OrderGroup"].value_counts().index.tolist()
T2I = {t: i for i, t in enumerate(vocab)}
PAD_O = len(vocab)

seq_idx_o = np.full((N, MAXLEN_O), PAD_O, dtype=np.int64)
seq_num_o = np.zeros((N, MAXLEN_O, NUMF_O), dtype=np.float32)
lengths_o = np.ones(N, dtype=np.int64)
per_enc_orders = {}
for lid, grp in orders.groupby("LogID", sort=False):
    r = row_of.get(lid)
    if r is None: continue
    grp = grp.tail(MAXLEN_O)
    toks = grp["OrderGroup"].tolist()
    per_enc_orders[lid] = toks
    reps = grp["RepeatCount"].tolist()
    gaps = grp["MinutesFromPrev"].tolist()
    times = grp["RunStart"].tolist()
    L = len(toks); lengths_o[r] = max(L, 1)
    for j in range(L):
        seq_idx_o[r, j] = T2I.get(toks[j], PAD_O)
        gap = gaps[j]; gap = 0.0 if pd.isna(gap) else max(float(gap), 0.0)
        t = times[j]; hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        seq_num_o[r, j] = [np.log1p(max(float(reps[j]), 0.0)),
                          np.log1p(gap),
                          (j + 1) / MAXLEN_O,
                          np.sin(2 * np.pi * hour / 24.0),
                          np.cos(2 * np.pi * hour / 24.0)]


class OrderGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(PAD_O + 1, EMB_DIM_O, padding_idx=PAD_O)
        self.gru = nn.GRU(EMB_DIM_O + NUMF_O, HID_O, batch_first=True)
        self.head = nn.Linear(HID_O, 2)
    def encode(self, idx, num, lens):
        x = torch.cat([self.emb(idx), num], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]
    def forward(self, idx, num, lens):
        return self.head(self.encode(idx, num, lens))


# ==========================================================================
# GRU training helpers (all-data + 5-fold OOF risk score)
# ==========================================================================
def _train_epoch(net, opt, lf, tr, Xi, Xn, Ln, Y, batch=4096, rng=None):
    net.train()
    perm = rng.permutation(tr) if rng is not None else tr
    for s in range(0, len(perm), batch):
        b = perm[s:s + batch]
        opt.zero_grad()
        loss = lf(net(Xi[b], Xn[b], Ln[b]), Y[b])
        loss.backward(); opt.step()


def train_gru_all(GRUcls, Xi, Xn, Ln, y_arr, tag, max_epochs=60, patience=8, batch=4096):
    """Train GRU on all rows (90% train / 10% val early-stopping); return encoder embedding."""
    set_seed(SEED)
    net = GRUcls().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    npos, nneg = float((y_arr == 1).sum()), float((y_arr == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)],
                     dtype=torch.float32, device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    Y = torch.tensor(y_arr, device=dev)
    rng = np.random.default_rng(SEED)
    idx_all = rng.permutation(len(y_arr))
    nv = int(0.1 * len(y_arr))
    va = idx_all[:nv]; tr = idx_all[nv:]
    best, state, pat = -1.0, None, patience
    for ep in range(max_epochs):
        _train_epoch(net, opt, lf, tr, Xi, Xn, Ln, Y, batch=batch, rng=rng)
        net.eval()
        with torch.no_grad():
            pv = torch.softmax(net(Xi[va], Xn[va], Ln[va]), -1)[:, 1].cpu().numpy()
        try: vauc = roc_auc_score(y_arr[va], pv)
        except ValueError: vauc = 0.5
        if vauc > best:
            best, state, pat = vauc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, patience
        else:
            pat -= 1
            if pat <= 0: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        emb = net.encode(Xi, Xn, Ln).cpu().numpy()
    print(f"[hier] {tag} GRU (all-data) best val AUROC {best:.3f}  emb shape {emb.shape}", flush=True)
    return emb


def oof_risk_gru(GRUcls, Xi, Xn, Ln, y_arr, tag, max_epochs=40, patience=6, batch=4096):
    """5-fold OOF predicted risk (softmax positive). No leakage."""
    p_risk = np.full(len(y_arr), np.nan)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    Y = torch.tensor(y_arr, device=dev)
    for fi, (tr, te) in enumerate(skf.split(np.zeros(len(y_arr)), y_arr)):
        set_seed(SEED + fi)
        net = GRUcls().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        npos, nneg = float((y_arr[tr] == 1).sum()), float((y_arr[tr] == 0).sum())
        w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)],
                         dtype=torch.float32, device=dev)
        lf = nn.CrossEntropyLoss(weight=w)
        rng = np.random.default_rng(SEED + fi)
        # carve inner val for early stopping
        nv = int(0.1 * len(tr))
        idx_tr = tr.copy(); rng.shuffle(idx_tr)
        va = idx_tr[:nv]; inner_tr = idx_tr[nv:]
        best, state, pat = -1.0, None, patience
        for ep in range(max_epochs):
            _train_epoch(net, opt, lf, inner_tr, Xi, Xn, Ln, Y, batch=batch, rng=rng)
            net.eval()
            with torch.no_grad():
                pv = torch.softmax(net(Xi[va], Xn[va], Ln[va]), -1)[:, 1].cpu().numpy()
            try: vauc = roc_auc_score(y_arr[va], pv)
            except ValueError: vauc = 0.5
            if vauc > best:
                best, state, pat = vauc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, patience
            else:
                pat -= 1
                if pat <= 0: break
        net.load_state_dict(state); net.eval()
        with torch.no_grad():
            p_risk[te] = torch.softmax(net(Xi[te], Xn[te], Ln[te]), -1)[:, 1].cpu().numpy()
        print(f"[hier] {tag} fold {fi+1}/{N_SPLITS} val AUROC {best:.3f}", flush=True)
    oof_auc = roc_auc_score(y_arr, p_risk)
    print(f"[hier] {tag} OOF AUROC = {oof_auc:.3f}", flush=True)
    return p_risk


# ==========================================================================
# Step 3 helper: SMD-constrained 1D outcome-tier partition
# ==========================================================================
def _pair_smd(p1, p2):
    d1, d2 = p1 * (1 - p1), p2 * (1 - p2)
    return abs(p1 - p2) / (np.sqrt((d1 + d2) / 2) + 1e-9)


def _tier_summary(assignments, y_arr):
    """Return list of (rate, n, [row_ids]) sorted ascending by rate."""
    K = int(assignments.max()) + 1
    summ = []
    for k in range(K):
        m = assignments == k
        n = int(m.sum())
        r = float(y_arr[m].mean()) if n else float("nan")
        summ.append((r, n, np.where(m)[0]))
    summ.sort(key=lambda t: t[0])
    return summ


def smd_partition(p_risk, y_arr, k_max=6, target=SMD_TARGET, min_size=MIN_TIER):
    """Sort by p_risk; K_max equal-frequency bins on p_risk; greedy adjacent-pair
    merge (by lowest SMD on outcome) until all pairs meet SMD >= target and every
    bin has >= min_size. Returns (tier_ids [N], meta_dict) or (None, meta_dict)
    if infeasible.
    """
    order = np.argsort(p_risk, kind="mergesort")
    # equal-frequency bins on ranks
    K = k_max
    bin_ids = np.empty(len(p_risk), dtype=int)
    cut_points = np.linspace(0, len(p_risk), K + 1).astype(int)
    for k in range(K):
        bin_ids[order[cut_points[k]:cut_points[k + 1]]] = k
    # now iteratively merge
    while True:
        summ = _tier_summary(bin_ids, y_arr)
        # remap contiguous
        rates = [r for r, n, _ in summ]
        ns    = [n for r, n, _ in summ]
        # check constraints
        # adjacent SMDs sorted by risk rate
        adj_smds = [(_pair_smd(rates[i], rates[i + 1]), i) for i in range(len(rates) - 1)]
        all_pair_smds = [_pair_smd(rates[i], rates[j])
                         for i in range(len(rates)) for j in range(i + 1, len(rates))]
        min_adj = min(adj_smds, key=lambda t: t[0]) if adj_smds else (float("inf"), None)
        min_pair = min(all_pair_smds) if all_pair_smds else float("inf")
        min_n = min(ns) if ns else 0
        ok = (min_pair >= target and min_n >= min_size and len(rates) >= 2)
        if ok or len(rates) <= 2:
            break
        # merge the adjacent pair with smallest SMD (also merge if any tier is too small)
        merge_idx = min_adj[1]
        # if a tier is below min_size, prefer merging it with its adjacent neighbor
        small_tiers = [i for i, n in enumerate(ns) if n < min_size]
        if small_tiers:
            i = small_tiers[0]
            # merge with adjacent (prefer smaller-SMD side)
            if i == 0:
                merge_idx = 0
            elif i == len(ns) - 1:
                merge_idx = len(ns) - 2
            else:
                left_smd = _pair_smd(rates[i - 1], rates[i])
                right_smd = _pair_smd(rates[i], rates[i + 1])
                merge_idx = i - 1 if left_smd <= right_smd else i
        # execute merge of bins at indices (merge_idx, merge_idx+1) in the sorted order
        _, _, rows_lo = summ[merge_idx]
        _, _, rows_hi = summ[merge_idx + 1]
        # both get remapped to a new id; simplest: rebuild ids by rank-order
        new_ids = np.empty(len(p_risk), dtype=int)
        cur = 0
        for i_s, (_, _, rows_i) in enumerate(summ):
            if i_s == merge_idx + 1:
                new_ids[rows_i] = cur - 1     # merge into previous
            else:
                new_ids[rows_i] = cur
                cur += 1
        bin_ids = new_ids
    # final feasibility check
    summ = _tier_summary(bin_ids, y_arr)
    rates = [r for r, n, _ in summ]
    ns    = [n for r, n, _ in summ]
    pair_smds = [_pair_smd(rates[i], rates[j])
                 for i in range(len(rates)) for j in range(i + 1, len(rates))]
    meta = {"target": target,
            "K_tier": len(rates),
            "rates": rates,
            "ns": ns,
            "min_pair_smd": min(pair_smds) if pair_smds else float("nan"),
            "adj_smds": [_pair_smd(rates[i], rates[i + 1]) for i in range(len(rates) - 1)],
            "feasible": (len(rates) >= 2
                         and (min(pair_smds) if pair_smds else 0.0) >= target
                         and min(ns) >= min_size)}
    return bin_ids, meta


def smd_partition_with_fallback(p_risk, y_arr, k_max=6, min_size=MIN_TIER):
    """Try target=0.30; if infeasible, try 0.20; return (ids, meta with final target)."""
    ids, meta = smd_partition(p_risk, y_arr, k_max=k_max, target=SMD_TARGET, min_size=min_size)
    if meta["feasible"]:
        meta["used_target"] = SMD_TARGET
        return ids, meta
    ids2, meta2 = smd_partition(p_risk, y_arr, k_max=k_max, target=SMD_RELAX, min_size=min_size)
    meta2["used_target"] = SMD_RELAX
    meta2["primary_failed"] = True
    return (ids2, meta2) if meta2["feasible"] else (None, meta2)


# ==========================================================================
# Step 4 helper: within-tier k-means with silhouette + min size
# ==========================================================================
def within_tier_kmeans(H_enc, tier_ids, k_range=(1, 2, 3, 4), min_sub_size=MIN_SUB):
    """For each unique tier, sub-cluster H_enc rows. Return array of sub-phenotype
    ids (0-indexed within tier) and metadata dict."""
    N = len(tier_ids)
    sub_ids = np.zeros(N, dtype=int)
    meta = {}
    for t in sorted(set(tier_ids.tolist())):
        mask = tier_ids == t
        rows = np.where(mask)[0]
        n_t = len(rows)
        X_t = H_enc[rows]
        # standardize within tier
        sc = StandardScaler().fit(X_t)
        X_t_s = sc.transform(X_t)
        best_k, best_sil, best_labels = 1, -1.0, np.zeros(n_t, dtype=int)
        for k in k_range:
            if k == 1:
                labels = np.zeros(n_t, dtype=int)
                sil = 0.0
                cluster_sizes = [n_t]
            else:
                if n_t < k * min_sub_size:
                    continue
                km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_t_s)
                labels = km.labels_
                sizes, _ = np.histogram(labels, bins=np.arange(k + 1))
                if sizes.min() < min_sub_size:
                    continue
                # silhouette on a random subsample if large (fast)
                sub_n = min(n_t, 3000)
                idx_s = np.random.default_rng(SEED).choice(n_t, sub_n, replace=False)
                try:
                    sil = silhouette_score(X_t_s[idx_s], labels[idx_s])
                except Exception:
                    sil = -1.0
                cluster_sizes = sizes.tolist()
            if k == 1 and best_k == 1 and best_sil < 0:
                best_k, best_sil, best_labels = 1, 0.0, labels
                continue
            if sil > best_sil:
                best_k, best_sil, best_labels = k, sil, labels
        sub_ids[rows] = best_labels
        meta[t] = {"n": n_t, "K_sub": best_k, "silhouette": best_sil}
        print(f"[hier]   tier {t}: n={n_t:5d}  K_sub={best_k}  silhouette={best_sil:+.3f}",
              flush=True)
    return sub_ids, meta


# ==========================================================================
# Signature helpers
# ==========================================================================
def unit_signature(row_indices, per_enc_units_local, log_ids_arr, top_n=3):
    """Return dict with top unit-type over-representation + top care paths (collapsed)."""
    from collections import Counter
    # 14-bucket unit-type composition (we use the 6-bucket collapsed here since UNITS)
    unit_counts = Counter()
    path_counts = Counter()
    n = 0
    for r in row_indices:
        lid = log_ids_arr[r]
        units = per_enc_units_local.get(lid, [])
        if not units: continue
        n += 1
        for u in units:
            unit_counts[u] += 1
        # collapsed adjacent duplicates
        collapsed = []
        for u in units:
            if not collapsed or collapsed[-1] != u:
                collapsed.append(u)
        path_counts["->".join(collapsed)] += 1
    return {
        "n_with_units": n,
        "unit_share": {u: unit_counts[u] / max(n, 1) for u in UNITS},
        "top_paths": path_counts.most_common(top_n),
    }


def unit_signature_lift(row_indices, per_enc_units_local, log_ids_arr,
                        cohort_unit_share, top_n=3):
    """Same but reports UNIT LIFT vs cohort (share_in_tier / share_cohort)."""
    sig = unit_signature(row_indices, per_enc_units_local, log_ids_arr, top_n=top_n)
    lift = {}
    for u in UNITS:
        base = max(cohort_unit_share.get(u, 1e-6), 1e-6)
        lift[u] = sig["unit_share"][u] / base
    sig["unit_lift"] = lift
    return sig


def order_signature_lift(row_indices, per_enc_orders_local, log_ids_arr,
                          cohort_token_share, top_n=5, min_sup=50):
    """Top over-represented OrderGroup tokens (lift vs cohort) with min support."""
    from collections import Counter
    tok_counts = Counter()
    n = 0
    for r in row_indices:
        lid = log_ids_arr[r]
        toks = per_enc_orders_local.get(lid, [])
        if not toks: continue
        n += 1
        # count each unique token once per encounter (presence not count)
        for t in set(toks):
            tok_counts[t] += 1
    lifts = []
    for t, c in tok_counts.items():
        if c < min_sup: continue
        share = c / max(n, 1)
        base  = max(cohort_token_share.get(t, 1e-6), 1e-6)
        lifts.append((t, share / base, c))
    lifts.sort(key=lambda x: -x[1])
    return {"n_with_orders": n, "top_tokens": lifts[:top_n]}


CLIN_KEYS = [("AgeYears", "age"), ("ASAClass", "ASA"),
             ("HCT", "HCT"), ("ALB", "ALB"), ("Creat", "Creat"),
             ("PatientType", "patient_type")]


def clinical_signature(row_indices):
    sig = {}
    d = merged.iloc[row_indices]
    for key, alias in CLIN_KEYS:
        if key == "PatientType":
            vals = d[key].astype(str)
            sig["pct_inpatient"] = float((vals == "I").mean())
        elif key == "ASAClass":
            vals = pd.to_numeric(d[key], errors="coerce")
            sig["ASA_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            sig["ASA_ge3_pct"] = float((vals >= 3).mean()) if vals.notna().any() else float("nan")
        else:
            vals = pd.to_numeric(d[key], errors="coerce")
            sig[f"{alias}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
    return sig


# ==========================================================================
# Cohort baselines for lift
# ==========================================================================
cohort_unit_share = {u: sum(units.count(u) for units in per_enc_units.values())
                     / max(1, sum(len(us) for us in per_enc_units.values()))
                     for u in UNITS}
_tok_present = {}
for lid, toks in per_enc_orders.items():
    for t in set(toks):
        _tok_present[t] = _tok_present.get(t, 0) + 1
n_with_orders_all = len(per_enc_orders)
cohort_token_share = {t: c / max(1, n_with_orders_all) for t, c in _tok_present.items()}


# ==========================================================================
# Run BOTH substrates
# ==========================================================================
def process_substrate(name, GRUcls, seq_idx, seq_num, lengths, per_enc, tag):
    print(f"\n============================================================")
    print(f"[hier] SUBSTRATE: {name}")
    print(f"============================================================", flush=True)
    Xi = torch.tensor(seq_idx, device=dev)
    Xn = torch.tensor(seq_num, device=dev)
    Ln = torch.tensor(lengths, device=dev)

    # Step 1: supervised embedding (all-data)
    H_enc = train_gru_all(GRUcls, Xi, Xn, Ln, y, f"{tag}-all")
    # Step 2: OOF risk
    p_risk = oof_risk_gru(GRUcls, Xi, Xn, Ln, y, f"{tag}-oof")

    # Step 3: SMD-constrained 1D partition
    tier_ids, tier_meta = smd_partition_with_fallback(p_risk, y, k_max=6, min_size=MIN_TIER)
    if tier_ids is None:
        print(f"[hier] {name}: INFEASIBLE — no partition meets SMD>={SMD_RELAX} at n>={MIN_TIER}")
        return None
    print(f"\n[hier] {name} — tier partition (target={tier_meta['used_target']}):")
    print(f"  K_tier={tier_meta['K_tier']}  min_pair_SMD={tier_meta['min_pair_smd']:.3f}  "
          f"tier rates={[f'{r*100:.2f}%' for r in tier_meta['rates']]}  "
          f"tier ns={tier_meta['ns']}")
    print(f"  adjacent SMDs: {[f'{s:.3f}' for s in tier_meta['adj_smds']]}")

    # Step 4: within-tier k-means
    sub_ids, sub_meta = within_tier_kmeans(H_enc, tier_ids)

    return {
        "name": name, "tag": tag,
        "H_enc": H_enc, "p_risk": p_risk,
        "tier_ids": tier_ids, "tier_meta": tier_meta,
        "sub_ids": sub_ids, "sub_meta": sub_meta,
    }


# Run substrate 1: CARE-UNIT
care_res = process_substrate("CARE-UNIT SEQUENCE", UnitGRU,
                              seq_idx_u, seq_num_u, lengths_u, per_enc_units, "care")

# Run substrate 2: ORDER
order_res = process_substrate("ORDER SEQUENCE", OrderGRU,
                               seq_idx_o, seq_num_o, lengths_o, per_enc_orders, "order")


# ==========================================================================
# Step 5: Characterize each (tier × sub) cell
# ==========================================================================
def characterize(res, sig_fn):
    if res is None:
        return
    name = res["name"]
    tier_ids = res["tier_ids"]; sub_ids = res["sub_ids"]
    K_tier = int(tier_ids.max()) + 1
    print(f"\n============================================================")
    print(f"[hier] CHARACTERIZATION — {name}")
    print(f"============================================================", flush=True)
    for t in range(K_tier):
        mask_t = tier_ids == t
        n_t = int(mask_t.sum())
        r_t = float(y[mask_t].mean()) if n_t else float("nan")
        print(f"\n  TIER {t}: n={n_t:5d} ({n_t/N*100:5.1f}%)  readmit={r_t*100:.2f}%")
        for sk in sorted(set(sub_ids[mask_t].tolist())):
            mask_ts = mask_t & (sub_ids == sk)
            rows = np.where(mask_ts)[0]
            n_ts = int(mask_ts.sum())
            r_ts = float(y[mask_ts].mean()) if n_ts else float("nan")
            clin = clinical_signature(rows)
            sig  = sig_fn(rows)
            print(f"    sub {sk}: n={n_ts:5d} ({n_ts/N*100:.1f}%)  readmit={r_ts*100:.2f}%  "
                  f"inpt={clin['pct_inpatient']*100:4.1f}%  ASA_mean={clin.get('ASA_mean', float('nan')):.2f}  "
                  f"ASA>=3={clin.get('ASA_ge3_pct', float('nan'))*100:4.1f}%  "
                  f"age={clin.get('age_mean', float('nan')):.1f}  HCT={clin.get('HCT_mean', float('nan')):.1f}  "
                  f"ALB={clin.get('ALB_mean', float('nan')):.2f}  Cr={clin.get('Creat_mean', float('nan')):.2f}")
            if "top_paths" in sig:
                paths = "; ".join(f"{p}({c})" for p, c in sig["top_paths"])
                top_lift_units = sorted(sig["unit_lift"].items(), key=lambda x: -x[1])[:3]
                lift_str = "  ".join(f"{u}:{v:.2f}x" for u, v in top_lift_units)
                print(f"          unit_lift: {lift_str}")
                print(f"          top_paths: {paths}")
            if "top_tokens" in sig:
                tok_str = "  ".join(f"{t}({lift:.2f}x, n={c})" for t, lift, c in sig["top_tokens"])
                print(f"          top OrderGroup lift: {tok_str}")


characterize(care_res,  lambda rows: unit_signature_lift(rows, per_enc_units, log_ids, cohort_unit_share))
characterize(order_res, lambda rows: order_signature_lift(rows, per_enc_orders, log_ids, cohort_token_share))


# ==========================================================================
# Step 6: Cross-substrate concordance
# ==========================================================================
if care_res is not None and order_res is not None:
    print(f"\n============================================================")
    print(f"[hier] CROSS-SUBSTRATE CONCORDANCE")
    print(f"============================================================", flush=True)
    t_care  = care_res["tier_ids"]
    t_order = order_res["tier_ids"]

    # Tier-level Cohen kappa (labels are already ordered by risk within each substrate)
    kappa = cohen_kappa_score(t_care, t_order)
    print(f"  Cohen kappa on tier labels: {kappa:+.3f}")

    # High-tier membership overlap (Jaccard)
    care_high = t_care == t_care.max()
    order_high = t_order == t_order.max()
    inter = int((care_high & order_high).sum())
    union = int((care_high | order_high).sum())
    jacc = inter / max(union, 1)
    print(f"  HIGH-tier sizes:  care={int(care_high.sum())}  order={int(order_high.sum())}")
    print(f"  HIGH-tier intersection: {inter}  union: {union}  Jaccard: {jacc:.3f}")

    # divergent: high on one, low on the other
    care_low = t_care == 0
    order_low = t_order == 0
    div_care_high_order_low = int((care_high & order_low).sum())
    div_order_high_care_low = int((order_high & care_low).sum())
    print(f"\n  DIVERGENT patients:")
    print(f"    care=HIGH  order=LOW : n={div_care_high_order_low}")
    print(f"    order=HIGH care=LOW  : n={div_order_high_care_low}")
    # readmit rates of the divergent groups
    for label, mask in [("care=HIGH order=LOW", care_high & order_low),
                         ("order=HIGH care=LOW", order_high & care_low),
                         ("both HIGH", care_high & order_high),
                         ("both LOW",  care_low  & order_low)]:
        if mask.sum():
            r = float(y[mask].mean()) * 100
            print(f"    {label:25s}: n={int(mask.sum()):5d}  readmit={r:5.2f}%")

# ==========================================================================
# Save per-patient assignments
# ==========================================================================
def _save(res, fname):
    if res is None:
        print(f"[hier] {fname}: substrate infeasible, no assignments saved")
        return
    df = pd.DataFrame({
        "LogID": log_ids,
        "tier": res["tier_ids"],
        "sub_phenotype": res["sub_ids"],
        "p_risk": res["p_risk"],
    })
    p = OUTDIR / fname
    df.to_csv(p, index=False)
    print(f"[hier] wrote {p}")


_save(care_res,  "hier_phenotypes_care.csv")
_save(order_res, "hier_phenotypes_order.csv")

print("\n[hier] DONE", flush=True)
