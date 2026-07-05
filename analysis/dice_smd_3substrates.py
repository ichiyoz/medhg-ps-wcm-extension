"""Improved DICE (LR-test G > 3.841 significance + SMD >= 0.30 effect-size)
applied to three substrates on the SAME cohort with the gold label:

  1. PATIENT ATTRIBUTES  : Xtab (clinical + one-hot CPT) -- "who they are"
  2. CARE-UNIT SEQUENCE  : A3 unit-seq GRU final hidden state -- "trajectory"
  3. ORDER SEQUENCE      : order-seq GRU final hidden state -- "what was done"

For each substrate, fit DICE (K=3, d=16, lam_smd=30, smd_target=0.30, spread=0).
Report per-tier n / readmit / pairwise SMDs + a substrate-specific signature.
Then cross-substrate: Adjusted Rand Index between cluster pairs, high-tier overlap.

Runs the GRU final-fit on ALL rows (descriptive, no CV).

Outputs:
    artifacts/newdata/dice_smd_3sub.log
    artifacts/newdata/dice_smd_3sub_patient.csv
    artifacts/newdata/dice_smd_3sub_unit.csv
    artifacts/newdata/dice_smd_3sub_order.csv
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (apply_preprocess, fit_preprocess, load_raw,
                            add_calendar_features, build_preop_trajectory_features,
                            load_order_sequence, collapse_order_runs)
from medhg_ps.deploy import _load_cpt_map, UNIT_BUCKET
from medhg_ps.train import set_seed, _resolve_device
import analysis.dice as dice

SEED = 42
dev = _resolve_device(C.DEFAULTS_TRAIN.device)

# ---- assemble merged frame with GOLD label -------------------------------
print("[3sub] loading raw + gold label...", flush=True)
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

# swap in gold label
gold = pd.read_parquet(
    "/Users/yiyezhang/Downloads/medhg_ps_data/bulk_features_with_label_gold.parquet")[
    ["LogID","ReadmittedWithin30Days_gold"]]
gold["LogID"] = gold["LogID"].astype(str)
merged = merged.merge(gold, on="LogID", how="left")
y = merged["ReadmittedWithin30Days_gold"].astype(int).values
N = len(merged); base_rate = y.mean()

feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
             + [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged.columns])
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)
print(f"[3sub] cohort N={N}  gold base rate {base_rate*100:.2f}%  feat_cols={len(feat_cols)}",
      flush=True)

# confounders (age + one-hot sex) — passed to DICE outcome head
_age = pd.to_numeric(merged["AgeYears"], errors="coerce").values.reshape(-1, 1)
_age = (np.nan_to_num(_age, nan=np.nanmedian(_age)) - np.nanmean(_age)) / (np.nanstd(_age) + 1e-8)
_gv = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(
    merged[["Gender"]].astype(str))
v_conf = np.hstack([_age, _gv]).astype(np.float32)

# ============================================================
# SUBSTRATE 1 — PATIENT ATTRIBUTES (Xtab)
# ============================================================
print("\n[3sub] building substrate 1: patient attributes (Xtab)...", flush=True)
_, st = fit_preprocess(merged[feat_cols], id_cols=[])
Xtab = apply_preprocess(merged[feat_cols], st)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr)
X_patient = StandardScaler().fit_transform(np.hstack([Xtab, ohe.transform(cpt_arr)]))
print(f"[3sub] X_patient shape: {X_patient.shape}", flush=True)

# ============================================================
# SUBSTRATE 2 — CARE-UNIT SEQUENCE (A3 GRU)
# ============================================================
print("\n[3sub] building substrate 2: unit-sequence GRU...", flush=True)
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
    print(f"[3sub] Unit_Names.xlsx not usable ({e}); using A3 UnitType as-is")
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
per_enc_units = {}                          # LogID -> collapsed unit tokens for signature
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

def train_gru_all(GRUcls, Xi, Xn, Ln, y_arr, tag, max_epochs=60, patience=8, batch=4096):
    set_seed(SEED)
    net = GRUcls().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    npos, nneg = float((y_arr == 1).sum()), float((y_arr == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)],
                     dtype=torch.float32, device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    Y = torch.tensor(y_arr, device=dev)
    # 90/10 train/val for early stopping
    rng = np.random.default_rng(SEED)
    idx_all = rng.permutation(len(y_arr))
    nv = int(0.1 * len(y_arr))
    va = idx_all[:nv]; tr = idx_all[nv:]
    best, state, pat = -1.0, None, patience
    for ep in range(max_epochs):
        net.train(); perm = rng.permutation(tr)
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            opt.zero_grad()
            loss = lf(net(Xi[b], Xn[b], Ln[b]), Y[b])
            loss.backward(); opt.step()
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
    print(f"[3sub] {tag} GRU best val AUROC {best:.3f}  emb shape {emb.shape}", flush=True)
    return emb

Xi_u = torch.tensor(seq_idx_u, device=dev)
Xn_u = torch.tensor(seq_num_u, device=dev)
Ln_u = torch.tensor(lengths_u, device=dev)
unit_emb = train_gru_all(UnitGRU, Xi_u, Xn_u, Ln_u, y, "unit-seq")
X_unit = StandardScaler().fit_transform(unit_emb)

# ============================================================
# SUBSTRATE 3 — ORDER SEQUENCE (order-seq GRU)
# ============================================================
print("\n[3sub] building substrate 3: order-sequence GRU...", flush=True)
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
per_enc_orders = {}                         # LogID -> order tokens (last MAXLEN)
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

Xi_o = torch.tensor(seq_idx_o, device=dev)
Xn_o = torch.tensor(seq_num_o, device=dev)
Ln_o = torch.tensor(lengths_o, device=dev)
order_emb = train_gru_all(OrderGRU, Xi_o, Xn_o, Ln_o, y, "order-seq")
X_order = StandardScaler().fit_transform(order_emb)

# ============================================================
# APPLY IMPROVED DICE (K=3, lam_smd=30, smd_target=0.30) TO EACH
# ============================================================
dice.LAM["smd"] = 30.0
print(f"\n[3sub] dice.LAM['smd']={dice.LAM['smd']}  smd_target=0.30  K=3  d=16", flush=True)

def fit_and_extract(name, X, K=3, d=16):
    print(f"\n[3sub] training DICE on {name}...", flush=True)
    m = dice.fit(X, y, K=K, d=d, v=v_conf, smd_target=0.30, spread=0.0, verbose=True)
    chat = dice.cluster_proba(m, X)
    hard = chat.argmax(1)
    return m, chat, hard

m_pat, chat_pat, hard_pat = fit_and_extract("PATIENT ATTRIBUTES", X_patient)
m_unit, chat_unit, hard_unit = fit_and_extract("CARE-UNIT SEQUENCE", X_unit)
m_ord, chat_ord, hard_ord = fit_and_extract("ORDER SEQUENCE", X_order)

# ============================================================
# TIER SUMMARY per substrate
# ============================================================
def pairwise_smd(rates):
    out = []
    for i in range(len(rates)):
        for j in range(i+1, len(rates)):
            p1, p2 = rates[i], rates[j]
            d1, d2 = p1*(1-p1), p2*(1-p2)
            out.append(abs(p1-p2) / (np.sqrt((d1+d2)/2) + 1e-6))
    return out

def tier_summary(name, hard, K=3):
    print(f"\n=== {name} — DICE K={K}, lam_smd=30, smd_target=0.30 ===")
    rows = sorted([(k, int((hard==k).sum()), float(y[hard==k].mean()) if (hard==k).any() else float('nan'))
                   for k in range(K)], key=lambda t: (t[2] if t[2]==t[2] else 9))
    for k, n, r in rows:
        print(f"  tier k={k}  n={n:5d} ({n/N*100:5.1f}%)  readmit={r*100:5.2f}%")
    rates = [r for _,n,r in rows if n>10 and r==r]
    if len(rates) >= 2:
        ratio = max(rates)/max(min(rates), 1e-6)
        smds = pairwise_smd(rates)
        print(f"  high/low ratio = {ratio:.2f}   pairwise SMDs = {[f'{s:.2f}' for s in smds]}   min SMD = {min(smds):.3f}")
    return rows

rows_pat = tier_summary("PATIENT ATTRIBUTES", hard_pat)
rows_unit = tier_summary("CARE-UNIT SEQUENCE", hard_unit)
rows_ord = tier_summary("ORDER SEQUENCE", hard_ord)

# ============================================================
# SUBSTRATE-SPECIFIC SIGNATURES (top over-represented features per tier)
# ============================================================
def order_tiers_by_risk(hard, K=3):
    """Return list of (k_original, rank_by_risk_low_to_high) so tier order matches display."""
    rates = [(k, y[hard==k].mean() if (hard==k).any() else np.nan) for k in range(K)]
    rates_valid = [(k, r) for k,r in rates if not np.isnan(r)]
    rates_valid.sort(key=lambda t: t[1])
    return {k: rank for rank,(k,_) in enumerate(rates_valid)}   # k -> rank_low_to_high

RANK_NAMES = ["LOW", "MID", "HIGH"]

print("\n=== SIGNATURES ===")

# ---- PATIENT ATTRIBUTES: top clinical features that differ from cohort mean ----
print("\n[PATIENT ATTRIBUTES] top features per tier (mean deviation from cohort)")
# use the raw feat_cols (numeric only, before one-hot) to keep interpretability
num_feats = [c for c in feat_cols if pd.to_numeric(merged[c], errors="coerce").notna().sum() > N * 0.5]
mat = np.array([pd.to_numeric(merged[c], errors="coerce").fillna(pd.to_numeric(merged[c], errors="coerce").median()).values
                for c in num_feats]).T   # N x nfeats
mat_z = (mat - mat.mean(0)) / (mat.std(0) + 1e-8)
rank_pat = order_tiers_by_risk(hard_pat)
for k in range(3):
    mask = hard_pat == k
    if mask.sum() < 10: continue
    dev = mat_z[mask].mean(0)
    top = np.argsort(np.abs(dev))[-6:][::-1]
    r = float(y[mask].mean())
    print(f"  {RANK_NAMES[rank_pat[k]]}-risk tier  (n={mask.sum()}, readmit={r*100:.1f}%): ",
          [f"{num_feats[i]}({dev[i]:+.2f}σ)" for i in top])

# ---- CARE-UNIT SEQUENCE: unit-bucket over-representation + top care paths ----
def collapse_runs(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x: out.append(x)
    return out

print("\n[CARE-UNIT SEQUENCE] unit-type frequency per tier + top paths")
UBUCKETS = UNITS  # ED / Acute / OR / Intensive / Intermediate / Other
rank_unit = order_tiers_by_risk(hard_unit)
for k in range(3):
    mask = hard_unit == k
    if mask.sum() < 10: continue
    tier_paths = []
    unit_touch = {u: 0 for u in UBUCKETS}
    n_touch = 0
    for lid in merged["LogID"].values[mask]:
        units = per_enc_units.get(lid, [])
        if not units:
            continue
        for u in set(units):
            if u in unit_touch: unit_touch[u] += 1
        n_touch += 1
        tier_paths.append("→".join(collapse_runs(units)[:6]))
    unit_pct = {u: (100*unit_touch[u]/max(n_touch,1)) for u in UBUCKETS}
    cohort_touch = {u: 0 for u in UBUCKETS}; c_n = 0
    for lid in merged["LogID"].values:
        units = per_enc_units.get(lid, [])
        if not units: continue
        for u in set(units):
            if u in cohort_touch: cohort_touch[u] += 1
        c_n += 1
    cohort_pct = {u: (100*cohort_touch[u]/max(c_n,1)) for u in UBUCKETS}
    lift = {u: unit_pct[u]/max(cohort_pct[u], 0.1) for u in UBUCKETS}
    top_units = sorted(UBUCKETS, key=lambda u: -lift[u])[:4]
    from collections import Counter
    top_paths = Counter(tier_paths).most_common(3)
    r = float(y[mask].mean())
    print(f"  {RANK_NAMES[rank_unit[k]]}-risk tier  (n={mask.sum()}, readmit={r*100:.1f}%)")
    print(f"    unit lifts: " + ", ".join(f"{u}: {lift[u]:.1f}x ({unit_pct[u]:.0f}%)" for u in top_units))
    print(f"    top paths:  " + " | ".join(f"{p}({n})" for p,n in top_paths))

# ---- ORDER SEQUENCE: top OrderGroup tokens by lift ----
print("\n[ORDER SEQUENCE] top OrderGroup tokens per tier (min support=50)")
rank_ord = order_tiers_by_risk(hard_ord)
# cohort baseline: fraction of encounters that ever ordered each token
from collections import Counter
tok_cohort = Counter()
n_with_orders_cohort = 0
for lid in merged["LogID"].values:
    toks = per_enc_orders.get(lid, [])
    if not toks: continue
    n_with_orders_cohort += 1
    for t in set(toks): tok_cohort[t] += 1
cohort_freq = {t: c/max(n_with_orders_cohort,1) for t,c in tok_cohort.items()}
for k in range(3):
    mask = hard_ord == k
    if mask.sum() < 10: continue
    tier_toks = Counter(); n_tier = 0
    for lid in merged["LogID"].values[mask]:
        toks = per_enc_orders.get(lid, [])
        if not toks: continue
        n_tier += 1
        for t in set(toks): tier_toks[t] += 1
    tier_freq = {t: c/max(n_tier,1) for t,c in tier_toks.items()}
    lifts = [(t, tier_freq[t]/max(cohort_freq.get(t,0.001),0.001), tier_toks[t])
             for t in tier_toks if tier_toks[t] >= 50]
    lifts.sort(key=lambda x: -x[1])
    r = float(y[mask].mean())
    print(f"  {RANK_NAMES[rank_ord[k]]}-risk tier  (n={mask.sum()}, readmit={r*100:.1f}%)")
    print("    top-6 lifted tokens: " + ", ".join(f"{t}({l:.1f}x, n={n})" for t,l,n in lifts[:6]))

# ============================================================
# CROSS-SUBSTRATE AGREEMENT
# ============================================================
print("\n=== CROSS-SUBSTRATE AGREEMENT ===")
ari_pu = adjusted_rand_score(hard_pat, hard_unit)
ari_po = adjusted_rand_score(hard_pat, hard_ord)
ari_uo = adjusted_rand_score(hard_unit, hard_ord)
print(f"  ARI  patient vs unit  = {ari_pu:+.3f}")
print(f"  ARI  patient vs order = {ari_po:+.3f}")
print(f"  ARI  unit    vs order = {ari_uo:+.3f}")

# high-tier overlap
hi_p = set(np.where(hard_pat == max(range(3), key=lambda k: y[hard_pat==k].mean() if (hard_pat==k).any() else -1))[0])
hi_u = set(np.where(hard_unit == max(range(3), key=lambda k: y[hard_unit==k].mean() if (hard_unit==k).any() else -1))[0])
hi_o = set(np.where(hard_ord  == max(range(3), key=lambda k: y[hard_ord==k].mean() if (hard_ord==k).any() else -1))[0])
union3 = hi_p | hi_u | hi_o
inter3 = hi_p & hi_u & hi_o
print(f"\n  HIGH-tier sizes: patient={len(hi_p)}, unit={len(hi_u)}, order={len(hi_o)}")
print(f"  union of HIGH tiers  = {len(union3)}")
print(f"  intersection (all 3) = {len(inter3)}   ({100*len(inter3)/max(len(union3),1):.1f}% of union)")
for name1,s1 in [("patient",hi_p),("unit",hi_u),("order",hi_o)]:
    for name2,s2 in [("patient",hi_p),("unit",hi_u),("order",hi_o)]:
        if name1 >= name2: continue
        j = len(s1 & s2) / max(len(s1 | s2), 1)
        print(f"  Jaccard {name1}-{name2}: {j:.3f}   overlap={len(s1&s2)}")

# ============================================================
# SAVE
# ============================================================
out_dir = Path("artifacts/newdata"); out_dir.mkdir(parents=True, exist_ok=True)
for tag, hard, chat in [("patient", hard_pat, chat_pat),
                         ("unit",    hard_unit, chat_unit),
                         ("order",   hard_ord,  chat_ord)]:
    dfout = pd.DataFrame({"LogID": merged["LogID"].values,
                          "y_gold": y,
                          "cluster": hard})
    for k in range(3):
        dfout[f"p_c{k}"] = chat[:, k]
    p = out_dir / f"dice_smd_3sub_{tag}.csv"
    dfout.to_csv(p, index=False)
    print(f"  saved {p}")

print("\n[3sub] DONE.", flush=True)
