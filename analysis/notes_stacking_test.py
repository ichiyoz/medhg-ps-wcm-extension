"""Notes ⊕ order-GRU ⊕ providers STACKING test — do they add together, or
redundant? Same calibrated-pooled-OOF + bootstrap-CI protocol as
table2_orders_final.py. Restricted to notes-covered subgroup (n≈12,454) so the
informative-missingness confound doesn't inflate deltas.

Rows:
  1. tab                         RF on Xtab
  2. tab+prov                    + provider block
  3. tab+gru                     + order-GRU emb
  4. tab+notes                   + notes mean-pool
  5. tab+prov+notes              + prov + notes
  6. tab+gru+notes               + order-GRU + notes           ← key question
  7. tab+gru+prov                + order-GRU + prov            (previous best on subgroup)
  8. tab+gru+prov+notes          all four                       ← everything
  9. gru+notes                   order-GRU + notes (no tab)
 10. notes+prov                  notes + prov (no tab)
 11. notes_only                  notes alone

Protocol: RF canonical (500 trees, min_leaf 10, sqrt, balanced), isotonic-
calibrated (cv=3) pooled OOF, 5-fold seed 42, bootstrap n=2,000 CIs on
AUROC/AUPRC. GRU trained per fold on train rows only.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (apply_preprocess, fit_preprocess, load_raw,
                           add_calendar_features, load_order_sequence,
                           collapse_order_runs, build_provider_team_features)
from medhg_ps.deploy import _load_cpt_map
from medhg_ps.evaluate import _bootstrap_ci
from medhg_ps.train import set_seed, _resolve_device

SEED, N_BOOT = 42, 2000
K, VAL_FRAC = 5, 0.10
MAX_EPOCHS, PATIENCE = 120, 12
MAXLEN = 128
EMB_DIM, HID, NUMF = 24, 48, 5
DEV = _resolve_device(C.DEFAULTS_TRAIN.device)
NOTES_NPZ = "artifacts/newdata/notes_embeddings_raw.npz"


def RF() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                  max_features="sqrt", class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


# ============================================================================
# 1. Load cohort + tabular + provider block
# ============================================================================
print("[stk] loading...", flush=True)
cpt_map = _load_cpt_map()
raw = load_raw()
enc_nodupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
merged = (raw.encounters.merge(enc_nodupes, on="LogID", how="inner")
          .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]], on="LogID", how="inner")
          .reset_index(drop=True))
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)

prov = build_provider_team_features(raw.enc_prov_edges, raw.prov_attrs, raw.encounters)
merged = merged.merge(prov, on="LogID", how="left")
for c in C.PROVIDER_FEATURE_COLUMNS:
    merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
             + [c for c in C.CALENDAR_FEATURE_COLUMNS if c in merged.columns])
prov_cols = [c for c in C.PROVIDER_FEATURE_COLUMNS if c in merged.columns]
y_all = merged["ReadmittedWithin30Days"].astype(int).values
N_full = len(merged)
cpt_arr_full = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)


# ============================================================================
# 2. Order-token sequences (same setup as cv_seq_gru_orders.py)
# ============================================================================
orders = collapse_order_runs(load_order_sequence())
orders["LogID"] = orders["LogID"].astype(str)
orders["RunStart"] = pd.to_datetime(orders["RunStart"], errors="coerce")
orders = orders.sort_values(["LogID", "SeqInEncounter"])
vocab = orders["OrderGroup"].value_counts().index.tolist()
T2I = {t: i for i, t in enumerate(vocab)}
PAD = len(vocab)

row_of_full = {l: i for i, l in enumerate(merged["LogID"])}
seq_idx_full = np.full((N_full, MAXLEN), PAD, dtype=np.int64)
seq_num_full = np.zeros((N_full, MAXLEN, NUMF), dtype=np.float32)
lengths_full = np.ones(N_full, dtype=np.int64)
for lid, grp in orders.groupby("LogID", sort=False):
    r = row_of_full.get(lid)
    if r is None:
        continue
    grp = grp.tail(MAXLEN)
    toks = grp["OrderGroup"].tolist()
    reps = grp["RepeatCount"].tolist()
    gaps = grp["MinutesFromPrev"].tolist()
    times = grp["RunStart"].tolist()
    L = len(toks); lengths_full[r] = max(L, 1)
    for j in range(L):
        seq_idx_full[r, j] = T2I.get(toks[j], PAD)
        gap = gaps[j]; gap = 0.0 if pd.isna(gap) else max(float(gap), 0.0)
        t = times[j]; hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        seq_num_full[r, j] = [np.log1p(max(float(reps[j]), 0.0)),
                              np.log1p(gap),
                              (j + 1) / MAXLEN,
                              np.sin(2 * np.pi * hour / 24.0),
                              np.cos(2 * np.pi * hour / 24.0)]

print(f"[stk] full cohort N={N_full}  base={y_all.mean()*100:.2f}%  vocab={len(vocab)}  "
      f"median seq={int(np.median(lengths_full))}  max seq={int(lengths_full.max())}", flush=True)


# ============================================================================
# 3. Notes embeddings + subgroup restriction
# ============================================================================
z = np.load(NOTES_NPZ, allow_pickle=True)
E_notes = z["E"]                                         # (47268, 768)
notes_lid = z["log_id"].astype(str)
notes_ntc = z["note_type_code"].astype(str)
notes_df = pd.DataFrame({"LogID": notes_lid, "NoteTypeCode": notes_ntc,
                         "_rowidx": np.arange(len(notes_lid))})
notes_lids_set = set(notes_df["LogID"].unique())

has_notes_full = merged["LogID"].isin(notes_lids_set).values
sub_mask = has_notes_full
sub_rows = np.where(sub_mask)[0]
N = int(sub_mask.sum())
merged_sub = merged.iloc[sub_rows].reset_index(drop=True)
y = y_all[sub_rows]
cpt_arr = cpt_arr_full[sub_rows]
seq_idx = seq_idx_full[sub_rows]
seq_num = seq_num_full[sub_rows]
lengths = lengths_full[sub_rows]
sub_case_ids = merged_sub["LogID"].values

# Per-case mean-pool notes embedding (aligned to sub_rows order)
notes_by_lid = notes_df.groupby("LogID")["_rowidx"].apply(list).to_dict()
dim = E_notes.shape[1]
notes_mean = np.zeros((N, dim), dtype=np.float32)
for i, lid in enumerate(sub_case_ids):
    ridx = notes_by_lid.get(lid)
    if ridx:
        notes_mean[i] = E_notes[np.asarray(ridx)].mean(0)

print(f"[stk] subgroup N={N} ({N/N_full*100:.1f}% of cohort)  base={y.mean()*100:.2f}%  "
      f"notes_dim={dim}", flush=True)


# ============================================================================
# 4. Per-fold GRU (train on subgroup train rows only)
# ============================================================================
class SeqGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(PAD + 1, EMB_DIM, padding_idx=PAD)
        self.gru = nn.GRU(EMB_DIM + NUMF, HID, batch_first=True)
        self.head = nn.Linear(HID, 2)

    def encode(self, idx, num, lens):
        x = torch.cat([self.emb(idx), num], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]

    def forward(self, idx, num, lens):
        return self.head(self.encode(idx, num, lens))


def train_gru(tr: np.ndarray, va: np.ndarray) -> np.ndarray:
    set_seed(SEED)
    net = SeqGRU().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    npos, nneg = float((y[tr] == 1).sum()), float((y[tr] == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)],
                     dtype=torch.float32, device=DEV)
    lf = nn.CrossEntropyLoss(weight=w)
    Xi = torch.tensor(seq_idx, device=DEV); Xn = torch.tensor(seq_num, device=DEV)
    Ln = torch.tensor(lengths, device=DEV); Y = torch.tensor(y, device=DEV)
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
        try:
            vauc = roc_auc_score(y[va], pv)
        except ValueError:
            vauc = 0.5
        if vauc > best:
            best, state, pat = vauc, {k: v.detach().cpu().clone()
                                      for k, v in net.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        emb = net.encode(Xi, Xn, Ln).cpu().numpy()
    return emb.astype(np.float32)


# ============================================================================
# 5. Design + calibrated OOF
# ============================================================================
def build_xtab(tr: np.ndarray) -> np.ndarray:
    _, st = fit_preprocess(merged_sub[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged_sub[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    return np.hstack([Xtab, ohe.transform(cpt_arr)])


ROWS = [
    # (name, use_tab, use_prov, use_gru, use_notes)
    ("tab",                  True,  False, False, False),
    ("tab+prov",             True,  True,  False, False),
    ("tab+gru",              True,  False, True,  False),
    ("tab+notes",            True,  False, False, True ),
    ("tab+prov+notes",       True,  True,  False, True ),
    ("tab+gru+notes",        True,  False, True,  True ),
    ("tab+gru+prov",         True,  True,  True,  False),
    ("tab+gru+prov+notes",   True,  True,  True,  True ),
    ("gru+notes",            False, False, True,  True ),
    ("notes+prov",           False, True,  False, True ),
    ("notes_only",           False, False, False, True ),
]

oof: dict[str, np.ndarray] = {name: np.full(N, np.nan) for name, *_ in ROWS}
skf = StratifiedKFold(K, shuffle=True, random_state=SEED)
prov_block = merged_sub[prov_cols].values.astype(np.float32)

for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
    t0 = time.time()
    rng = np.random.default_rng(SEED + fi)
    tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]

    Xtab = build_xtab(tr)                          # (N, ftab)  fit on train
    gru_emb = train_gru(tr, va)                    # (N, HID) — expensive
    sc_gru = StandardScaler().fit(gru_emb[tr])
    gru_z = sc_gru.transform(gru_emb).astype(np.float32)
    sc_notes = StandardScaler().fit(notes_mean[tr])
    notes_z = sc_notes.transform(notes_mean).astype(np.float32)

    for name, use_tab, use_prov, use_gru, use_notes in ROWS:
        blocks = []
        if use_tab:   blocks.append(Xtab)
        if use_prov:  blocks.append(prov_block)
        if use_gru:   blocks.append(gru_z)
        if use_notes: blocks.append(notes_z)
        X = np.hstack(blocks)
        # Standardize the whole design fit on train (matches cv_seq_gru_orders)
        sc = StandardScaler(with_mean=False if X.shape[1] > 1500 else True).fit(X[tr])
        X = sc.transform(X)
        est = CalibratedClassifierCV(RF(), method="isotonic", cv=3).fit(X[tr], y[tr])
        oof[name][te] = est.predict_proba(X[te])[:, 1]
    print(f"[stk] fold {fi+1}/{K}  elapsed {time.time()-t0:.0f}s", flush=True)

# ============================================================================
# 6. Metrics + bootstrap CIs
# ============================================================================
results = []
for name, *_ in ROWS:
    p = oof[name]
    assert not np.isnan(p).any(), f"{name} has NaN OOF entries"
    au = roc_auc_score(y, p); ap = average_precision_score(y, p); br = brier_score_loss(y, p)
    au_lo, au_hi = _bootstrap_ci(y, p, roc_auc_score, n_boot=N_BOOT, seed=0)
    ap_lo, ap_hi = _bootstrap_ci(y, p, average_precision_score, n_boot=N_BOOT, seed=1)
    results.append(dict(name=name, auroc=au, au_lo=au_lo, au_hi=au_hi,
                        auprc=ap, ap_lo=ap_lo, ap_hi=ap_hi, brier=br))

res = pd.DataFrame(results)
res.to_csv("artifacts/newdata/notes_stacking_results.csv", index=False)
np.savez_compressed("artifacts/newdata/notes_stacking_oof.npz", y=y,
                    log_id=sub_case_ids.astype(object),
                    **{name: oof[name] for name, *_ in ROWS})

print(f"\n=== Notes ⊕ order-GRU ⊕ prov STACKING (subgroup N={N}, base {y.mean()*100:.2f}%) ===")
print(f"{'Model':<24} {'AUROC (95% CI)':<24} {'AUPRC (95% CI)':<24} {'Brier':>7}")
for r in results:
    print(f"{r['name']:<24} {r['auroc']:.3f} ({r['au_lo']:.3f}-{r['au_hi']:.3f})  "
          f"{r['auprc']:.3f} ({r['ap_lo']:.3f}-{r['ap_hi']:.3f})  {r['brier']:.4f}")

# Key deltas
def get(name):
    r = next(x for x in results if x["name"] == name)
    return r["auroc"], r["auprc"]

au_tab, ap_tab = get("tab")
au_gru, ap_gru = get("tab+gru")
au_gpr, ap_gpr = get("tab+gru+prov")

print("\n=== Key deltas (paired on identical OOF) ===")
for name in [n for n, *_ in ROWS]:
    if name == "tab":
        continue
    au, ap = get(name)
    print(f"  {name:<24} vs tab               dAUROC {au-au_tab:+.4f}  dAUPRC {ap-ap_tab:+.4f}")
print()
print(f"  tab+gru+notes vs tab+gru      dAUROC {get('tab+gru+notes')[0]-au_gru:+.4f}  "
      f"dAUPRC {get('tab+gru+notes')[1]-ap_gru:+.4f}   (does notes add to GRU?)")
print(f"  tab+gru+prov+notes vs tab+gru+prov  dAUROC {get('tab+gru+prov+notes')[0]-au_gpr:+.4f}  "
      f"dAUPRC {get('tab+gru+prov+notes')[1]-ap_gpr:+.4f}   (does notes add to deployable?)")
au_tn, ap_tn = get("tab+notes")
au_all, ap_all = get("tab+gru+prov+notes")
print(f"  tab+gru+prov+notes vs tab+notes    dAUROC {au_all-au_tn:+.4f}  "
      f"dAUPRC {ap_all-ap_tn:+.4f}   (does GRU+prov add to notes?)")

# Best by each metric
best_au = max(results, key=lambda r: r["auroc"])
best_ap = max(results, key=lambda r: r["auprc"])
print(f"\nBest AUROC:  {best_au['name']}  {best_au['auroc']:.3f}")
print(f"Best AUPRC:  {best_ap['name']}  {best_ap['auprc']:.3f}")
print("\n[stk] saved: artifacts/newdata/notes_stacking_results.csv")
print("[stk] saved: artifacts/newdata/notes_stacking_oof.npz")
