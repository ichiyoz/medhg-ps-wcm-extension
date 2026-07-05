"""Out-of-sample validation of the K=3 graph-DICE care-delivery-clinical
phenotypes. 5-fold: fit GNN+DICE on TRAIN only, order the 3 clusters by TRAIN
readmission, assign HELD-OUT test patients, and report their test-set
readmission per tier. Checks whether the in-sample 1.2/8.5/17.3% gradient holds
out of sample, and whether the renal/transplant high tier + ambulatory low tier
recur across folds.
"""
from __future__ import annotations
import os
os.environ["MEDHG_PS_DEVICE"] = "cpu"          # avoid MPS contention with Table-3 suite
os.environ.setdefault("DICE_SURROGATE_SIG", "1")

from collections import Counter
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

import analysis.cv_dice_gnn as G               # reuses data load + gnn_embeddings + build_v
import analysis.dice as dice

merged, y, N, anyz = G.merged, G.y, G.N, G.anyz
SEED, VAL_FRAC, K, D = 42, 0.10, 3, 16
INSAMPLE = [1.2, 8.5, 17.3]
TIERNAME = ["low", "mid", "high"]

def flag(col):
    return merged[col].astype(str).str.strip().str.lower().isin(["yes", "insulin", "non-insulin"]).values
dial = flag("Preop Dialysis"); immuno = flag("Immunosuppressive Therapy")
outpt = merged["PatientType"].astype(str).str.strip().isin(["O", "Outpatient"]).values

# per-encounter set of specific units (for lift-based stability profiling)
a3 = G.raw.enc_unit_edges.copy(); a3["LogID"] = a3["LogID"].astype(str)
unit_by_log = a3.groupby("LogID")["DepartmentName"].apply(lambda s: set(s.dropna()))
enc_units = pd.Series(merged["LogID"].astype(str).values).map(unit_by_log).apply(
    lambda s: s if isinstance(s, set) else set())
cohort_cnt = Counter()
for s in enc_units: cohort_cnt.update(s)

def top_units(mask, k=3, minsup=20):
    cl = Counter()
    for s in enc_units[mask]: cl.update(s)
    n = int(mask.sum()); out = []
    for u, c in cl.items():
        if c < minsup: continue
        out.append((u, round((c / n) / max(cohort_cnt[u] / N, 1e-9), 1)))
    return sorted(out, key=lambda t: -t[1])[:k]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pooled = np.full(N, -1)
rows = []
print(f"[val] N={N} base={y.mean()*100:.2f}%  K={K}  (CPU, surrogate sig)\n", flush=True)
for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
    rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N)); val, tr = tr_all[:nv], tr_all[nv:]
    full_tr = np.concatenate([tr, val])
    E = G.gnn_embeddings(tr, val)                          # GNN trained on train only
    dm = dice.fit(E[full_tr], y[full_tr], K, D, G.build_v(full_tr)[full_tr])
    hard = dice.cluster_proba(dm, E).argmax(1)
    # order clusters by TRAIN readmission -> low/mid/high (mapping fixed from train)
    tr_rate = {k: (y[full_tr][hard[full_tr] == k].mean() if (hard[full_tr] == k).any() else np.nan)
               for k in range(K)}
    order = sorted(range(K), key=lambda k: tr_rate[k] if tr_rate[k] == tr_rate[k] else 9)
    cl2tier = {cl: t for t, cl in enumerate(order)}
    tier = np.array([cl2tier[c] for c in hard])
    te_mask = np.zeros(N, bool); te_mask[te] = True
    pooled[te] = tier[te]
    for t in range(K):
        m = (tier == t) & te_mask
        rows.append(dict(fold=fi, tier=t, n_test=int(m.sum()),
                         test_readmit=float(y[m].mean()) if m.sum() else np.nan,
                         train_readmit=float(tr_rate[order[t]])))
    hi = tier == 2; lo = tier == 0
    print(f"fold {fi}: TRAIN tiers {[round(tr_rate[order[t]]*100,1) for t in range(K)]}%", flush=True)
    print(f"   HIGH n={int(hi.sum())} dialysis={dial[hi].mean()*100:.0f}% immunosupp={immuno[hi].mean()*100:.0f}% "
          f"units={top_units(hi)}", flush=True)
    print(f"   LOW  n={int(lo.sum())} outpatient={outpt[lo].mean()*100:.0f}% units={top_units(lo)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("artifacts/newdata/validate_phenotypes_folds.csv", index=False)
print("\n=== HELD-OUT TEST readmission per tier (mean +/- std across 5 folds) ===")
for t in range(K):
    s = df[df.tier == t]
    print(f"  {TIERNAME[t]:4s}: test {s.test_readmit.mean()*100:5.2f}% +/- {s.test_readmit.std()*100:4.2f}   "
          f"(train {s.train_readmit.mean()*100:5.2f}%, in-sample {INSAMPLE[t]}%)")
print("\n=== POOLED held-out (each patient scored on its held-out fold) ===")
for t in range(K):
    m = pooled == t
    print(f"  {TIERNAME[t]:4s}: n={int(m.sum()):5d}  pooled test readmit {y[m].mean()*100:.2f}%")
ratios = [round(df[(df.fold == f) & (df.tier == 2)].test_readmit.values[0]
                 / max(df[(df.fold == f) & (df.tier == 0)].test_readmit.values[0], 1e-9), 1) for f in range(5)]
print(f"\nper-fold high/low TEST risk ratio: {ratios}  (in-sample ~14x)")
