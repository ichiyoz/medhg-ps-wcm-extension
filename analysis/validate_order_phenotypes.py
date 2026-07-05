"""Out-of-sample validation of the K=4 order-sequence DICE phenotypes.
5-fold: fit GRU-on-orders + DICE on TRAIN only, order the 4 clusters by TRAIN
readmission, assign HELD-OUT test patients, and report their test-set
readmission per tier. Checks whether the in-sample 3.3 / 5.3 / 12.3% gradient
holds out of sample, and whether the rehab high tier + infection-workup mid
tier recur across folds. Mirrors analysis/validate_phenotypes.py for the
care-graph phenotypes."""
from __future__ import annotations
import os
os.environ.setdefault("DICE_SURROGATE_SIG", "1")            # exact-LR-test is expensive per fold

from collections import Counter
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

import analysis.cv_dice_orders as O                          # reuses order-GRU + tokens + labels
import analysis.dice as dice

y_all, N = O.y_all, O.N
merged, enc_tokens = O.merged, O.enc_tokens
SEED, VAL_FRAC, K, D = 42, 0.10, 4, 16
INSAMPLE = [3.3, 5.3, 12.3, 12.3]                            # in-sample K=4 collapsed to 3 populated
TIERNAME = ["low", "mid1", "mid2", "high"]

# Cohort-level token frequency (for lift-based phenotype signature stability)
cohort_freq: dict[str, int] = {}
for toks in enc_tokens:
    for t in set(toks):
        cohort_freq[t] = cohort_freq.get(t, 0) + 1
cohort_pct = {t: v / N for t, v in cohort_freq.items()}

def top_tokens(mask, k=5, minsup=20):
    n = int(mask.sum())
    if n == 0: return []
    f: dict[str, int] = {}
    for i in np.where(mask)[0]:
        for t in set(enc_tokens[i]):
            f[t] = f.get(t, 0) + 1
    lifts = [(t, round((f[t] / n) / max(cohort_pct[t], 1e-9), 2), f[t])
             for t in f if cohort_pct.get(t, 0) > 0 and f[t] >= minsup]
    return sorted(lifts, key=lambda z: -z[1])[:k]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pooled_tier = np.full(N, -1)
rows = []
high_tokens_per_fold: list[list[tuple]] = []
mid_tokens_per_fold:  list[list[tuple]] = []

print(f"[val_ord] N={N} base={y_all.mean()*100:.2f}%  K={K}  (surrogate sig, per-fold GRU)\n",
      flush=True)
for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y_all)):
    rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N)); val, tr = tr_all[:nv], tr_all[nv:]
    full_tr = np.concatenate([tr, val])
    # 1) order-GRU trained on TRAIN only; embeddings extracted for all rows
    E = O.gru_embeddings(tr, val)
    # 2) DICE fit on TRAIN embeddings only; assignments via cluster_proba for all
    dm = dice.fit(E[full_tr], y_all[full_tr], K, D, O.build_v(full_tr)[full_tr])
    hard = dice.cluster_proba(dm, E).argmax(1)
    # 3) order the K clusters by TRAIN readmission -> low..high (mapping fixed on train)
    tr_rate = {k_: (y_all[full_tr][hard[full_tr] == k_].mean()
                    if (hard[full_tr] == k_).any() else np.nan)
               for k_ in range(K)}
    order = sorted(range(K), key=lambda k_: (tr_rate[k_] if tr_rate[k_] == tr_rate[k_] else 9))
    cl2tier = {cl: t for t, cl in enumerate(order)}
    tier = np.array([cl2tier[c] for c in hard])
    te_mask = np.zeros(N, bool); te_mask[te] = True
    pooled_tier[te] = tier[te]
    for t in range(K):
        m = (tier == t) & te_mask
        rows.append(dict(fold=fi, tier=t, n_train=int(((tier == t) & ~te_mask).sum()),
                         n_test=int(m.sum()),
                         test_readmit=float(y_all[m].mean()) if m.sum() else np.nan,
                         train_readmit=float(tr_rate[order[t]])
                         if tr_rate[order[t]] == tr_rate[order[t]] else np.nan))
    hi = tier == K - 1                                       # after ordering, K-1 == high
    mid = tier == 1                                          # first mid tier (mid1)
    ht = top_tokens(hi); mt = top_tokens(mid)
    high_tokens_per_fold.append(ht); mid_tokens_per_fold.append(mt)
    print(f"fold {fi}: TRAIN tiers "
          f"{[round(tr_rate[order[t]]*100,1) if tr_rate[order[t]]==tr_rate[order[t]] else 'NA' for t in range(K)]}%",
          flush=True)
    print(f"   HIGH n_test={int((hi & te_mask).sum())}  "
          f"top tokens: {[(t, l) for (t, l, _) in ht]}", flush=True)
    print(f"   MID1 n_test={int((mid & te_mask).sum())}  "
          f"top tokens: {[(t, l) for (t, l, _) in mt]}", flush=True)

df = pd.DataFrame(rows)
out_csv = "artifacts/newdata/validate_order_phenotypes_folds.csv"
df.to_csv(out_csv, index=False)

print("\n=== HELD-OUT TEST readmission per tier (mean +/- std across 5 folds) ===")
for t in range(K):
    s = df[df.tier == t]
    tr_mean = s.train_readmit.mean() * 100
    te_mean = s.test_readmit.mean() * 100
    te_std  = s.test_readmit.std() * 100
    print(f"  {TIERNAME[t]:4s}: test {te_mean:5.2f}% +/- {te_std:4.2f}   "
          f"(train {tr_mean:5.2f}%, in-sample {INSAMPLE[t]}%)")

print("\n=== POOLED held-out (each patient scored on its held-out fold) ===")
for t in range(K):
    m = pooled_tier == t
    r = y_all[m].mean() * 100 if m.sum() else float("nan")
    print(f"  {TIERNAME[t]:4s}: n={int(m.sum()):5d}  pooled test readmit {r:5.2f}%")

ratios = []
for f in range(5):
    hi_r = df[(df.fold == f) & (df.tier == K - 1)].test_readmit.values
    lo_r = df[(df.fold == f) & (df.tier == 0)].test_readmit.values
    if len(hi_r) and len(lo_r) and lo_r[0] > 0:
        ratios.append(round(hi_r[0] / lo_r[0], 2))
    else:
        ratios.append(float("nan"))
print(f"\nper-fold high/low TEST risk ratio: {ratios}  (in-sample ~3.71)")

print("\n=== Phenotype stability across folds — HIGH tier top-5 order tokens (lift) ===")
rehab_kw = ("SLP", "REHAB", "PT", "THERAP", "NEURO", "VASC")
inf_kw   = ("ISOLA", "MICRO", "CONSULT", "CASE REQUEST")
def match_any(tokens, kws):
    up = [t.upper() for t, _, _ in tokens]
    return sum(any(k in u for u in up) for k in kws)
for fi, ht in enumerate(high_tokens_per_fold):
    print(f"  fold {fi}: {[(t, l) for (t, l, _) in ht]}  "
          f"(rehab/geriatric hits={match_any(ht, rehab_kw)}/{len(rehab_kw)})")
print("\n=== Phenotype stability across folds — MID1 tier top-5 order tokens (lift) ===")
for fi, mt in enumerate(mid_tokens_per_fold):
    print(f"  fold {fi}: {[(t, l) for (t, l, _) in mt]}  "
          f"(infection/consult hits={match_any(mt, inf_kw)}/{len(inf_kw)})")

print(f"\nwrote {out_csv}")
