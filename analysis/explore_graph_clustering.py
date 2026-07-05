"""Explore outcome-driven (DICE) clustering across graph representations.

Base heterograph encounter nodes ALREADY carry the full clinical feature set
(X_enc = preprocessed merged[feat_cols]); the base GNN embedding is a
message-passed transform of clinical+structure, and clustering it gave
STRUCTURE phenotypes because message passing smooths clinical signal into the
provider/unit neighborhood. We test whether re-injecting / restructuring
clinical information shifts the phenotypes:

  V1  DICE on base GNN encounter embedding E                (reference)
  V3  DICE on [E (concat) standardized clinical features]   (un-smoothed clinical)
  V2  DICE on GNN embedding of the 5-node clinical-concept graph
      (encounter+provider+unit+diagnosis+procedure nodes)

For each: sweep K in {3,4,5,6} (surrogate sig for speed), report per-cluster
n+readmit, a CLINICAL vs STRUCTURAL character score (how well each modality
predicts cluster membership), and descriptive clinical+structural profiles.
"""
import os
os.environ.setdefault("DICE_SURROGATE_SIG", "1")           # speed across many fits
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import analysis.cv_dice_gnn as base                         # reuse merged, raw, feat_cols, gnn_embeddings, build_xtab
import analysis.dice as dice
merged, raw, feat_cols, N, y = base.merged, base.raw, base.feat_cols, base.N, base.y
NORM = lambda s: s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)

# ---------- structural feature block (for character scoring + profiles) -------
a3 = raw.enc_unit_edges.copy(); a3["LogID"] = NORM(a3["LogID"])
INST = sorted(a3["InstitutionType"].dropna().unique().tolist())
merged["LogID"] = NORM(merged["LogID"])
def struct_block():
    g = a3.groupby("LogID")
    rows = {}
    comp = a3.assign(one=1).pivot_table(index="LogID", columns="InstitutionType", values="one",
                                        aggfunc="sum", fill_value=0)
    comp = comp.div(comp.sum(1).clip(lower=1), axis=0)      # fraction of stays per 14-bucket
    nu = g["UnitType"].nunique().rename("n_units")
    icu = g["UnitType"].apply(lambda s: int((s == "Intensive").any())).rename("has_ICU")
    ed = g["UnitType"].apply(lambda s: int((s == "ED").any())).rename("has_ED")
    npr = NORM(raw.enc_prov_edges["LogID"]).value_counts().rename("n_prov")
    a1 = raw.encounters.copy(); a1["LogID"] = NORM(a1["LogID"]); a1["ps"] = NORM(a1["PrimarySurgID"])
    a4 = raw.prov_attrs.copy(); a4["pid"] = NORM(a4["ProvID"])
    vol = a1.merge(a4[["pid", "CaseVolume2yr"]], left_on="ps", right_on="pid", how="left") \
            .set_index("LogID")["CaseVolume2yr"]
    S = comp.join([nu, icu, ed]).join(npr).join(vol.rename("surg_vol"))
    S = merged[["LogID"]].merge(S, left_on="LogID", right_index=True, how="left").drop(columns="LogID")
    return S.fillna(0.0)
Xstruct = StandardScaler().fit_transform(struct_block().values)
Xclin = base.build_xtab(np.arange(N))                       # standardized clinical + CPT block
print(f"[explore] cohort={N:,} base={y.mean()*100:.2f}%  clin dims={Xclin.shape[1]} struct dims={Xstruct.shape[1]}", flush=True)

def character(labels):
    """5-fold accuracy of predicting cluster from clinical-only vs structure-only.
    Higher => that modality defines the clusters."""
    def acc(X):
        return cross_val_score(LogisticRegression(max_iter=200),
                               X, labels, cv=3, scoring="accuracy").mean()
    ac, as_ = acc(Xclin), acc(Xstruct)
    tag = "CLINICAL" if ac > as_ + 0.03 else ("STRUCTURE" if as_ > ac + 0.03 else "HYBRID")
    return ac, as_, tag

comps = merged  # alias
ASA = pd.to_numeric(merged["ASAClass"], errors="coerce")
alb = pd.to_numeric(merged["ALB"], errors="coerce"); hct = pd.to_numeric(merged["HCT"], errors="coerce")
age = pd.to_numeric(merged["AgeYears"], errors="coerce")
inpt = merged["PatientType"].astype(str).eq("I")
def profile(labels, K):
    out = []
    for k in range(K):
        m = labels == k
        if m.sum() == 0:
            continue
        out.append(dict(k=k, n=int(m.sum()), readmit=round(y[m].mean()*100, 1),
                        ASAge3=round((ASA[m] >= 3).mean()*100), inpt=round(inpt[m].mean()*100),
                        age=round(age[m].median()), alb=round(alb[m].median(), 1),
                        hct=round(hct[m].median(), 1)))
    return pd.DataFrame(out).sort_values("readmit")

def run_variant(name, X):
    print(f"\n########## {name}  (input dims={X.shape[1]}) ##########", flush=True)
    best = None
    for K in (3, 4, 5, 6):
        m = dice.fit(X, y, K=K, d=16, v=base.build_v(np.arange(N)), seed=42)
        lab = dice.cluster_proba(m, X).argmax(1)
        pops = np.bincount(lab, minlength=K)
        empty = int((pops < 30).sum())
        rates = [y[lab == k].mean() for k in range(K) if (lab == k).sum()]
        rr = max(rates)/min(rates) if len(rates) >= 2 and min(rates) > 0 else float("nan")
        ac, as_, tag = character(lab)
        print(f"  K={K} pops={pops.tolist()} empty(<30)={empty} riskRatio={rr:.2f} "
              f"| char: clin-acc {ac:.2f} vs struct-acc {as_:.2f} -> {tag}", flush=True)
        if empty == 0 and (best is None or K > best[0]):
            best = (K, lab, tag)
    if best:
        K, lab, tag = best
        print(f"  --> cleanest all-populated K={K} ({tag}); profile:")
        print(profile(lab, K).to_string(index=False))
        return dict(variant=name, K=K, character=tag)
    return dict(variant=name, K=None, character="degenerate")

# ---- shared base GNN embedding (train on 90%, embed all) ----
rng = np.random.default_rng(42); idx = rng.permutation(N)
val = idx[:int(0.1*N)]; tr = idx[int(0.1*N):]
print("[explore] training base GNN for embeddings...", flush=True)
E = base.gnn_embeddings(tr, val)
Escaled = StandardScaler().fit_transform(E)

results = []
results.append(run_variant("V1 base GNN embedding", Escaled))
results.append(run_variant("V3 embedding + raw clinical", np.hstack([Escaled, Xclin])))

# ---- V2: 5-node clinical-concept graph embedding ----
try:
    import analysis._concept_embed as ce  # optional helper if present
    E2 = ce.embed()
except Exception as e:
    E2 = None
    print(f"\n[explore] V2 (5-node concept-graph embedding) skipped: {e}", flush=True)
if E2 is not None:
    results.append(run_variant("V2 concept-graph embedding", StandardScaler().fit_transform(E2)))

pd.DataFrame(results).to_csv("artifacts/newdata/explore_graph_clustering.csv", index=False)
print("\n[explore] summary:", results, flush=True)
