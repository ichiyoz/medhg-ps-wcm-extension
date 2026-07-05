"""Care-delivery phenotyping of outcome-driven graph clusters (granular).

Fits DICE on the ie-HGCN encounter embeddings (final fit, all rows), assigns each
encounter to a cluster, then characterizes each cluster by GRANULAR care structure:
14-bucket unit composition, the specific named units most over-represented (by lift),
and provider-team composition (ProvType mix, surgeon volume, resident/fellow presence)
-- to test whether clustering the graph yields genuine care-delivery phenotypes
(distinct pathways/teams) or just an acuity gradient.

Reuses cv_dice_gnn's assembly + GNN final-fit machinery (imported).

    PYTHONPATH=. python analysis/phenotype_graph_clusters.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import analysis.cv_dice_gnn as base
raw, merged, y, N = base.raw, base.merged, base.y, base.N
dice, SEED, VAL_FRAC = base.dice, base.SEED, base.VAL_FRAC
dice.EXACT_SIG = True

import os
KS = tuple(int(x) for x in os.environ.get("PHENO_KS", "4,3").split(","))
OUT_CSV = os.environ.get(
    "PHENO_OUT",
    "/Users/yiyezhang/Library/CloudStorage/Dropbox/Surgery/medhg-ps/artifacts/newdata/graph_cluster_phenotypes.csv")
INST_CATS = ["Procedural Area", "Med/Surg", "ICU", "ED", "PICU",
             "Labor and Delivery", "Postpartum", "Pediatrics", "Antepartum", "Nursery"]
PROV_ROLES = ["Attending", "Resident", "Fellow", "Anesthesiologist", "Nurse Anesthetist"]


def collapse(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


lids = merged["LogID"].astype(str)
df = pd.DataFrame({"LogID": lids.values, "y": y})

# ---- units: coarse pathway + 14-bucket InstitutionType + dept sets ----------
a3 = raw.enc_unit_edges.copy(); a3["LogID"] = a3["LogID"].astype(str)
a3["InTime"] = pd.to_datetime(a3["InTime"], errors="coerce")
a3 = a3.sort_values(["LogID", "SeqInEncounter", "InTime"])
seqs = a3.groupby("LogID", sort=False)["UnitType"].apply(list)
df["has_ICU"] = df["LogID"].map(seqs.apply(lambda s: int("Intensive" in s)))
df["has_ED"] = df["LogID"].map(seqs.apply(lambda s: int("ED" in s)))
df["OR_only"] = df["LogID"].map(seqs.apply(lambda s: int(set(collapse(s)) == {"OR"})))
df["n_units"] = df["LogID"].map(seqs.apply(lambda s: len(set(s))))
df["traj"] = df["LogID"].map(seqs.apply(lambda s: "->".join(collapse(s)))).fillna("(none)")
inst = a3.groupby("LogID")["InstitutionType"].apply(lambda s: set(s.dropna()))
for cat in INST_CATS:
    df[f"inst::{cat}"] = df["LogID"].map(inst.apply(lambda st, c=cat: int(c in st))).fillna(0).astype(int)

# ---- providers/team: ProvType mix, surgeon volume, resident/fellow ----------
pa = raw.enc_prov_edges[["LogID", "ProvID"]].assign(LogID=lambda d: d["LogID"].astype(str))
pa["ptype"] = pa["ProvID"].map(raw.prov_attrs.set_index("ProvID")["ProvType"])
comp = (pa.pivot_table(index="LogID", columns="ptype", values="ProvID", aggfunc="count", fill_value=0))
for role in PROV_ROLES:
    df[f"n_{role.split()[0].lower()}"] = df["LogID"].map(comp[role]).fillna(0) if role in comp else 0
df["n_providers"] = df["LogID"].map(pa.groupby("LogID")["ProvID"].nunique())
df["has_resident"] = (df["n_resident"] > 0).astype(int)
df["has_fellow"] = (df["n_fellow"] > 0).astype(int)
vol = pd.to_numeric(raw.prov_attrs.set_index("ProvID")["CaseVolume2yr"], errors="coerce")
surg = raw.encounters.assign(LogID=lambda d: d["LogID"].astype(str)).set_index("LogID")["PrimarySurgID"]
df["surg_vol"] = df["LogID"].map(surg).map(vol)
hv = df["surg_vol"].quantile(0.75)
df["highvol_surg"] = (df["surg_vol"] >= hv).astype(int)

# ---- case mix ----
df["inpatient"] = (merged["PatientType"].astype(str).values == "I").astype(int)
df["asa"] = pd.to_numeric(merged["ASAClass"], errors="coerce").values
df["age"] = pd.to_numeric(merged["AgeYears"], errors="coerce").values
df["optime"] = pd.to_numeric(merged["CutToClose"], errors="coerce").values
df["cpt"] = merged["PrimaryCPT"].astype(str).values

# dept-level table (unique LogID x DepartmentName) for lift
ld = a3[["LogID", "DepartmentName"]].dropna().drop_duplicates()
cohort_dshare = ld.groupby("DepartmentName")["LogID"].nunique() / N


def fit_clusters():
    rng = np.random.default_rng(SEED); allr = np.arange(N); rng.shuffle(allr)
    nv = int(round(VAL_FRAC * N)); val, tr = allr[:nv], allr[nv:]
    E = base.gnn_embeddings(tr, val)
    v = base.build_v(np.arange(N))
    for K in KS:
        m = dice.fit(E, y, K, base.DD, v, lam_bal=1.0)
        hard = dice.cluster_proba(m, E).argmax(1)
        counts = np.bincount(hard, minlength=K)
        if (counts >= 30).all():                          # require genuinely populated tiers
            print(f"[pheno] using K={K}: {counts.tolist()}", flush=True)
            return hard, K
        print(f"[pheno] K={K} degenerate ({counts.tolist()}), trying smaller", flush=True)
    return hard, K


def top_depts(cluster_lids, n_k):
    sub = ld[ld["LogID"].isin(cluster_lids)]
    cnt = sub.groupby("DepartmentName")["LogID"].nunique()
    lift = (cnt / n_k / cohort_dshare).where(cnt >= 20).dropna().sort_values(ascending=False).head(5)
    return "; ".join(f"{d} (x{l:.1f}, n={int(cnt[d])})" for d, l in lift.items())


def name_cluster(p, depts):
    unit = ("ICU/critical-care" if p["inst::ICU"] >= 25 else
            "ambulatory/procedural" if p["pct_ORonly"] >= 40 else
            "ED-origin" if p["has_ED"] >= 40 else "inpatient-ward")
    acu = "high-acuity" if p["asa_ge3"] >= 55 else "low-acuity" if p["asa_ge3"] <= 35 else "mixed-acuity"
    setng = "inpatient" if p["inpatient"] >= 55 else "outpatient" if p["inpatient"] <= 35 else "mixed"
    team = "high-vol attendings" if p["pct_highvol"] >= 55 else "teaching (resident/fellow)" if p["pct_resident"] >= 55 else ""
    lead = depts.split(" (")[0] if depts else ""
    return f"{unit}, {setng}, {acu}" + (f"; {team}" if team else "") + (f"; lead unit: {lead}" if lead else "")


def main():
    hard, K = fit_clusters()
    df["cluster"] = hard
    rows = []
    for k in sorted(range(K), key=lambda k: df.loc[df.cluster == k, "y"].mean()):
        g = df[df.cluster == k]; n_k = len(g)
        depts = top_depts(set(g["LogID"]), n_k)
        p = {"cluster": k, "n": n_k, "pct": 100 * n_k / N, "readmit": 100 * g["y"].mean(),
             "inpatient": 100 * g["inpatient"].mean(), "asa_ge3": 100 * (g["asa"] >= 3).mean(),
             "has_ED": 100 * g["has_ED"].mean(), "pct_ORonly": 100 * g["OR_only"].mean(),
             "med_age": g["age"].median(), "med_optime": g["optime"].median(),
             "med_nprov": g["n_providers"].median(), "med_surgvol": g["surg_vol"].median(),
             "pct_highvol": 100 * g["highvol_surg"].mean(), "pct_resident": 100 * g["has_resident"].mean(),
             "pct_fellow": 100 * g["has_fellow"].mean()}
        for cat in INST_CATS:
            p[f"inst::{cat}"] = 100 * g[f"inst::{cat}"].mean()
        for role in PROV_ROLES:
            p[f"med_{role.split()[0].lower()}"] = g[f"n_{role.split()[0].lower()}"].median()
        p["top_depts_lift"] = depts
        p["top_traj"] = "; ".join(f"{t}({n})" for t, n in g["traj"].value_counts().head(2).items())
        p["top_cpt"] = "; ".join(f"{c}({n})" for c, n in g["cpt"].value_counts().head(3).items())
        p["name"] = name_cluster(p, depts)
        rows.append(p)
    P = pd.DataFrame(rows); P.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 230)
    print(f"\n=== GRANULAR care-delivery phenotypes of GRAPH-DICE clusters (K={K}) ===")
    core = ["name", "n", "readmit", "inpatient", "asa_ge3", "med_age", "med_optime", "med_nprov", "med_surgvol", "pct_highvol", "pct_resident"]
    with pd.option_context("display.float_format", lambda x: f"{x:.1f}", "display.max_colwidth", 60):
        print(P[core].to_string(index=False))
    print("\n  14-bucket unit composition (% of cluster encounters touching):")
    with pd.option_context("display.float_format", lambda x: f"{x:.0f}"):
        print(P[["readmit"] + [f"inst::{c}" for c in INST_CATS[:6]]].to_string(index=False))
    print("\n  team composition (median # per encounter): " + ", ".join(PROV_ROLES))
    print(P[["readmit"] + [f"med_{r.split()[0].lower()}" for r in PROV_ROLES]].to_string(index=False))
    print("\n  most over-represented named units (lift vs cohort) + top CPTs, low->high risk:")
    for _, r in P.iterrows():
        print(f"   [{r['readmit']:.1f}%] {r['name']}")
        print(f"        units: {r['top_depts_lift']}")
        print(f"        traj : {r['top_traj']}    cpt: {r['top_cpt']}")
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
