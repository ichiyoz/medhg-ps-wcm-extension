"""Combined CARE-DELIVERY + CLINICAL phenotypes of outcome-driven graph-DICE clusters.

Reuses analysis/phenotype_graph_clusters.py (GNN embeddings, DICE K=4 fit, cluster
assignment, and the care-delivery descriptors: 14-bucket unit composition, named-unit
lifts, team composition, surgeon volume, trajectories, CPTs). Adds a full CLINICAL
profile per cluster (acuity, labs, comorbidities) and names each phenotype by BOTH axes.

    MEDHG_PS_DEVICE=cpu PYTHONPATH=. python analysis/care_delivery_clinical_phenotypes.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import analysis.phenotype_graph_clusters as ph          # triggers GNN assembly + df build

df, merged, N = ph.df, ph.merged, ph.N
INST_CATS, PROV_ROLES = ph.INST_CATS, ph.PROV_ROLES
OUT_CSV = "/Users/yiyezhang/Library/CloudStorage/Dropbox/Surgery/medhg-ps/artifacts/newdata/care_delivery_clinical_phenotypes.csv"

# ---- clinical columns (df rows align with merged rows) ----------------------
LABS = ["ALB", "HCT", "Creat", "WBC", "PLT", "INR", "NA", "BUN"]
for lab in LABS:
    df[f"lab_{lab}"] = pd.to_numeric(merged[lab], errors="coerce").values

def _yes(col):
    return merged[col].astype(str).str.strip().str.lower().eq("yes").values

COMORB = {
    "Diabetes": merged["Diabetes Mellitus"].astype(str).str.strip().isin(["Insulin", "Non-insulin"]).values,
    "HTN": _yes("Hypertension requiring medication"),
    "HeartFailure": _yes("Heart Failure"),
    "COPD": _yes("History of Severe COPD"),
    "DisseminatedCancer": _yes("Disseminated Cancer"),
    "PreopAKI": _yes("Preop Acute Kidney Injury"),
    "PreopDialysis": _yes("Preop Dialysis"),
    "BleedingDisorder": _yes("Bleeding Disorder"),
    "Immunosuppressive": _yes("Immunosuppressive Therapy"),
    "Smoker": _yes("Current Smoker within 1 year"),
}
for name, arr in COMORB.items():
    df[f"cm_{name}"] = np.asarray(arr).astype(int)
df["weight"] = pd.to_numeric(merged["Weight (kg)"], errors="coerce").values


def name_combined(p, depts):
    """Name phenotype by acuity (ASA + albumin/HCT) x setting x care structure."""
    low_alb = p["lab_ALB"] < 3.6
    acu = ("high-acuity" if (p["asa_ge3"] >= 45 or low_alb) else
           "low-acuity" if (p["asa_ge3"] <= 20 and p["lab_ALB"] >= 4.0) else "mixed-acuity")
    setng = "inpatient" if p["inpatient"] >= 55 else "outpatient" if p["inpatient"] <= 35 else "mixed"
    unit = ("ICU/critical-care" if p["inst::ICU"] >= 20 else
            "ambulatory/day-surgery" if p["pct_ORonly"] >= 40 else
            "ED-origin" if p["has_ED"] >= 40 else "inpatient-ward")
    labs = f"albumin {p['lab_ALB']:.1f}, HCT {p['lab_HCT']:.0f}"
    lead = depts.split(" (")[0] if depts else ""
    return f"{acu} {setng} {unit} — {labs}, ASA≥3 {p['asa_ge3']:.0f}%, age {p['med_age']:.0f}" + (f"; lead unit {lead}" if lead else "")


def main():
    hard, K = ph.fit_clusters()
    df["cluster"] = hard
    rows = []
    for k in sorted(range(K), key=lambda k: df.loc[df.cluster == k, "y"].mean()):
        g = df[df.cluster == k]; n_k = len(g)
        depts = ph.top_depts(set(g["LogID"]), n_k)
        p = {"cluster": k, "n": n_k, "pct": 100 * n_k / N, "readmit": 100 * g["y"].mean(),
             # --- care-delivery / acuity ---
             "inpatient": 100 * g["inpatient"].mean(), "asa_ge3": 100 * (g["asa"] >= 3).mean(),
             "has_ED": 100 * g["has_ED"].mean(), "pct_ORonly": 100 * g["OR_only"].mean(),
             "med_age": g["age"].median(), "med_weight": g["weight"].median(),
             "med_optime": g["optime"].median(), "med_nprov": g["n_providers"].median(),
             "med_surgvol": g["surg_vol"].median(), "pct_highvol": 100 * g["highvol_surg"].mean(),
             "pct_resident": 100 * g["has_resident"].mean(), "pct_fellow": 100 * g["has_fellow"].mean()}
        for cat in INST_CATS:
            p[f"inst::{cat}"] = 100 * g[f"inst::{cat}"].mean()
        for role in PROV_ROLES:
            p[f"med_{role.split()[0].lower()}"] = g[f"n_{role.split()[0].lower()}"].median()
        for lab in LABS:                                   # --- labs ---
            p[f"lab_{lab}"] = g[f"lab_{lab}"].median()
        for name in COMORB:                                # --- comorbidities (%) ---
            p[f"cm_{name}"] = 100 * g[f"cm_{name}"].mean()
        p["top_depts_lift"] = depts
        p["top_traj"] = "; ".join(f"{t}({n})" for t, n in g["traj"].value_counts().head(2).items())
        p["top_cpt"] = "; ".join(f"{c}({n})" for c, n in g["cpt"].value_counts().head(3).items())
        p["name"] = name_combined(p, depts)
        rows.append(p)
    P = pd.DataFrame(rows); P.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 240)
    print(f"\n=== CARE-DELIVERY + CLINICAL phenotypes of GRAPH-DICE clusters (K={K}) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.1f}", "display.max_colwidth", 90):
        print("\n-- overview --")
        print(P[["name", "n", "readmit"]].to_string(index=False))
        print("\n-- CLINICAL block --  (labs = median)")
        print(P[["readmit", "asa_ge3", "med_age", "inpatient", "lab_ALB", "lab_HCT", "lab_Creat", "lab_WBC"]].to_string(index=False))
        print("\n-- comorbidities (% of cluster) --")
        print(P[["readmit"] + [f"cm_{n}" for n in COMORB]].to_string(index=False))
        print("\n-- CARE-DELIVERY block --")
        print(P[["readmit", "pct_ORonly", "has_ED", "inst::ICU", "inst::Med/Surg", "med_optime", "med_surgvol", "pct_highvol", "pct_resident"]].to_string(index=False))
    print("\n-- named units (lift) + trajectory + CPTs, low->high risk --")
    for _, r in P.iterrows():
        print(f"   [{r['readmit']:.1f}%] {r['name']}")
        print(f"        units: {r['top_depts_lift']}")
        print(f"        traj : {r['top_traj']}    cpt: {r['top_cpt']}")
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
