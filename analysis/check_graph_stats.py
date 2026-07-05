"""Verify the care-graph structure numbers reported in the manuscript.
Aggregate counts only; no record-level output.

   PYTHONPATH=. python analysis/check_graph_stats.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from medhg_ps.deploy import assemble_training_frame, _load_cpt_map
from medhg_ps.data import load_raw

COMORB = ["Diabetes Mellitus", "Hypertension requiring medication", "Heart Failure",
          "History of Severe COPD", "Ascites", "Disseminated Cancer", "Bleeding Disorder",
          "Preop Acute Kidney Injury", "Preop Dialysis", "Ventilator Dependent",
          "Immunosuppressive Therapy", "Current Smoker within 1 year",
          "Preop RBC Transfusions (72h)"]
NEG = {"no", "0", "nan", "none", "", "0.0"}


def norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def main():
    merged, feat_cols, cpt_arr, Fseq, seq_all, y = assemble_training_frame()
    cohort = set(merged["LogID"].astype(str))
    N = len(cohort)
    raw = load_raw()
    cpt_map = _load_cpt_map()

    # ---- encounter-provider ----
    a2 = raw.enc_prov_edges.copy()
    a2["LogID"] = norm(a2["LogID"]); a2["ProvID"] = norm(a2["ProvID"])
    a2c = a2[a2["LogID"].isin(cohort) & ~a2["ProvID"].isin(["nan", ""])]
    prov_nodes = a2c["ProvID"].nunique()
    ep_rows = len(a2c)
    ep_pairs = a2c.drop_duplicates(["LogID", "ProvID"]).shape[0]

    # ---- encounter-unit ----
    a3 = raw.enc_unit_edges.copy()
    a3["LogID"] = norm(a3["LogID"]); a3["DepartmentID"] = norm(a3["DepartmentID"])
    a3c = a3[a3["LogID"].isin(cohort) & ~a3["DepartmentID"].isin(["nan", ""])]
    unit_nodes = a3c["DepartmentID"].nunique()
    eu_rows = len(a3c)
    eu_pairs = a3c.drop_duplicates(["LogID", "DepartmentID"]).shape[0]

    # ---- diagnosis ----
    dx_cols = [c for c in COMORB if c in merged.columns]
    dx_deg = {}
    for dx in dx_cols:
        present = ~merged[dx].astype(str).str.strip().str.lower().isin(NEG)
        dx_deg[dx] = int(present.sum())
    edx = sum(dx_deg.values())

    # ---- cpt / procedure ----
    cpt_vals = [cpt_map.get(l, "UNK") for l in merged["LogID"].astype(str)]
    cpt_clean = [v for v in cpt_vals if v not in ("UNK", "", "nan")]
    cpt_nodes = len(set(cpt_clean))
    ecpt = len(cpt_clean)
    cpt_series = pd.Series(cpt_clean)
    cpt_maxdeg = int(cpt_series.value_counts().iloc[0]) if len(cpt_series) else 0

    total_nodes = N + prov_nodes + unit_nodes + len(dx_cols) + cpt_nodes

    def cmp(label, got, manuscript):
        flag = "OK" if got == manuscript else "DIFF"
        print(f"  {label:34s} computed={got:>10,}   manuscript={manuscript:>10,}   [{flag}]")

    print(f"\n=== NODES (cohort N={N:,}) ===")
    cmp("encounter nodes", N, 44721)
    cmp("provider nodes", prov_nodes, 3385)
    cmp("care-unit nodes", unit_nodes, 213)
    cmp("diagnosis nodes", len(dx_cols), 13)
    cmp("procedure (CPT) nodes", cpt_nodes, 46)
    cmp("total nodes", total_nodes, 48378)

    print(f"\n=== EDGES ===")
    print(f"  enc-provider:  rows={ep_rows:,}   unique pairs={ep_pairs:,}   manuscript=207,592")
    print(f"  enc-unit:      rows={eu_rows:,}   unique pairs={eu_pairs:,}   manuscript=129,466")
    cmp("enc-diagnosis (sum present)", edx, 102835)
    cmp("enc-procedure", ecpt, 44721)
    approx_total_pairs = ep_pairs + eu_pairs + edx + ecpt
    approx_total_rows = ep_rows + eu_rows + edx + ecpt
    print(f"  total (pairs)  ~ {approx_total_pairs:,}   (rows) ~ {approx_total_rows:,}   manuscript~485,000")

    print(f"\n=== DEGREES ===")
    prov_deg = a2c.drop_duplicates(["LogID", "ProvID"]).groupby("ProvID")["LogID"].nunique()
    unit_deg = a3c.drop_duplicates(["LogID", "DepartmentID"]).groupby("DepartmentID")["LogID"].nunique()
    print(f"  provider degree: median={int(prov_deg.median())} (manu 20)  max={int(prov_deg.max()):,} (manu 2,030)")
    print(f"  care-unit degree: max={int(unit_deg.max()):,} (manu 11,363)")
    print(f"  procedure degree: max={cpt_maxdeg:,} (manu 10,500)")
    print(f"  diagnosis degree: max={max(dx_deg.values()):,} (manu 41,499)  "
          f"[{max(dx_deg, key=dx_deg.get)}]")

    # per-encounter prov+unit node count (median)
    epn = a2c.drop_duplicates(["LogID", "ProvID"]).groupby("LogID").size()
    eun = a3c.drop_duplicates(["LogID", "DepartmentID"]).groupby("LogID").size()
    combined = epn.add(eun, fill_value=0)
    print(f"  per-encounter prov+unit nodes: median={int(combined.median())} (manu 6)")

    print(f"\n=== TRAJECTORY ===")
    linkable = a3c["LogID"].nunique()
    print(f"  linkable encounters: {linkable:,} ({100*linkable/N:.1f}%)   manuscript 40,540 (90.7%)")


if __name__ == "__main__":
    main()
