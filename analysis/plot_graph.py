"""Visualise the clinical-concept-node heterograph.

Panel A: the node/edge schema (5 node types and their relations).
Panel B: a real sampled sub-graph around a few encounters sharing a procedure,
showing how patients connect through shared providers, units, diagnoses, and the
procedure node.

Writes graph_schema.png.   PYTHONPATH=. python analysis/plot_graph.py
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

import medhg_ps.config as C
from medhg_ps.data import load_raw

COMORB = ["Diabetes Mellitus", "Hypertension requiring medication", "Heart Failure",
          "History of Severe COPD", "Ascites", "Disseminated Cancer", "Bleeding Disorder",
          "Preop Acute Kidney Injury", "Preop Dialysis", "Ventilator Dependent",
          "Immunosuppressive Therapy", "Current Smoker within 1 year",
          "Preop RBC Transfusions (72h)"]
COHORT_COLS = ["LogID", "EncounterCSN", "PAT_ID", "SurgeryDate", "SurgeryYear", "PrimaryCPT", "AgeYears"]
COL = {"enc": "#9aa6b2", "prov": "#e3b23c", "unit": "#5a8fc7", "dx": "#d9695f", "cpt": "#5fae73"}
LBL = {"enc": "Encounter (case)", "prov": "Provider", "unit": "Care unit",
       "dx": "Diagnosis", "cpt": "Procedure (CPT)"}
SHORT = {"Hypertension requiring medication": "HTN", "Diabetes Mellitus": "Diabetes",
         "Heart Failure": "Heart failure", "History of Severe COPD": "COPD",
         "Current Smoker within 1 year": "Smoker", "Bleeding Disorder": "Bleeding dis."}


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def load_cpt():
    for d in [str(C.DATA_DIR), "/Users/yiyezhang/Dropbox/Surgery"]:
        for p in glob.glob(os.path.join(d, "*.csv")):
            try:
                h = pd.read_csv(p, nrows=1, header=None, dtype=str)
            except Exception:
                continue
            if str(h.iloc[0, 0]).strip() == "LogID":
                df = pd.read_csv(p, low_memory=False)
                if {"LogID", "PrimaryCPT"} <= set(df.columns):
                    return dict(zip(_norm(df["LogID"]), _norm(df["PrimaryCPT"])))
            elif h.shape[1] == len(COHORT_COLS):
                df = pd.read_csv(p, header=None, names=COHORT_COLS, low_memory=False)
                return dict(zip(_norm(df["LogID"]), _norm(df["PrimaryCPT"])))
    raise FileNotFoundError("cohort CSV with PrimaryCPT not found")


print("[viz] loading...", flush=True)
cpt_map = load_cpt()
raw = load_raw()
enc = raw.encounters.merge(
    raw.enc_features[["LogID"] + [c for c in COMORB if c in raw.enc_features.columns]],
    on="LogID", how="inner")
enc["LogID"] = enc["LogID"].astype(str)
dx_cols = [c for c in COMORB if c in enc.columns]
a2 = raw.enc_prov_edges[["LogID", "ProvID"]].copy(); a2["LogID"] = _norm(a2["LogID"]); a2["ProvID"] = _norm(a2["ProvID"])
a3 = raw.enc_unit_edges[["LogID", "DepartmentID", "UnitType"]].copy()
a3["LogID"] = _norm(a3["LogID"]); a3["DepartmentID"] = _norm(a3["DepartmentID"])
prov_by = a2.groupby("LogID")["ProvID"].apply(list).to_dict()
unit_by = a3.groupby("LogID")["DepartmentID"].apply(lambda s: list(dict.fromkeys(s))).to_dict()
unit_name = dict(zip(a3["DepartmentID"], a3["UnitType"].astype(str)))

# --- pick a procedure shared by a few encounters (with cross-links) ---
enc["cpt"] = enc["LogID"].map(cpt_map)
counts = enc["cpt"].value_counts()
target_cpt = counts[(counts > 50) & (counts < 4000)].index[0]
pool = enc[enc["cpt"] == target_cpt]["LogID"].tolist()
sel, seen_prov = [], set()
for lid in pool:
    if lid not in prov_by:
        continue
    sel.append(lid)
    if len(sel) >= 4:
        break

G = nx.Graph()
G.add_node(f"CPT:{target_cpt}", t="cpt")
for i, lid in enumerate(sel):
    en = f"ENC{i+1}"
    G.add_node(en, t="enc"); G.add_edge(en, f"CPT:{target_cpt}")
    for pv in prov_by.get(lid, [])[:2]:
        G.add_node(f"P:{pv}", t="prov"); G.add_edge(en, f"P:{pv}")
    for du in unit_by.get(lid, [])[:2]:
        nm = unit_name.get(du, du)[:10]
        G.add_node(f"U:{nm}", t="unit"); G.add_edge(en, f"U:{nm}")
    row = enc[enc["LogID"] == lid].iloc[0]
    for dx in dx_cols:
        v = str(row[dx]).strip().lower()
        if v not in ("no", "0", "nan", "none", ""):
            nm = SHORT.get(dx, dx.split()[0])
            G.add_node(f"DX:{nm}", t="dx"); G.add_edge(en, f"DX:{nm}")
print(f"[viz] sample: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
      f"(CPT {target_cpt}, {len(sel)} encounters)", flush=True)

# ---- figure ----------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 7.2))

# Panel A: schema
S = nx.Graph()
for t in ["prov", "unit", "dx", "cpt"]:
    S.add_node(t); S.add_edge("enc", t)
S.add_node("enc")
posS = {"enc": (0, 0), "prov": (-1, 0.7), "unit": (1, 0.7), "dx": (-1, -0.7), "cpt": (1, -0.7)}
elabels = {("enc", "prov"): "treated by", ("enc", "unit"): "transferred through",
           ("enc", "dx"): "has", ("enc", "cpt"): "underwent"}
nx.draw_networkx_edges(S, posS, ax=axA, edge_color="#999", width=1.6)
nx.draw_networkx_nodes(S, posS, ax=axA, node_size=2600,
                       node_color=[COL[n] for n in S.nodes()], edgecolors="#333", linewidths=1.2)
nx.draw_networkx_labels(S, posS, ax=axA, labels={n: LBL[n].replace(" ", "\n") for n in S.nodes()}, font_size=9)
nx.draw_networkx_edge_labels(S, posS, ax=axA, edge_labels=elabels, font_size=8, font_color="#555")
axA.set_title("A  Heterograph schema (5 node types)", fontsize=13, loc="left", fontweight="bold")
axA.axis("off")

# Panel B: sample
posB = nx.spring_layout(G, seed=3, k=0.9)
nx.draw_networkx_edges(G, posB, ax=axB, edge_color="#bbb", width=1.0)
for t in COL:
    ns = [n for n in G.nodes() if G.nodes[n]["t"] == t]
    nx.draw_networkx_nodes(G, posB, ax=axB, nodelist=ns, node_color=COL[t],
                           node_size=[620 if t == "enc" else 520 for _ in ns],
                           edgecolors="#333", linewidths=0.8)
lab = {n: (n.split(":", 1)[1] if ":" in n else n) for n in G.nodes()}
nx.draw_networkx_labels(G, posB, ax=axB, labels=lab, font_size=7)
axB.set_title("B  Sampled sub-graph: patients linked through a shared procedure",
              fontsize=13, loc="left", fontweight="bold")
axB.axis("off")
axB.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[t],
                           markersize=11, markeredgecolor="#333", label=LBL[t]) for t in COL],
           loc="lower left", fontsize=9, frameon=True)

plt.tight_layout()
out = "graph_schema.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"[viz] wrote {out}", flush=True)
