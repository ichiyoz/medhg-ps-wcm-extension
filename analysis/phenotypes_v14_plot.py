"""Per-phenotype network visualizations from the unified hospital-operations
graph (v14). Three plots, one per phenotype (P1 low, P2 moderate, P3 high),
each showing the compact subgraph structure:
  - Top-6 care-unit nodes
  - Top-8 order-group nodes
  - Provider role nodes (5 roles)
  - Edges weighted by within-phenotype co-occurrence prevalence

Plus one summary plot: UMAP of the 64-d Node2Vec embedding colored by
phenotype.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import log, DATA_DIR, GOLD
from analysis.final_push_080_v8 import _norm_id, prov_role, PROV_ROLES
from medhg_ps.data import load_raw, load_order_sequence, collapse_order_runs
from medhg_ps.deploy import assemble_training_frame

OUT_DIR = Path("artifacts/newdata/phenotypes_v14_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_LABELS = {"Expired","Expired in Medical Facility","Hospice/Home",
                  "Hospice/Medical Facility","Acute / Short Term Hospital",
                  "Left Against Medical Advice"}

def short_label(og):
    s = str(og).replace("MED:", "M:").replace("PROC:", "P:")
    return s[:25]

def unit_label(u, unit_names):
    u_str = str(u)
    return unit_names.get(u_str, f"U:{u_str[:6]}")


def main():
    # Load cluster assignments + cohort
    clus = pd.read_csv("artifacts/newdata/phenotypes_v14_clusters.csv")
    clus["LogID"] = clus["LogID"].astype(str)

    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID","ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str); merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    excl = merged["Discharge Disposition"].astype(str).isin(EXCLUDE_LABELS)
    merged = merged.loc[~excl].reset_index(drop=True)
    merged = merged.merge(clus[["LogID","cluster"]], on="LogID", how="left")

    # Sort clusters by readmit rate to label as low/moderate/high
    rates = clus.groupby("cluster")["y"].mean().sort_values()
    cluster_rank = {c: i for i, c in enumerate(rates.index)}   # low=0, mod=1, high=2
    cluster_labels = {0: "P1 (low)", 1: "P2 (moderate)", 2: "P3 (high)"}
    cluster_colors_bg = {0: "#e3f2fd", 1: "#fff3e0", 2: "#ffebee"}

    # Load ancillary tables
    raw = load_raw()
    a3 = pd.read_parquet(f"{DATA_DIR}/A3_enc_unit_edges.parquet")
    a3["LogID"] = a3["LogID"].astype(str)
    orders_df = collapse_order_runs(load_order_sequence())
    orders_df["LogID"] = orders_df["LogID"].astype(str)

    a2 = raw.enc_prov_edges.copy(); a2["LogID"] = _norm_id(a2["LogID"])
    a2["ProvID"] = _norm_id(a2["ProvID"])
    a4 = raw.prov_attrs.copy(); a4["ProvID"] = _norm_id(a4["ProvID"])
    a4["role"] = a4["ProvType"].apply(prov_role)
    a2 = a2.merge(a4[["ProvID","role"]], on="ProvID", how="left")
    a2["role"] = a2["role"].fillna("other")

    # Attach cluster to a3/orders/a2 via LogID
    log_to_cluster = dict(zip(merged["LogID"], merged["cluster"]))
    a3["cluster"] = a3["LogID"].map(log_to_cluster)
    orders_df["cluster"] = orders_df["LogID"].map(log_to_cluster)
    a2["cluster"] = a2["LogID"].map(log_to_cluster)

    # Unit name lookup (from A5)
    unit_names = {}
    try:
        a5 = raw.unit_attrs
        for _, row in a5.iterrows():
            unit_names[str(row["DepartmentID"])] = str(row.get("DepartmentName", ""))[:20]
    except Exception:
        pass

    # Build one subgraph plot per phenotype
    for internal_c, rank in cluster_rank.items():
        pheno_label = cluster_labels[rank]
        color_bg = cluster_colors_bg[rank]
        n_enc = int((merged["cluster"] == internal_c).sum())
        y_rate = float(merged.loc[merged["cluster"] == internal_c, "ReadmittedWithin30Days_gold"].mean())

        log(f"building plot for {pheno_label}  n={n_enc}  rate={y_rate*100:.2f}%")

        sub_a3 = a3[a3["cluster"] == internal_c]
        sub_ord = orders_df[orders_df["cluster"] == internal_c]
        sub_prv = a2[a2["cluster"] == internal_c]

        # Top-6 care units by prevalence (fraction of encounters that visited)
        n_denom = max(sub_a3["LogID"].nunique(), 1)
        top_units = (sub_a3.groupby("DepartmentID")["LogID"].nunique() / n_denom)\
                    .sort_values(ascending=False).head(6)
        # Top-8 order groups by prevalence (fraction of encounters)
        n_ord_denom = max(sub_ord["LogID"].nunique(), 1)
        top_ogs = (sub_ord.groupby("OrderGroup")["LogID"].nunique() / n_ord_denom)\
                  .sort_values(ascending=False).head(8)
        # Provider role counts (mean per encounter)
        role_freq = {r: 0.0 for r in PROV_ROLES}
        role_denom = max(sub_prv["LogID"].nunique(), 1)
        rc = sub_prv.groupby("role")["LogID"].nunique() / role_denom
        for r in PROV_ROLES:
            if r in rc.index: role_freq[r] = float(rc[r])

        # Build networkx graph
        G = nx.Graph()
        # Encounter hub (styled specially)
        G.add_node("ENC", kind="encounter",
                   size=3000, label=f"{pheno_label}\nn={n_enc:,}\nreadmit {y_rate*100:.1f}%")
        # Care unit nodes
        for u, prev in top_units.items():
            G.add_node(f"U:{u}", kind="unit", size=800 + 2200*prev,
                       label=unit_label(u, unit_names))
            G.add_edge("ENC", f"U:{u}", weight=prev*4)
        # Order group nodes
        for og, prev in top_ogs.items():
            G.add_node(f"O:{og}", kind="order", size=800 + 2200*prev,
                       label=short_label(og))
            G.add_edge("ENC", f"O:{og}", weight=prev*4)
        # Provider role nodes
        for r, freq in role_freq.items():
            if freq < 0.01: continue
            G.add_node(f"R:{r}", kind="role", size=800 + 2200*freq,
                       label=r.upper())
            G.add_edge("ENC", f"R:{r}", weight=freq*4)

        # Layout
        pos = nx.spring_layout(G, seed=42, k=2.5, iterations=100)
        # Put encounter at center
        pos["ENC"] = np.array([0.0, 0.0])

        # Colors by node kind
        color_map = {"encounter": "#4a148c",  # deep purple
                     "unit": "#1976d2",       # blue
                     "order": "#388e3c",       # green
                     "role": "#f57c00"}        # orange

        fig, ax = plt.subplots(figsize=(11, 9), facecolor=color_bg)
        ax.set_facecolor(color_bg)
        # Draw edges with weight
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1.0)
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="gray", alpha=0.35, linewidth=max(0.5, min(5, w*0.8)))
        # Draw nodes
        for n, d in G.nodes(data=True):
            x, y_ = pos[n]
            size = d.get("size", 800)
            c = color_map[d.get("kind", "encounter")]
            ax.scatter(x, y_, s=size, c=c, edgecolors="white", linewidths=1.2, zorder=3)
            # Label
            lbl = d.get("label", n)
            if d["kind"] == "encounter":
                ax.text(x, y_, lbl, ha="center", va="center", fontsize=10,
                        color="white", fontweight="bold", zorder=5)
            else:
                yoff = 0.09 if y_ >= 0 else -0.09
                va = "bottom" if y_ >= 0 else "top"
                ax.text(x, y_ + yoff, lbl, ha="center", va=va, fontsize=8.5,
                        color="#222", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                   ec="#999", alpha=0.9, lw=0.5))
        # Legend
        legend_handles = [
            mpatches.Patch(color=color_map["encounter"], label="Encounter (hub)"),
            mpatches.Patch(color=color_map["unit"], label="Care unit"),
            mpatches.Patch(color=color_map["order"], label="Order group"),
            mpatches.Patch(color=color_map["role"], label="Provider role"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9,
                  fontsize=9, title="Node type", title_fontsize=10)
        ax.set_title(f"{pheno_label} — hospital-operations subgraph\n"
                     f"node size ∝ within-phenotype prevalence · edge width ∝ prevalence",
                     fontsize=12)
        ax.set_axis_off()
        outpath = OUT_DIR / f"phenotype_{rank+1}_{pheno_label.split(' ')[0]}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=140, bbox_inches="tight", facecolor=color_bg)
        plt.close()
        log(f"  saved {outpath}")

    # UMAP summary plot
    log("building UMAP summary")
    z = np.load("artifacts/newdata/phenotypes_v14_embedding.npz")
    emb = z["emb"]
    y = z["y"]
    try:
        from umap import UMAP
        um = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        emb2d = um.fit_transform(emb)
    except Exception as e:
        log(f"  UMAP not available ({e}); falling back to PCA")
        from sklearn.decomposition import PCA
        emb2d = PCA(n_components=2, random_state=42).fit_transform(emb)

    cluster_arr = merged["cluster"].values
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    palette = {0: "#1976d2", 1: "#f57c00", 2: "#c62828"}   # low blue, mod orange, high red
    for internal_c, rank in cluster_rank.items():
        m = cluster_arr == internal_c
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=6, c=palette[rank], alpha=0.35,
                   label=cluster_labels[rank], edgecolors="none")
    ax.set_title("UMAP of unified-graph Node2Vec embedding\ncolored by DICE phenotype",
                 fontsize=13)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    lg = ax.legend(fontsize=11, markerscale=3, framealpha=0.9,
                    title="Phenotype", title_fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "umap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()
    log(f"  saved {OUT_DIR}/umap_summary.png")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
