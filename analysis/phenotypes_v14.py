"""Hospital-operations phenotypes (v14).

Unify care-unit trajectory + order stream + provider team as ONE graph
per encounter, then apply outcome-driven clustering (DICE) to derive
phenotypes.

Graph structure:
  Node types (4):
    encounter (14,009 -> 13,858 after exclusions)
    care_unit  (from A3 A3_enc_unit_edges — ICU / SDU / Med-Surg / etc.)
    order_group (from Order_sequence — 81 categories)
    provider   (from A2/A4 — grouped by role: attending / resident / CRNA /
                anesth / other; individual provider IDs kept as node identity)
  Edges (all bipartite, encounter as hub):
    encounter <-> care_unit    (from A3, weighted by hours)
    encounter <-> order_group  (from Order_sequence, weighted by count)
    encounter <-> provider     (from A2, unweighted)

Embedding: Node2Vec walks over the unified graph -> 64-d encounter vector.

Clustering: DICE (Deep Significance Clustering) with:
  K = 3 (low, moderate, high risk phenotypes)
  SMD constraint 0.30 between all cluster pairs
  Significance constraint (chi-square df=1 alpha=0.05)
  Anti-collapse KL balance term

Phenotype descriptions computed per cluster:
  - Readmission rate + size
  - Modal care-unit trajectory pattern
  - Top-5 order categories by frequency
  - Provider role mix
  - Length of stay
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import log, SEED, DATA_DIR, GOLD
from analysis.final_push_080_v7 import _random_walks
from analysis.final_push_080_v8 import _norm_id, prov_role, PROV_ROLES

from medhg_ps.data import load_raw, load_order_sequence, collapse_order_runs
from medhg_ps.deploy import assemble_training_frame
import medhg_ps.config as C
from analysis import dice as dice_mod

EXCLUDE_LABELS = {"Expired","Expired in Medical Facility","Hospice/Home",
                  "Hospice/Medical Facility","Acute / Short Term Hospital",
                  "Left Against Medical Advice"}

OUT_LOG = Path("artifacts/newdata/phenotypes_v14.log")
OUT_EMB = Path("artifacts/newdata/phenotypes_v14_embedding.npz")
OUT_CLUS = Path("artifacts/newdata/phenotypes_v14_clusters.csv")
OUT_PHENO = Path("artifacts/newdata/phenotypes_v14_descriptions.csv")


def build_unified_graph(merged, orders_df, a3, A2, A4):
    """Return (indptr, indices, N_enc, N_total, node_id_map)."""
    N_enc = len(merged)
    all_logids = _norm_id(merged["LogID"]).values
    lid2idx = {lid: i for i, lid in enumerate(all_logids)}

    # Provider IDs (with role attribute)
    a2 = A2.copy(); a2["LogID"] = _norm_id(a2["LogID"])
    a2["ProvID"] = _norm_id(a2["ProvID"])
    a2 = a2[a2["LogID"].isin(lid2idx)]
    a4 = A4.copy(); a4["ProvID"] = _norm_id(a4["ProvID"])
    a4["role"] = a4["ProvType"].apply(prov_role)
    a2 = a2.merge(a4[["ProvID","role"]], on="ProvID", how="left")
    a2["role"] = a2["role"].fillna("other")
    prov_tokens = sorted(a2["ProvID"].unique())
    prov2id = {p: i for i, p in enumerate(prov_tokens)}
    log(f"  providers: {len(prov_tokens):,}")

    # Care-unit IDs (from A3)
    a3 = a3.copy(); a3["LogID"] = _norm_id(a3["LogID"])
    a3 = a3[a3["LogID"].isin(lid2idx)]
    a3["DepartmentID"] = a3["DepartmentID"].astype(str)
    unit_tokens = sorted(a3["DepartmentID"].unique())
    unit2id = {u: i for i, u in enumerate(unit_tokens)}
    log(f"  care units: {len(unit_tokens):,}")

    # Order-group IDs
    orders = orders_df.copy(); orders["LogID"] = _norm_id(orders["LogID"])
    orders = orders[orders["LogID"].isin(lid2idx)]
    orders["OrderGroup"] = orders["OrderGroup"].astype(str).fillna("UNK")
    og_tokens = sorted(orders["OrderGroup"].unique())
    og2id = {g: i for i, g in enumerate(og_tokens)}
    log(f"  order groups: {len(og_tokens):,}")

    # Node numbering: [encounter | provider | care_unit | order_group]
    N_prov = len(prov_tokens); N_unit = len(unit_tokens); N_og = len(og_tokens)
    N_total = N_enc + N_prov + N_unit + N_og
    P_off = N_enc
    U_off = N_enc + N_prov
    O_off = N_enc + N_prov + N_unit

    # Build edge list (undirected — both directions added)
    rows, cols = [], []
    # encounter <-> provider
    a2["enc_i"] = a2["LogID"].map(lid2idx).astype(int)
    a2["prov_i"] = a2["ProvID"].map(prov2id).astype(int)
    for e, p in zip(a2["enc_i"].values, a2["prov_i"].values):
        rows += [e, P_off + p]; cols += [P_off + p, e]
    log(f"  enc-prov edges: {len(a2):,}")
    # encounter <-> care_unit (weighted by hours)
    a3["enc_i"] = a3["LogID"].map(lid2idx).astype(int)
    a3["unit_i"] = a3["DepartmentID"].map(unit2id).astype(int)
    for e, u in zip(a3["enc_i"].values, a3["unit_i"].values):
        rows += [e, U_off + u]; cols += [U_off + u, e]
    log(f"  enc-unit edges: {len(a3):,}")
    # encounter <-> order_group
    orders["enc_i"] = orders["LogID"].map(lid2idx).astype(int)
    orders["og_i"] = orders["OrderGroup"].map(og2id).astype(int)
    og_edges = orders[["enc_i","og_i"]].drop_duplicates()
    for e, o in zip(og_edges["enc_i"].values, og_edges["og_i"].values):
        rows += [e, O_off + o]; cols += [O_off + o, e]
    log(f"  enc-og edges (unique pairs): {len(og_edges):,}")

    # ---- order_group <-> care_unit (via timestamps) ----
    # For each order, find the unit stay whose [InTime, OutTime] contains OrderTime.
    # Aggregate to unique (order_group, care_unit) pairs.
    ts_col = "RunStart" if "RunStart" in orders.columns else "OrderTime"
    orders_ts = orders.dropna(subset=[ts_col]).copy()
    orders_ts[ts_col] = pd.to_datetime(orders_ts[ts_col], errors="coerce")
    orders_ts = orders_ts.dropna(subset=[ts_col])
    a3_ts = a3.copy()
    a3_ts["InTime"] = pd.to_datetime(a3_ts["InTime"], errors="coerce")
    a3_ts["OutTime"] = pd.to_datetime(a3_ts["OutTime"], errors="coerce")
    a3_ts = a3_ts.dropna(subset=["InTime","OutTime"])
    # Do the interval-containment join per LogID (fast pandas merge_asof + filter)
    a3_ts["unit_i"] = a3_ts["DepartmentID"].astype(str).map(unit2id).astype("Int64")
    a3_ts = a3_ts.dropna(subset=["unit_i"])
    # merge_asof requires BOTH left and right sorted by the on-key globally
    left = (orders_ts[["LogID", ts_col, "og_i"]]
            .rename(columns={ts_col:"t"})
            .sort_values("t")
            .reset_index(drop=True))
    right = (a3_ts[["LogID","InTime","OutTime","unit_i"]]
             .rename(columns={"InTime":"in_"})
             .sort_values("in_")
             .reset_index(drop=True))
    joined = pd.merge_asof(
        left, right, by="LogID", left_on="t", right_on="in_",
        direction="backward",
    )
    joined = joined.dropna(subset=["unit_i","OutTime"])
    joined = joined[joined["t"] <= joined["OutTime"]]
    joined["unit_i"] = joined["unit_i"].astype(int)
    # Aggregate to unique (og, unit) pairs (any co-occurrence)
    og_unit_pairs = joined[["og_i","unit_i"]].drop_duplicates()
    for og_i, u in zip(og_unit_pairs["og_i"].values, og_unit_pairs["unit_i"].values):
        rows += [O_off + og_i, U_off + u]
        cols += [U_off + u, O_off + og_i]
    log(f"  og-unit edges (unique pairs, via timestamps): {len(og_unit_pairs):,} "
        f"from {len(joined):,} matched order-events")

    # CSR adjacency
    rows = np.asarray(rows); cols = np.asarray(cols)
    order = np.argsort(rows)
    rows_s = rows[order]; cols_s = cols[order]
    indptr = np.zeros(N_total + 1, dtype=np.int64)
    np.add.at(indptr, rows_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = cols_s.astype(np.int64)
    log(f"  unified graph: {N_total:,} nodes / {len(rows)//2:,} undirected edges "
        f"(avg deg {len(rows)/N_total:.1f})")

    return dict(indptr=indptr, indices=indices, N_enc=N_enc, N_total=N_total,
                lid2idx=lid2idx,
                prov_tokens=prov_tokens, unit_tokens=unit_tokens, og_tokens=og_tokens,
                prov2id=prov2id, unit2id=unit2id, og2id=og2id,
                a2_join=a2, a3_join=a3, orders_join=orders)


def compute_n2v_embedding(gs, dim=64, walks=8, walk_len=15, window=5, seed=SEED):
    log(f"  generating walks (walks={walks}, walk_len={walk_len})")
    t0 = time.time()
    walks_list = _random_walks(gs["indptr"], gs["indices"], gs["N_enc"],
                                gs["N_total"], walks_per_node=walks,
                                walk_len=walk_len, seed=seed)
    log(f"  {len(walks_list):,} walks in {time.time()-t0:.1f}s")

    log(f"  fitting Word2Vec skip-gram (dim={dim}, window={window})")
    t0 = time.time()
    w2v = Word2Vec(sentences=walks_list, vector_size=dim, window=window,
                   min_count=1, sg=1, workers=4, seed=seed, epochs=5)
    log(f"  Word2Vec fit in {time.time()-t0:.1f}s vocab {len(w2v.wv.key_to_index):,}")
    emb = np.zeros((gs["N_enc"], dim), dtype=np.float32)
    for i in range(gs["N_enc"]):
        k = str(i)
        if k in w2v.wv:
            emb[i] = w2v.wv[k]
    log(f"  encounter embedding {emb.shape}")
    return emb


def describe_phenotypes(hard_clusters, y, gs, merged):
    """Compute descriptive statistics per phenotype."""
    K = int(hard_clusters.max()) + 1
    N_enc = gs["N_enc"]
    a3 = gs["a3_join"]
    a2 = gs["a2_join"]
    orders = gs["orders_join"]

    # Care-unit trajectory: modal 1st and 2nd unit per encounter
    # LOS from A3
    a3_local = a3.copy()
    a3_local["InTime"] = pd.to_datetime(a3_local["InTime"], errors="coerce")
    a3_local["OutTime"] = pd.to_datetime(a3_local["OutTime"], errors="coerce")
    los_df = a3_local.groupby("LogID").agg(
        first_in=("InTime","min"), last_out=("OutTime","max"))
    los_df["los_hr"] = (los_df["last_out"] - los_df["first_in"]).dt.total_seconds() / 3600
    los_df = los_df.reset_index()
    lid2idx = gs["lid2idx"]
    los_df["enc_i"] = los_df["LogID"].map(lid2idx)
    los_by_enc = np.zeros(N_enc, dtype=np.float32)
    ok = los_df.dropna(subset=["enc_i"])
    los_by_enc[ok["enc_i"].astype(int).values] = ok["los_hr"].values.astype(np.float32)

    # First unit per encounter
    a3_local = a3_local.dropna(subset=["InTime"]).sort_values(["LogID","InTime"])
    first_unit = a3_local.groupby("LogID").first()["DepartmentID"].astype(str)
    first_unit = pd.DataFrame({"LogID": first_unit.index,
                                "first_unit": first_unit.values})
    first_unit["enc_i"] = first_unit["LogID"].map(lid2idx)
    first_unit_by_enc = np.array(["UNK"] * N_enc, dtype=object)
    ok = first_unit.dropna(subset=["enc_i"])
    first_unit_by_enc[ok["enc_i"].astype(int).values] = ok["first_unit"].values

    # Number of orders per encounter
    n_orders = orders.groupby("LogID").size().reset_index(name="n")
    n_orders["enc_i"] = n_orders["LogID"].map(lid2idx)
    n_orders_by_enc = np.zeros(N_enc, dtype=np.float32)
    ok = n_orders.dropna(subset=["enc_i"])
    n_orders_by_enc[ok["enc_i"].astype(int).values] = ok["n"].values.astype(np.float32)

    # Provider role counts per encounter
    role_counts = {r: np.zeros(N_enc, dtype=np.float32) for r in PROV_ROLES}
    for role in PROV_ROLES:
        sub = a2[a2["role"] == role]
        cnt = sub.groupby("enc_i").size()
        role_counts[role][cnt.index.values] = cnt.values.astype(np.float32)

    # Age, sex from merged
    age = pd.to_numeric(merged["AgeYears"], errors="coerce").fillna(60).values
    sex_female = (merged["Gender"].astype(str) == "F").astype(float).values

    # Discharge to home
    home_disp = merged["Discharge Disposition"].astype(str).isin({
        "Home or Self Care","Home-Health Care Svc"}).values

    # Description per cluster
    rows = []
    for k in range(K):
        mask = (hard_clusters == k)
        n = int(mask.sum())
        if n == 0: continue
        readmit_rate = float(y[mask].mean())
        # Modal first unit
        first_unit_series = pd.Series(first_unit_by_enc[mask])
        top_first_unit = first_unit_series.value_counts().head(1).index[0]
        # Top-5 order categories
        og_join = gs["orders_join"]
        og_by_enc = og_join[og_join["enc_i"].isin(np.where(mask)[0])]\
                    .groupby("OrderGroup").size().sort_values(ascending=False).head(5)
        top5_orders = "; ".join(f"{k[:35]} ({v:,})" for k,v in og_by_enc.items())
        rows.append(dict(
            phenotype=f"P{k+1}",
            n=n,
            readmit_rate=readmit_rate,
            median_los_hr=float(np.median(los_by_enc[mask])),
            median_age=float(np.median(age[mask])),
            fraction_female=float(sex_female[mask].mean()),
            fraction_home_disposition=float(home_disp[mask].mean()),
            median_n_orders=float(np.median(n_orders_by_enc[mask])),
            median_n_attending=float(np.median(role_counts["attending"][mask])),
            median_n_resident=float(np.median(role_counts["resident"][mask])),
            median_n_crna=float(np.median(role_counts["crna"][mask])),
            median_n_anesth=float(np.median(role_counts["anesth"][mask])),
            most_common_first_unit=str(top_first_unit),
            top5_order_groups=top5_orders,
        ))
    return pd.DataFrame(rows)


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== PHENOTYPES v14 (unified hospital-operations graph + DICE) START ===")

    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID","ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str); merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    excl = merged["Discharge Disposition"].astype(str).isin(EXCLUDE_LABELS)
    merged = merged.loc[~excl].reset_index(drop=True)
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N}  base rate {y.mean()*100:.2f}%")

    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()
    a3 = pd.read_parquet(f"{DATA_DIR}/A3_enc_unit_edges.parquet")

    log("building unified hospital-operations graph")
    gs = build_unified_graph(merged, orders_df, a3, raw.enc_prov_edges,
                              raw.prov_attrs)

    log("computing Node2Vec encounter embeddings on unified graph")
    emb = compute_n2v_embedding(gs)
    np.savez(OUT_EMB, y=y, emb=emb)

    # Standardize before DICE
    scaler = StandardScaler(); emb_std = scaler.fit_transform(emb).astype(np.float32)

    # DICE clustering — match prior 3-substrate config that had good separation:
    #   K=3, d=16, lam_smd=30, smd_target=0.30, spread=0.0
    log("running DICE clustering (K=3, d=16, lam_smd=30, smd_target=0.30)")
    K = 3
    d_latent = 16
    from analysis import dice as dice_mod
    from analysis.dice import fit as dice_fit, cluster_proba
    dice_mod.LAM["smd"] = 30.0   # raise SMD-constraint weight (prior config)

    t0 = time.time()
    model = dice_fit(
        X=emb_std, y=y.astype(np.float32), K=K, d=d_latent,
        smd_target=0.30, spread=0.0, verbose=False,
        seed=SEED,
    )
    log(f"  DICE fit in {time.time()-t0:.0f}s")

    # Extract hard clusters
    P = cluster_proba(model, emb_std)
    hard = P.argmax(axis=1)
    log(f"  cluster sizes: {np.bincount(hard, minlength=K)}")

    # Readmission rates per cluster
    for k in range(K):
        m = hard == k
        log(f"  cluster P{k+1}: n={int(m.sum()):,}  "
            f"readmit rate={y[m].mean()*100:.2f}%")

    # Descriptive phenotype table
    log("computing phenotype descriptions")
    df = describe_phenotypes(hard, y, gs, merged)
    df = df.sort_values("readmit_rate").reset_index(drop=True)
    df["phenotype"] = [f"P{i+1} ({name})" for i, name in
                        zip(range(len(df)),
                            ["low","moderate","high"][:len(df)])]
    df.to_csv(OUT_PHENO, index=False)
    log(f"\nsaved {OUT_PHENO}")
    print(df.to_string(index=False))

    # Save cluster assignments
    lid_df = pd.DataFrame({
        "LogID": merged["LogID"].astype(str).values,
        "y": y,
        "cluster": hard,
    })
    lid_df.to_csv(OUT_CLUS, index=False)
    log(f"saved {OUT_CLUS}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
