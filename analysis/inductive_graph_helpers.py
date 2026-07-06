"""Inductive graph-model helpers for fold-honest evaluation.

Both MedHG-PS ie-HGCN and Node2Vec are transductive by default (their
embeddings are computed on a graph that includes the test-fold encounters).
For deployment-honest CV we need INDUCTIVE embeddings: test encounters
must NOT participate in training message passing / walks, and their
embeddings must be produced by a well-defined aggregation over
training-only structure.

Two helpers here:

train_gnn_and_encode_inductive(...):
    Trains ie-HGCN on the training subgraph (built from train-only
    encounter nodes + all providers and care-units they touched).
    After training, test-encounter embeddings are computed as the
    aggregation of their neighbours' trained embeddings — the same
    inductive step used by GraphSAGE and consistent with what a
    deployment pipeline would do for a genuinely new patient. Returns
    a full-length (N, D) embedding matrix aligned to merged row order.

compute_node2vec_inductive(...):
    Runs Node2Vec walks over the training subgraph only. Test
    encounters do not appear in walks. Their embeddings are computed
    inductively as the mean of the walk-based embeddings of the
    provider/unit/order-group nodes they connect to. This mirrors
    what would happen at deployment: a new patient does not
    contribute to walks; their embedding is aggregated from the
    trained neighbourhood embeddings.

Both helpers return an (N, D) matrix aligned to the merged frame order,
with rows corresponding to test-fold encounters filled with inductively
computed embeddings — never with embeddings that saw the test encounter
during training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from copy import copy

from medhg_ps.data import fit_preprocess, apply_preprocess
from medhg_ps.graph import build_graph
from medhg_ps.train import train_model, set_seed
from medhg_ps.extract_embeddings import extract_embeddings
from analysis.final_push_080 import log
from analysis.final_push_080_v8 import _norm_id


def _restrict_edges_to_train(edges_df, id_col, train_logids):
    """Keep only edges whose encounter endpoint is in the training set.
    id_col is the encounter identifier column name (e.g. LogID)."""
    sub = edges_df.copy()
    sub[id_col] = sub[id_col].astype(str)
    return sub[sub[id_col].isin(train_logids)]


def train_gnn_and_encode_inductive(
    merged, feat_cols, cpt_arr, y, tr_idx,
    raw_bundle, prov_ids, X_prov, unit_ids, X_unit,
    fold_seed, gnn_cfg, val_frac=0.10,
):
    """Inductive MedHG-PS ie-HGCN training and encoding.

    Steps
    -----
    1. Build a training subgraph containing ONLY the fold's training encounter
       nodes (test encounters and their edges are excluded). Providers and
       care-units are retained regardless (their identity is stable and their
       features do not depend on the test set).
    2. Train ie-HGCN on this training subgraph with an inner val split.
    3. Extract provider and care-unit trained embeddings.
    4. For every encounter in merged (train AND test), compute its inductive
       embedding as the mean of its neighbours' trained embeddings — training
       encounters use the same aggregation for consistency. This is the
       GraphSAGE inductive step and mirrors deployment behaviour on a new
       patient.
    5. Also compute an aggregated provider embedding per encounter (mirrors
       the transductive helper's output shape).

    Returns
    -------
    ndarray of shape (N, D_enc + D_prov), the full-cohort inductive embedding
    matrix aligned to the merged row order.
    """
    N = len(merged)
    rng = np.random.default_rng(fold_seed)
    tr_all = np.array(tr_idx).copy()
    rng.shuffle(tr_all)
    n_val = int(round(val_frac * len(tr_all)))
    val = tr_all[:n_val]
    tr = tr_all[n_val:]
    te = np.array([i for i in range(N) if i not in set(tr_idx)],
                  dtype=np.int64)

    train_logids = set(_norm_id(merged["LogID"].iloc[tr_all]).astype(str))
    all_logids = merged["LogID"].astype(str).values

    # ---- Restrict encounter-provider and encounter-unit edges to train
    #      encounters. Provider/unit nodes stay full.
    raw_train = copy(raw_bundle)
    raw_train.enc_prov_edges = _restrict_edges_to_train(
        raw_bundle.enc_prov_edges, "LogID", train_logids)
    raw_train.enc_unit_edges = _restrict_edges_to_train(
        raw_bundle.enc_unit_edges, "LogID", train_logids)

    # Encounter feature preprocessing fit on train-only rows
    feat_all = merged[feat_cols].copy()
    _, st = fit_preprocess(feat_all.iloc[tr].reset_index(drop=True),
                            id_cols=[])
    X_enc = apply_preprocess(feat_all, st)

    # ---- Build training subgraph: encounter nodes for TRAIN rows only.
    train_only_merged = merged.iloc[tr_all].reset_index(drop=True)
    tm_local = np.zeros(len(tr_all), bool)
    tm_local[:len(tr)] = True
    vm_local = np.zeros(len(tr_all), bool)
    vm_local[len(tr):] = True
    em_local = np.zeros(len(tr_all), bool)      # no test encounters in training graph

    X_enc_train = X_enc[tr_all]

    art_train = build_graph(
        raw=raw_train, encounters_merged=train_only_merged,
        enc_features=X_enc_train,
        prov_ids=prov_ids, prov_features=X_prov,
        unit_ids=unit_ids, unit_features=X_unit,
        train_mask=tm_local, val_mask=vm_local, test_mask=em_local,
    )

    # ---- Train the ie-HGCN on the training subgraph.
    set_seed(fold_seed)
    model, _ = train_model(art_train, cfg=gnn_cfg, save_dir=None, verbose=False)

    # ---- Extract provider + unit trained embeddings from the trained model.
    tabs_train = extract_embeddings(
        model, art_train,
        raw_prov_attrs=raw_bundle.prov_attrs,
        raw_unit_attrs=raw_bundle.unit_attrs,
        device=gnn_cfg.device,
    )
    prov_emb_df = tabs_train.provider.copy()
    prov_emb_df["ProvID"] = _norm_id(prov_emb_df["ProvID"]).astype(str)
    unit_emb_df = tabs_train.unit.copy()
    unit_emb_df["DepartmentID"] = unit_emb_df["DepartmentID"].astype(str)

    prov_cols = [c for c in prov_emb_df.columns if c.startswith("emb_")]
    unit_cols = [c for c in unit_emb_df.columns if c.startswith("emb_")]
    D_p = len(prov_cols); D_u = len(unit_cols)
    prov_map = dict(zip(prov_emb_df["ProvID"],
                        prov_emb_df[prov_cols].values.astype(np.float32)))
    unit_map = dict(zip(unit_emb_df["DepartmentID"],
                        unit_emb_df[unit_cols].values.astype(np.float32)))

    # ---- For every encounter (train + test), compute inductive encounter
    #      embedding as the mean of its provider and unit neighbours' trained
    #      embeddings. Aggregated provider embedding is also returned so the
    #      output matches the transductive helper's (D_enc + D_prov) shape.
    ep_all = raw_bundle.enc_prov_edges.copy()
    ep_all["LogID"] = _norm_id(ep_all["LogID"]).astype(str)
    ep_all["ProvID"] = _norm_id(ep_all["ProvID"]).astype(str)
    eu_all = raw_bundle.enc_unit_edges.copy()
    eu_all["LogID"] = _norm_id(eu_all["LogID"]).astype(str)
    eu_all["DepartmentID"] = eu_all["DepartmentID"].astype(str)

    # Compute encounter embedding = mean(provider neighbours' emb, unit neighbours' emb)
    # Aggregated provider embedding = mean(provider neighbours' emb)
    enc_emb = np.zeros((N, D_p), dtype=np.float32)   # use provider-emb dim as encounter emb dim
    prov_agg = np.zeros((N, D_p), dtype=np.float32)

    # Per-encounter provider neighbourhood aggregation
    for logid, group in ep_all.groupby("LogID"):
        i = np.where(all_logids == logid)[0]
        if len(i) == 0: continue
        i = int(i[0])
        embs = [prov_map[p] for p in group["ProvID"].values if p in prov_map]
        if embs:
            m = np.mean(np.stack(embs), axis=0)
            enc_emb[i] += m
            prov_agg[i] = m

    # Per-encounter unit neighbourhood aggregation, projected to same dim as
    # provider emb by simple truncation/pad if dims differ.
    for logid, group in eu_all.groupby("LogID"):
        i = np.where(all_logids == logid)[0]
        if len(i) == 0: continue
        i = int(i[0])
        embs = [unit_map[u] for u in group["DepartmentID"].values if u in unit_map]
        if embs:
            u_mean = np.mean(np.stack(embs), axis=0)
            # match dims: repeat / truncate to D_p if D_u != D_p
            if D_u < D_p:
                u_mean = np.concatenate([u_mean, np.zeros(D_p - D_u, dtype=np.float32)])
            elif D_u > D_p:
                u_mean = u_mean[:D_p]
            enc_emb[i] += u_mean
    enc_emb /= 2.0   # simple mean between two aggregations

    log(f"  inductive: enc_emb {enc_emb.shape}  prov_agg {prov_agg.shape}  "
        f"n_test_covered {int((enc_emb.sum(axis=1) != 0)[te].sum())}/{len(te)}")

    return np.hstack([enc_emb, prov_agg]).astype(np.float32)


def compute_node2vec_inductive(
    gs, train_ids, dim=64, walks=5, walk_len=10, window=5, seed=42,
):
    """Inductive Node2Vec: walks on the training subgraph only. Test
    encounters get their embedding by mean-aggregating the walk-based
    embeddings of the provider / unit / order-group nodes they connect to.

    Parameters
    ----------
    gs : dict from precompute_graph — contains N, M_ep, M_eo, prov_tokens etc.
    train_ids : sequence of int, encounter indices to include in walks.
    dim, walks, walk_len, window : Node2Vec / Word2Vec hyperparameters.

    Returns
    -------
    ndarray of shape (N, dim), one row per encounter. Train rows contain
    walk-based embeddings; test rows contain aggregated-neighbour embeddings.
    """
    from gensim.models import Word2Vec
    from analysis.final_push_080_v7 import _random_walks

    N_enc = gs["N"]
    train_set = set(int(i) for i in train_ids)
    N_prov = len(gs["prov_tokens"])
    N_og = len(gs["og_tokens"])

    # Node id numbering: enc[0..N_enc), prov[N_enc..N_enc+N_prov), og[...]
    P_off = N_enc
    O_off = N_enc + N_prov

    # Build CSR adjacency restricted to TRAIN encounters + all prov/og nodes.
    rows, cols = [], []
    a2 = gs["a2_prov_per_enc"]
    prov2id = gs["prov2id"]
    for enc_i, provs in a2.items():
        if int(enc_i) not in train_set:
            continue
        for p in provs:
            if p in prov2id:
                pj = P_off + prov2id[p]
                rows.append(int(enc_i)); cols.append(pj)
                rows.append(pj); cols.append(int(enc_i))
    M_eo_coo = gs["M_eo"].tocoo()
    for i, k, _ in zip(M_eo_coo.row, M_eo_coo.col, M_eo_coo.data):
        if int(i) not in train_set:
            continue
        og_j = O_off + int(k)
        rows.append(int(i)); cols.append(og_j)
        rows.append(og_j); cols.append(int(i))

    total = N_enc + N_prov + N_og
    order = np.argsort(np.asarray(rows))
    rows_s = np.asarray(rows)[order]
    cols_s = np.asarray(cols)[order]
    indptr = np.zeros(total + 1, dtype=np.int64)
    np.add.at(indptr, rows_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = cols_s.astype(np.int64)
    log(f"  inductive n2v: train-subgraph {total:,} nodes, "
        f"{len(rows)//2:,} undirected edges (train-only encounter side)")

    walks_list = _random_walks(indptr, indices, N_enc, total,
                               walks_per_node=walks, walk_len=walk_len,
                               seed=seed)
    w2v = Word2Vec(sentences=walks_list, vector_size=dim, window=window,
                   min_count=1, sg=1, workers=1, seed=seed, epochs=5)

    emb = np.zeros((N_enc, dim), dtype=np.float32)
    # Train encounters: use their walk-based embedding
    for i in train_set:
        k = str(i)
        if k in w2v.wv:
            emb[i] = w2v.wv[k]

    # Test encounters: mean of neighbour node embeddings from training walks
    test_ids = [i for i in range(N_enc) if i not in train_set]
    for i in test_ids:
        nbr_embs = []
        # provider neighbours
        for p in a2.get(i, []):
            if p in prov2id:
                key = str(P_off + prov2id[p])
                if key in w2v.wv:
                    nbr_embs.append(w2v.wv[key])
        # order-group neighbours
        m_row = gs["M_eo"].getrow(i).tocoo()
        for k in m_row.col:
            key = str(O_off + int(k))
            if key in w2v.wv:
                nbr_embs.append(w2v.wv[key])
        if nbr_embs:
            emb[i] = np.mean(np.stack(nbr_embs), axis=0)

    n_covered = int((emb.sum(axis=1) != 0).sum())
    log(f"  inductive n2v: emb {emb.shape}  n_covered {n_covered}/{N_enc}")
    return emb
