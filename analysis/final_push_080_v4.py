"""Final push v4 — TEMPORAL-EDGE heterograph, no GRU/redundant blocks.

Feature blocks (leaner than v3):
  A base tabular + CPT one-hot (~200)
  B LACE utility (Charlson x 17 + ED180d + admits365d, 25)
  C LACE + HOSPITAL scores + components (16)
  H geocode + ACS (6)
  E_temp TEMPORAL-EDGE graph embedding (64) — NEW

E_temp construction:
  Load order_sequence.parquet. For each (LogID, OrderGroup) pair:
    norm_pos = mean(SeqInEncounter / max(SeqInEncounter per LogID)) in [0, 1]
             — 0 = at admission, 1 = at discharge
    edge_weight = norm_pos (later-in-encounter orders weighted more)
  Heterograph:
    ('encounter', 'ordered', 'order_group')  edata['w'] = tensor
    ('order_group', 'ordered_by', 'encounter')  edata['w']
    ('encounter', 'treated_by', 'provider')  from A2, unweighted
    ('provider', 'treats', 'encounter')
  Encounter node feats = standardized tabular (train-fold fit).
  Order_group node feats = one-hot per token id.
  Provider node feats = ProviderType one-hot from A4.
  Model: custom 2-layer weighted-mean HeteroConv (torch + dgl.function.u_mul_e / sum).
  Train per fold: BCE class-weighted, Adam, early-stop on val AUROC, VAL_FRAC=0.10.
  Extract encounter embedding (64-d) as E_temp.

Dropped (redundant with E_temp or subsumed):
  D_order (order-seq GRU) — E_temp replaces it
  D_unit (Fseq care-path tabular)
  E1_v3 (static unweighted graph) — E_temp is the weighted improvement
  E2 (unit-seq GRU)
"""
from __future__ import annotations
import os, sys, time, warnings
from copy import copy
from dataclasses import replace
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import dgl, dgl.function as fn
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H,
    learner_rf, learner_hgb, eval_pooled_oof,
    SEED, DATA_DIR, GOLD,
)

import medhg_ps.config as C, medhg_ps.data as d
from medhg_ps.data import fit_preprocess, apply_preprocess, load_raw
from medhg_ps.deploy import assemble_training_frame

OUT_LOG = Path("artifacts/newdata/final_push_080_v4.log")
OUT_RES = Path("artifacts/newdata/final_push_080_v4_results.csv")
OUT_OOF = Path("artifacts/newdata/final_push_080_v4_oof.npz")
OUT_ABL = Path("artifacts/newdata/final_push_080_v4_ablation.csv")

VAL_FRAC = 0.10
EMB_DIM = 64
GNN_EPOCHS = 40
GNN_PATIENCE = 8
GNN_LR = 1e-2
GNN_DEV = torch.device("cpu")     # DGL heterograph doesn't support MPS


# ============================================================
# Build the temporal-edge heterograph (built ONCE, edges + node
# feature shells persist across folds — only encounter feats and
# labels/train masks change per fold)
# ============================================================
def build_temporal_hg(merged, orders_df, A2, A4):
    """Return (hg, {node_type: node_feats_np}, log_id_to_enc_id).
    Weighted edges: encounter <-> order_group by norm_pos."""
    all_logids = merged["LogID"].astype(str).values
    enc2id = {lid: i for i, lid in enumerate(all_logids)}
    N_enc = len(all_logids)

    # ---- ORDER-GROUP nodes + temporal-weighted edges ----
    orders = orders_df.copy()
    orders["LogID"] = orders["LogID"].astype(str)
    orders["OrderGroup"] = orders["OrderGroup"].astype(str).fillna("UNK")
    # compute norm_pos (fraction of way through the encounter)
    orders["SeqInEncounter"] = pd.to_numeric(orders["SeqInEncounter"], errors="coerce")
    orders["max_seq"] = orders.groupby("LogID")["SeqInEncounter"].transform("max")
    orders["norm_pos"] = orders["SeqInEncounter"] / orders["max_seq"].replace(0, np.nan)
    orders = orders.dropna(subset=["norm_pos"])
    # per (LogID, OrderGroup): mean norm_pos + count
    grouped = (orders.groupby(["LogID", "OrderGroup"])
               .agg(count=("norm_pos", "size"), mean_pos=("norm_pos", "mean"))
               .reset_index())
    # cap to top 30 order-groups per encounter (by count)
    grouped["rk"] = grouped.groupby("LogID")["count"].rank(ascending=False, method="first")
    grouped = grouped[grouped["rk"] <= 30]

    # only keep encounters that exist in cohort
    grouped = grouped[grouped["LogID"].isin(enc2id)]
    grouped["enc_id"] = grouped["LogID"].map(enc2id).astype(np.int64)

    og_tokens = sorted(grouped["OrderGroup"].unique())
    og2id = {t: i for i, t in enumerate(og_tokens)}
    grouped["og_id"] = grouped["OrderGroup"].map(og2id).astype(np.int64)
    N_og = len(og_tokens)

    # edge weight = mean_pos (0 = at admission, 1 = at discharge)
    e_weight = grouped["mean_pos"].values.astype(np.float32)
    log(f"  edges enc->og: {len(grouped):,}  order-groups: {N_og}  "
        f"weight median {np.median(e_weight):.2f}  p25 {np.percentile(e_weight,25):.2f} "
        f"p75 {np.percentile(e_weight,75):.2f}")

    enc_src_og = torch.tensor(grouped["enc_id"].values, dtype=torch.int64)
    og_dst = torch.tensor(grouped["og_id"].values, dtype=torch.int64)
    ew = torch.tensor(e_weight, dtype=torch.float32)

    # ---- PROVIDER nodes + edges (unweighted) ----
    a2 = A2.copy()
    a2["LogID"] = a2["LogID"].astype(str)
    a2["ProvID"] = a2["ProvID"].astype(str).str.replace(r"\.0+$", "", regex=True)
    # keep only providers with attributes
    a4 = A4.copy()
    a4["ProvID"] = a4["ProvID"].astype(str).str.replace(r"\.0+$", "", regex=True)
    a2 = a2[a2["LogID"].isin(enc2id) & a2["ProvID"].isin(a4["ProvID"])]
    prov_tokens = sorted(a2["ProvID"].unique())
    prov2id = {p: i for i, p in enumerate(prov_tokens)}
    a2["enc_id"] = a2["LogID"].map(enc2id).astype(np.int64)
    a2["prov_id"] = a2["ProvID"].map(prov2id).astype(np.int64)
    N_prov = len(prov_tokens)
    log(f"  edges enc->prov: {len(a2):,}  providers: {N_prov}")

    enc_src_prov = torch.tensor(a2["enc_id"].values, dtype=torch.int64)
    prov_dst = torch.tensor(a2["prov_id"].values, dtype=torch.int64)

    # ---- Build heterograph ----
    data_dict = {
        ("encounter", "ordered", "order_group"): (enc_src_og, og_dst),
        ("order_group", "ordered_by", "encounter"): (og_dst, enc_src_og),
        ("encounter", "treated_by", "provider"): (enc_src_prov, prov_dst),
        ("provider", "treats", "encounter"): (prov_dst, enc_src_prov),
    }
    num_nodes = {"encounter": N_enc, "order_group": N_og, "provider": N_prov}
    hg = dgl.heterograph(data_dict, num_nodes_dict=num_nodes)

    # Edge weights (order edges only)
    hg.edges["ordered"].data["w"] = ew
    hg.edges["ordered_by"].data["w"] = ew
    # provider edges: uniform weight 1.0
    hg.edges["treated_by"].data["w"] = torch.ones(len(a2), dtype=torch.float32)
    hg.edges["treats"].data["w"] = torch.ones(len(a2), dtype=torch.float32)

    # ---- Node features ----
    og_feats = np.eye(N_og, dtype=np.float32)   # one-hot per order-group
    # provider ProvType one-hot
    a4_sub = a4[a4["ProvID"].isin(prov_tokens)].set_index("ProvID").loc[prov_tokens]
    ptype = a4_sub["ProvType"].fillna("Unknown").astype(str).values
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    prov_feats = ohe.fit_transform(ptype.reshape(-1, 1)).astype(np.float32)
    log(f"  provider feature dim: {prov_feats.shape[1]}")

    return hg, {"order_group": og_feats, "provider": prov_feats}, enc2id


# ============================================================
# 2-layer weighted-mean heterograph GNN
# ============================================================
class WeightedHeteroConv(nn.Module):
    """One heterograph conv layer. For each edge type, computes
        msg = h_src @ W_rel
        aggregate via weighted-mean: sum(w * msg) / sum(w)
    then per destination type, mean-aggregates over incoming rels."""
    def __init__(self, in_dims: dict, out_dim: int, rels: list):
        super().__init__()
        self.rels = rels
        self.lin = nn.ModuleDict({
            f"{s}__{r}__{d}": nn.Linear(in_dims[s], out_dim)
            for (s, r, d) in rels
        })

    def forward(self, hg, feats):
        with hg.local_scope():
            for nt, x in feats.items():
                hg.nodes[nt].data["h"] = x
            for (s, r, d) in self.rels:
                key = f"{s}__{r}__{d}"
                proj = self.lin[key](hg.nodes[s].data["h"])
                hg.nodes[s].data[f"proj_{r}"] = proj
                hg.apply_edges(fn.u_mul_e(f"proj_{r}", "w", f"m_{r}"), etype=r)
                hg.update_all(
                    lambda edges: {f"m_{r}": edges.data[f"m_{r}"]},
                    lambda nodes: {f"agg_{r}": nodes.mailbox[f"m_{r}"].mean(1)},
                    etype=r,
                )
            out = {}
            for nt in feats:
                aggs = [hg.nodes[nt].data[f"agg_{r}"]
                        for (s, rr, d) in self.rels if d == nt
                        for r in [rr] if f"agg_{r}" in hg.nodes[nt].data]
                if aggs:
                    out[nt] = torch.stack(aggs, 0).mean(0)
                else:
                    out[nt] = feats[nt]
            return out


class TemporalHeteroGNN(nn.Module):
    def __init__(self, n_enc_in, n_og_in, n_prov_in, hidden=EMB_DIM):
        super().__init__()
        rels = [
            ("encounter", "ordered", "order_group"),
            ("order_group", "ordered_by", "encounter"),
            ("encounter", "treated_by", "provider"),
            ("provider", "treats", "encounter"),
        ]
        self.enc_in = nn.Linear(n_enc_in, hidden)
        self.og_in = nn.Linear(n_og_in, hidden)
        self.prov_in = nn.Linear(n_prov_in, hidden)
        self.h1 = WeightedHeteroConv(
            {"encounter": hidden, "order_group": hidden, "provider": hidden},
            hidden, rels,
        )
        self.h2 = WeightedHeteroConv(
            {"encounter": hidden, "order_group": hidden, "provider": hidden},
            hidden, rels,
        )
        self.drop = nn.Dropout(0.2)
        self.clf = nn.Linear(hidden, 1)

    def forward(self, hg, feats):
        feats = {
            "encounter": F.relu(self.enc_in(feats["encounter"])),
            "order_group": F.relu(self.og_in(feats["order_group"])),
            "provider": F.relu(self.prov_in(feats["provider"])),
        }
        feats = {k: F.relu(v) for k, v in self.h1(hg, feats).items()}
        feats = {k: self.drop(v) for k, v in feats.items()}
        feats = {k: F.relu(v) for k, v in self.h2(hg, feats).items()}
        logits = self.clf(feats["encounter"]).squeeze(-1)
        return logits, feats["encounter"]


def train_temporal_gnn(hg, static_feats, X_enc_full, y, tr, val, N,
                       seed=SEED):
    """Train the temporal-edge GNN on encounter nodes indexed by tr,
    early-stop on val AUROC. Return encounter embedding (64-d) for all rows."""
    torch.manual_seed(seed)
    hg = hg.to(GNN_DEV)
    feats0 = {
        "encounter": torch.tensor(X_enc_full, dtype=torch.float32, device=GNN_DEV),
        "order_group": torch.tensor(static_feats["order_group"], dtype=torch.float32, device=GNN_DEV),
        "provider": torch.tensor(static_feats["provider"], dtype=torch.float32, device=GNN_DEV),
    }
    n_enc_in = X_enc_full.shape[1]
    n_og_in = static_feats["order_group"].shape[1]
    n_prov_in = static_feats["provider"].shape[1]

    model = TemporalHeteroGNN(n_enc_in, n_og_in, n_prov_in).to(GNN_DEV)
    opt = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=1e-4)
    y_t = torch.tensor(y, dtype=torch.float32, device=GNN_DEV)
    tr_t = torch.tensor(tr, dtype=torch.int64, device=GNN_DEV)
    val_t = torch.tensor(val, dtype=torch.int64, device=GNN_DEV)
    # class-weighted BCE
    pos_w = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())],
                         dtype=torch.float32, device=GNN_DEV)

    best_val_au, best_state, patience = -1, None, 0
    for epoch in range(GNN_EPOCHS):
        model.train()
        opt.zero_grad()
        logits, _ = model(hg, feats0)
        loss = F.binary_cross_entropy_with_logits(
            logits[tr_t], y_t[tr_t], pos_weight=pos_w)
        loss.backward()
        opt.step()

        # val AUROC
        model.eval()
        with torch.no_grad():
            logits_v, _ = model(hg, feats0)
            p_val = torch.sigmoid(logits_v[val_t]).cpu().numpy()
            if p_val.std() < 1e-6:
                val_au = 0.5
            else:
                val_au = roc_auc_score(y[val], p_val)
        if val_au > best_val_au + 1e-4:
            best_val_au = val_au
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= GNN_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, emb = model(hg, feats0)
        E = emb.cpu().numpy().astype(np.float32)
    log(f"  GNN best val AUROC {best_val_au:.3f} (epoch stop {epoch+1})  emb {E.shape}")
    return E


# ============================================================
# Main
# ============================================================
def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== FINAL PUSH v4 (temporal-edge graph, drop redundant) START ===")

    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID", "ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str)
    merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N} base rate {y.mean()*100:.2f}%")

    log("BLOCK B: LACE utility"); XB, _ = make_B(merged); log(f"  B {XB.shape}")
    log("BLOCK C: LACE + HOSPITAL scores"); XC, _ = make_C(merged); log(f"  C {XC.shape}")
    log("BLOCK H: geocode + ACS"); XH, _ = make_H(merged); log(f"  H {XH.shape}")

    # Build heterograph once
    log("BLOCK E_temp: building temporal-edge heterograph")
    from medhg_ps.data import load_order_sequence, collapse_order_runs
    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()
    hg, static_feats, enc2id = build_temporal_hg(
        merged, orders_df, raw.enc_prov_edges, raw.prov_attrs)
    log(f"  hg node types: {hg.ntypes}  edge types: {hg.etypes}")

    # 5-fold CV
    log("=== CROSS VALIDATION (5-fold seed 42) ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_oof = {"rf_base": np.full(N, np.nan),
             "rf_big":  np.full(N, np.nan),
             "hgb":     np.full(N, np.nan)}
    p_ab_no_Etemp = np.full(N, np.nan)

    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5  train {len(tr)} test {len(te)} ---")
        train_mask = np.zeros(N, bool); train_mask[tr] = True

        XA = make_A(merged, feat_cols, cpt_arr, train_mask)
        log(f"  A {XA.shape}")

        # split val from train
        rng = np.random.default_rng(SEED + fi)
        tr_perm = rng.permutation(tr)
        n_val = int(round(VAL_FRAC * N))
        val = tr_perm[:n_val]
        tr_only = tr_perm[n_val:]

        # Fit encounter-node preprocessing on tr_only, apply to all
        feat_all = merged[feat_cols].copy()
        _, st = fit_preprocess(
            feat_all.loc[tr_only].reset_index(drop=True), id_cols=[])
        X_enc = apply_preprocess(feat_all, st).astype(np.float32)

        log("  train temporal-edge GNN")
        try:
            XE_temp = train_temporal_gnn(
                hg, static_feats, X_enc, y, tr_only, val, N,
                seed=SEED + fi)
        except Exception as e:
            log(f"  E_temp FAILED: {e}; falling back to zeros")
            XE_temp = np.zeros((N, EMB_DIM), dtype=np.float32)

        X_full = np.hstack([XA, XB, XC, XH, XE_temp]).astype(np.float32)
        X_noE  = np.hstack([XA, XB, XC, XH]).astype(np.float32)
        if fi == 0:
            log(f"  X_full {X_full.shape}  X_noE {X_noE.shape}")

        for name, mk in [("rf_base", lambda: learner_rf()),
                         ("rf_big",  lambda: learner_rf(big=True)),
                         ("hgb",     lambda: learner_hgb())]:
            try:
                est = CalibratedClassifierCV(
                    mk(), method="isotonic", cv=3).fit(X_full[tr], y[tr])
                p_oof[name][te] = est.predict_proba(X_full[te])[:, 1]
                log(f"  fold {fi + 1} {name} done")
            except Exception as e:
                log(f"  fold {fi + 1} {name} FAILED: {e}")

        try:
            est = CalibratedClassifierCV(
                learner_rf(), method="isotonic", cv=3).fit(X_noE[tr], y[tr])
            p_ab_no_Etemp[te] = est.predict_proba(X_noE[te])[:, 1]
        except Exception as e:
            log(f"  ablation FAILED: {e}")

    log("=== POOLED OOF EVAL ===")
    results = []
    for name in ["rf_base", "rf_big", "hgb"]:
        p = p_oof[name]
        if np.isnan(p).any():
            log(f"  {name} has NaN, skipping"); continue
        r = eval_pooled_oof(y, p, name)
        log(f"  {name:10s} AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    valid = [k for k in p_oof if not np.isnan(p_oof[k]).any()]
    if len(valid) >= 2:
        p_ens = np.mean([p_oof[k] for k in valid], axis=0)
        r = eval_pooled_oof(y, p_ens, "ensemble_v4")
        log(f"  ensemble_v4 AUROC {r['auroc']:.3f} "
            f"({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    log("=== ABLATION (rf_base only) ===")
    ablation = []
    for tag, p in [("rf_base_full",     p_oof["rf_base"]),
                   ("rf_base_no_Etemp", p_ab_no_Etemp)]:
        if np.isnan(p).any():
            log(f"  {tag} has NaN skipping"); continue
        au = roc_auc_score(y, p); ap = average_precision_score(y, p)
        ablation.append(dict(config=tag, auroc=au, auprc=ap))
        log(f"  {tag:20s} AUROC {au:.3f}  AUPRC {ap:.3f}")

    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    pd.DataFrame(ablation).to_csv(OUT_ABL, index=False)
    np.savez(OUT_OOF, y=y, p_ab_no_Etemp=p_ab_no_Etemp, **p_oof)
    log(f"saved {OUT_RES}, {OUT_ABL}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
