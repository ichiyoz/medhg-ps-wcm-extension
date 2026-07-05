"""Graph-Aware Random Forest (v12) — real tree-based graph model.

Implements the actual G-GBM/GGBoost-style architecture the user asked
about: trees where split rules can be GRAPH QUERIES rather than only
axis-parallel scalar comparisons.

Split types considered at each node:
  A) Axis-parallel      x[i, j] > threshold          (standard)
  B) Neighborhood-mean  mean(x[N(i), j]) > threshold  (static graph query)
  C) Neighborhood-max   max(x[N(i), j]) > threshold   (static graph query)
  D) Neighborhood-share  |N(i) ∩ current_node_samples| / |N(i)| > threshold
     ← GENUINELY graph-aware: depends on which samples are in the current
     node's partition, which changes as the tree grows. This is what makes
     the forest a "tree-based graph algorithm" in the sense of G-GBM.

Prediction: for a test sample, traverse the tree evaluating the stored
split rule at each node. For type-D splits, the "current node samples"
recorded during training are used as the reference partition.

Wrapping: sklearn-compatible fit/predict_proba so it drops into
CalibratedClassifierCV like any other estimator.

Cohort: N=13,858 (corrected), Discharge Disposition included as tabular
feature. Feature matrix: A + B + C + H + dispo. Graph: encounter–provider
and encounter–order-group edges (tripartite via v6's precompute_graph).

Ensemble: 100 trees, max_depth 10, min_samples_leaf 10, 30 split
candidates per node (mix of A/B/C/D). Bootstrap sampling per tree.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.final_push_080 import (
    log, make_A, make_B, make_C, make_H, eval_pooled_oof,
    SEED, DATA_DIR, GOLD,
)
from analysis.final_push_080_v6 import precompute_graph
from analysis.final_push_080_v8 import _norm_id

from medhg_ps.data import load_raw, load_order_sequence, collapse_order_runs
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.evaluate import _bootstrap_ci


OUT_LOG = Path("artifacts/newdata/graph_aware_forest_v12.log")
OUT_RES = Path("artifacts/newdata/graph_aware_forest_v12_results.csv")
OUT_OOF = Path("artifacts/newdata/graph_aware_forest_v12_oof.npz")


EXCLUDE_LABELS = {
    "Expired", "Expired in Medical Facility",
    "Hospice/Home", "Hospice/Medical Facility",
    "Acute / Short Term Hospital", "Left Against Medical Advice",
}
HOME_LABELS = {"Home or Self Care", "Home-Health Care Svc"}


# ============================================================
# The tree
# ============================================================
class _GraphAwareTree:
    """A single graph-aware decision tree with four split types.

    Nodes are stored as a dict list; leaves store class probability.

    adj_indptr, adj_indices define a CSR adjacency over the ENCOUNTER
    node set — X is aligned to encounters, N = X.shape[0]. Neighbors of
    i are indices[indptr[i]:indptr[i+1]]. Neighbors are encounter indices
    (they too have rows in X). For the tripartite graph we collapse to
    encounter-encounter co-occurrence via shared providers/order-groups.
    """
    __slots__ = ("max_depth", "min_samples_leaf", "n_split_candidates",
                 "share_prob", "nbr_agg_prob", "rng",
                 "nodes_",   # list of dicts, root at index 0
                 )

    def __init__(self, max_depth, min_samples_leaf, n_split_candidates,
                 share_prob, nbr_agg_prob, rng):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_split_candidates = n_split_candidates
        self.share_prob = share_prob      # prob of type-D split rule
        self.nbr_agg_prob = nbr_agg_prob   # prob of type-B/C
        self.rng = rng
        self.nodes_ = []

    # -----------------------------------------------------
    # Fit
    # -----------------------------------------------------
    def fit(self, X, y, w, sample_ids, feat_ids,
            adj_indptr, adj_indices):
        """Grow tree on the bootstrap sample_ids using feat_ids as the
        random feature subset for this tree."""
        self.nodes_ = []
        stack = [(sample_ids, 0, -1, None)]  # (samples, depth, parent, is_left)

        while stack:
            samples, depth, parent, is_left = stack.pop()

            node_idx = len(self.nodes_)
            self.nodes_.append(dict(
                is_leaf=True, leaf_prob=self._leaf_prob(y, samples, w),
                sample_ids=samples,
            ))
            if parent >= 0:
                key = "left" if is_left else "right"
                self.nodes_[parent][key] = node_idx

            if (depth >= self.max_depth
                    or len(samples) < 2 * self.min_samples_leaf
                    or np.all(y[samples] == y[samples[0]])):
                continue

            best = self._find_best_split(
                X, y, w, samples, feat_ids, adj_indptr, adj_indices)
            if best is None:
                continue
            split_type, feat, thresh, left, right = best
            if len(left) < self.min_samples_leaf or \
               len(right) < self.min_samples_leaf:
                continue

            self.nodes_[node_idx].update(
                is_leaf=False, split_type=split_type,
                feat=feat, thresh=thresh,
                # For type-D we save the TRAINING samples in the current node
                # for use at inference time.
                partition_samples=(samples if split_type == "share" else None),
            )
            # push right first so left is popped first (deterministic order)
            stack.append((right, depth + 1, node_idx, False))
            stack.append((left, depth + 1, node_idx, True))

    # -----------------------------------------------------
    # Leaf value
    # -----------------------------------------------------
    def _leaf_prob(self, y, samples, w):
        if len(samples) == 0:
            return 0.5
        num = float((w[samples] * y[samples]).sum())
        den = float(w[samples].sum())
        return num / den if den > 0 else 0.5

    # -----------------------------------------------------
    # Find best split
    # -----------------------------------------------------
    def _find_best_split(self, X, y, w, samples, feat_ids,
                          adj_indptr, adj_indices):
        best_gain = -np.inf
        best = None
        S = np.asarray(samples, dtype=np.int64)
        y_s = y[S]
        w_s = w[S]
        base_gini = self._gini(y_s, w_s)

        for _ in range(self.n_split_candidates):
            r = self.rng.random()
            if r < self.share_prob:
                stype = "share"
            elif r < self.share_prob + self.nbr_agg_prob:
                stype = self.rng.choice(("nbr_mean", "nbr_max"))
            else:
                stype = "axis"

            if stype == "axis":
                feat = int(self.rng.choice(feat_ids))
                vals = X[S, feat]
                thresh = self._sample_threshold(vals)
                if thresh is None:
                    continue
                mask = vals <= thresh
            elif stype in ("nbr_mean", "nbr_max"):
                feat = int(self.rng.choice(feat_ids))
                vals = self._nbr_agg(X, S, feat, stype,
                                     adj_indptr, adj_indices)
                thresh = self._sample_threshold(vals)
                if thresh is None:
                    continue
                mask = vals <= thresh
            elif stype == "share":
                # Fraction of neighbors that are ALSO in current node.
                in_node = np.zeros(X.shape[0], dtype=bool)
                in_node[S] = True
                vals = self._nbr_share(in_node, S, adj_indptr, adj_indices)
                thresh = self._sample_threshold(vals)
                if thresh is None:
                    continue
                mask = vals <= thresh
                feat = -1   # marker
            else:
                continue

            gain = self._split_gain(y_s, w_s, mask, base_gini)
            if gain > best_gain:
                best_gain = gain
                left = S[mask]; right = S[~mask]
                best = (stype, feat, float(thresh), left, right)
        return best

    def _sample_threshold(self, vals):
        if vals.size < 4:
            return None
        pct = float(self.rng.uniform(0.1, 0.9))
        thr = float(np.quantile(vals, pct))
        return thr

    def _nbr_agg(self, X, S, feat, stype, indptr, indices):
        out = np.empty(len(S), dtype=np.float32)
        for k, i in enumerate(S):
            lo, hi = indptr[i], indptr[i + 1]
            if hi == lo:
                out[k] = X[i, feat]
            else:
                nbr_vals = X[indices[lo:hi], feat]
                out[k] = nbr_vals.mean() if stype == "nbr_mean" else nbr_vals.max()
        return out

    def _nbr_share(self, in_node, S, indptr, indices):
        out = np.empty(len(S), dtype=np.float32)
        for k, i in enumerate(S):
            lo, hi = indptr[i], indptr[i + 1]
            if hi == lo:
                out[k] = 0.0
            else:
                nbrs = indices[lo:hi]
                out[k] = float(in_node[nbrs].mean())
        return out

    @staticmethod
    def _gini(y, w):
        s = float(w.sum())
        if s <= 0:
            return 0.0
        p = float((w * y).sum() / s)
        return 2.0 * p * (1.0 - p)

    def _split_gain(self, y, w, mask, base):
        wL = w[mask]; wR = w[~mask]
        sL = float(wL.sum()); sR = float(wR.sum()); s = sL + sR
        if sL <= 0 or sR <= 0:
            return -np.inf
        yL = y[mask]; yR = y[~mask]
        gL = self._gini(yL, wL); gR = self._gini(yR, wR)
        return base - (sL / s) * gL - (sR / s) * gR

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------
    def predict_proba(self, X, adj_indptr, adj_indices):
        N = X.shape[0]
        probs = np.zeros(N, dtype=np.float32)
        for i in range(N):
            node = 0
            while not self.nodes_[node]["is_leaf"]:
                nd = self.nodes_[node]
                stype = nd["split_type"]; feat = nd["feat"]; thresh = nd["thresh"]
                if stype == "axis":
                    val = X[i, feat]
                elif stype == "nbr_mean":
                    lo, hi = adj_indptr[i], adj_indptr[i + 1]
                    val = X[i, feat] if hi == lo else \
                          X[adj_indices[lo:hi], feat].mean()
                elif stype == "nbr_max":
                    lo, hi = adj_indptr[i], adj_indptr[i + 1]
                    val = X[i, feat] if hi == lo else \
                          X[adj_indices[lo:hi], feat].max()
                elif stype == "share":
                    part = nd["partition_samples"]
                    part_set = getattr(nd, "_part_set", None)
                    if part_set is None:
                        part_set = set(int(x) for x in part)
                        nd["_part_set"] = part_set
                    lo, hi = adj_indptr[i], adj_indptr[i + 1]
                    if hi == lo:
                        val = 0.0
                    else:
                        nbrs = adj_indices[lo:hi]
                        val = sum(1 for x in nbrs if int(x) in part_set) / len(nbrs)
                if val <= thresh:
                    node = nd["left"]
                else:
                    node = nd["right"]
            probs[i] = self.nodes_[node]["leaf_prob"]
        return probs


# ============================================================
# The forest (sklearn-compatible)
# ============================================================
class GraphAwareForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=10, min_samples_leaf=10,
                 n_split_candidates=25, feat_frac=0.5, share_prob=0.15,
                 nbr_agg_prob=0.30, class_weight="balanced",
                 adj_indptr=None, adj_indices=None,
                 random_state=42, verbose=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_split_candidates = n_split_candidates
        self.feat_frac = feat_frac
        self.share_prob = share_prob
        self.nbr_agg_prob = nbr_agg_prob
        self.class_weight = class_weight
        self.adj_indptr = adj_indptr
        self.adj_indices = adj_indices
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, train_ids=None):
        """X is FULL-cohort (N, D). train_ids selects which rows to train on.
        The graph adjacency is aligned to X (full cohort), so neighbor
        lookups always work regardless of which rows are training rows."""
        assert self.adj_indptr is not None and self.adj_indices is not None, \
            "pass adj_indptr and adj_indices in __init__"
        self.adj_indptr_ = np.asarray(self.adj_indptr, dtype=np.int64)
        self.adj_indices_ = np.asarray(self.adj_indices, dtype=np.int64)
        if train_ids is None:
            train_ids = np.arange(len(y), dtype=np.int64)
        train_ids = np.asarray(train_ids, dtype=np.int64)
        rng_root = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        self.classes_ = np.array([0, 1])
        N, D = X.shape
        if self.class_weight == "balanced":
            n_pos = int(y.sum()); n_neg = N - n_pos
            wp = float(N) / max(1, 2 * n_pos)
            wn = float(N) / max(1, 2 * n_neg)
            w = np.where(y == 1, wp, wn).astype(np.float32)
        else:
            w = np.ones(N, dtype=np.float32)
        self.trees_ = []
        n_train = len(train_ids)
        for t in range(self.n_estimators):
            rng = np.random.default_rng(int(rng_root.integers(0, 2**31 - 1)))
            # bootstrap from TRAIN ids only (still returns full-cohort row indices)
            boot = train_ids[rng.integers(0, n_train, size=n_train)]
            k_feats = max(1, int(D * self.feat_frac))
            feat_ids = rng.choice(D, size=k_feats, replace=False)
            tree = _GraphAwareTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_split_candidates=self.n_split_candidates,
                share_prob=self.share_prob,
                nbr_agg_prob=self.nbr_agg_prob,
                rng=rng,
            )
            tree.fit(X, y, w, boot, feat_ids,
                     self.adj_indptr_, self.adj_indices_)
            self.trees_.append(tree)
            if self.verbose and (t + 1) % 20 == 0:
                log(f"    tree {t+1}/{self.n_estimators}")
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]
        probs = np.zeros(N, dtype=np.float32)
        for tree in self.trees_:
            probs += tree.predict_proba(X, self.adj_indptr_, self.adj_indices_)
        probs /= len(self.trees_)
        out = np.zeros((N, 2), dtype=np.float32)
        out[:, 1] = probs; out[:, 0] = 1.0 - probs
        return out



# ============================================================
# Build encounter–encounter adjacency
# ============================================================
def build_encounter_encounter_adj(gs, k=8):
    """kNN encounter-encounter graph — for each encounter, keep the top-k
    most similar others by shared providers + shared order-groups.
    Keeping degree bounded is critical for tree training speed."""
    from scipy.sparse import csr_matrix
    M_ep = (gs["M_ep"] > 0).astype(np.float32)
    M_eo = (gs["M_eo"] > 0).astype(np.float32)
    S = (M_ep @ M_ep.T + M_eo @ M_eo.T).tocsr()
    S.setdiag(0); S.eliminate_zeros()
    N = S.shape[0]
    rows, cols = [], []
    for i in range(N):
        lo, hi = S.indptr[i], S.indptr[i + 1]
        if hi == lo:
            continue
        vals = S.data[lo:hi]
        cs = S.indices[lo:hi]
        if len(vals) <= k:
            top = cs
        else:
            idx = np.argpartition(-vals, k)[:k]
            top = cs[idx]
        for c in top:
            rows.append(i); cols.append(int(c))
            rows.append(int(c)); cols.append(i)
    # dedupe
    edges = set(zip(rows, cols))
    rows2 = np.fromiter((r for r, _ in edges), dtype=np.int64)
    cols2 = np.fromiter((c for _, c in edges), dtype=np.int64)
    order = np.argsort(rows2)
    rows_s = rows2[order]; cols_s = cols2[order]
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, rows_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, cols_s.astype(np.int64)


# ============================================================
# Main
# ============================================================
def _boot(y, p, metric, seed):
    return _bootstrap_ci(y, p, metric, n_boot=2000, seed=seed)


def _eval(y, p, name):
    au = roc_auc_score(y, p); ap = average_precision_score(y, p)
    au_ci = _boot(y, p, roc_auc_score, 0)
    ap_ci = _boot(y, p, average_precision_score, 1)
    br = brier_score_loss(y, p)
    best_f1 = 0
    for t in np.linspace(0.02, 0.35, 80):
        yh = (p >= t).astype(int)
        if yh.sum() < 5: continue
        best_f1 = max(best_f1, f1_score(y, yh))
    return dict(model=name, auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1],
                brier=br, f1=best_f1)


def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== GRAPH-AWARE FOREST v12 START ===")

    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID", "ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str)
    merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    excl_mask = merged["Discharge Disposition"].astype(str).isin(EXCLUDE_LABELS)
    merged = merged.loc[~excl_mask].reset_index(drop=True)
    cpt_arr = np.asarray(cpt_arr).ravel()[~excl_mask.values].reshape(-1, 1)
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N} base rate {y.mean()*100:.2f}%")

    XB, _ = make_B(merged); XC, _ = make_C(merged); XH, _ = make_H(merged)
    X_dispo = merged["Discharge Disposition"].astype(str).isin(HOME_LABELS)\
                 .astype(np.float32).values.reshape(-1, 1)
    train_mask = np.ones(N, bool)
    XA = make_A(merged, feat_cols, cpt_arr, train_mask)
    X_full = np.hstack([XA, XB, XC, XH, X_dispo]).astype(np.float32)
    log(f"X_full {X_full.shape}")

    log("building encounter–encounter adjacency")
    orders_df = collapse_order_runs(load_order_sequence())
    raw = load_raw()
    kept = set(_norm_id(merged["LogID"]).values)
    ep = raw.enc_prov_edges.copy(); ep["LogID"] = _norm_id(ep["LogID"])
    ep = ep[ep["LogID"].isin(kept)]
    gs = precompute_graph(merged, orders_df, ep, raw.prov_attrs)
    indptr, indices = build_encounter_encounter_adj(gs, k=8)
    log(f"  enc-enc kNN adj: {len(indices)//2:,} undirected edges  "
        f"avg deg {len(indices)/N:.1f}")

    log("=== 5-fold OUTER CV ===")
    p_gaf = np.full(N, np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5 ---")
        t0 = time.time()
        gaf = GraphAwareForest(
            n_estimators=50, max_depth=8, min_samples_leaf=30,
            n_split_candidates=20, feat_frac=0.5,
            share_prob=0.20, nbr_agg_prob=0.30,
            class_weight="balanced",
            adj_indptr=indptr, adj_indices=indices,
            random_state=SEED + fi, verbose=1,
        )
        # Fit on FULL cohort features with train_ids selecting the fold's train rows.
        # This lets the graph adjacency (aligned to full cohort) work correctly.
        gaf.fit(X_full, y, train_ids=tr)
        # Predict on all rows; extract test slice for OOF.
        p_all = gaf.predict_proba(X_full)[:, 1]
        p_gaf[te] = p_all[te]
        log(f"  fold {fi + 1} done in {time.time()-t0:.0f}s")

    r = _eval(y, p_gaf, "Graph-aware forest (v12)")
    log(f"  {r['model']:35s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) AUPRC {r['auprc']:.3f}")
    pd.DataFrame([r]).to_csv(OUT_RES, index=False)
    np.savez(OUT_OOF, y=y, p_gaf=p_gaf)
    log(f"saved {OUT_RES}, {OUT_OOF}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
