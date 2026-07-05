"""
Explainability layer.

Implements the three XAI techniques applied in Chen et al. (npj Health
Systems 2025) §Explainable AI to identify important feature contributing
to predicting patient safety outcome:

    1. Meta-Path Analysis (MPA)
       "...the mean normalized attention coefficients of edges a^l from
        each model layer was recorded, capturing the interactions
        between specific node type pairs (e.g., Acute Care Unit and
        Encounter nodes). The overall importance of each meta-path was
        then determined by multiplying the attention coefficients along
        the sequence of edges in the meta-path."

       Implemented as: enumerate meta-paths of length up to `max_len`
       through the node-type graph, multiply per-layer mean attention
       weights along the edge sequence, sort.

    2. SHAP
       Captum's GradientShap on the encounter input features. The graph
       is held fixed; the encounter feature matrix is the input
       attributed against per-target encounter nodes.

    3. LIME
       lime.lime_tabular.LimeTabularExplainer over a model wrapper
       that takes a single encounter's feature row and returns a
       probability (graph structure is held fixed via lookup).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import config as C
from .graph import GraphArtifacts
from .model import MedHGPS


# =====================================================================
# Meta-Path Analysis
# =====================================================================
@dataclass
class MetaPath:
    sequence: Tuple[str, ...]  # e.g. (encounter, provider, encounter)
    importance: float          # product of mean attention weights along the path


def _step_weight(
    layer_attn: Dict[str, Dict[str, float]],
    src_type: str, dst_type: str, etype: str,
) -> float:
    """Returns the mean normalised attention weight that the layer
    placed on  src_type --etype--> dst_type  (i.e. how much the dst
    nodes' representations attended to the src channel)."""
    channel_key = f"{src_type}::{etype}"
    return layer_attn.get(dst_type, {}).get(channel_key, 0.0)


def enumerate_meta_paths(
    model: MedHGPS,
    max_len: int = 4,
    start_type: str = C.ENC_NTYPE,
) -> List[MetaPath]:
    """Walk the layer-aware graph schema, expanding meta-paths of the
    form  T_0 -> T_1 -> ... -> T_L  where L <= max_len. At each step we
    multiply by the corresponding layer's attention weight from the
    paper's Eq 2."""
    per_layer_attn = model.collect_attention()
    # Edge schema mapping target -> list of (src_type, etype_name).
    incoming: Dict[str, List[Tuple[str, str]]] = {nt: [] for nt in C.NODE_TYPES}
    for src, etype, dst in C.EDGE_TYPES:
        incoming[dst].append((src, etype))

    paths: List[MetaPath] = []
    # BFS over (current_type, current_layer, running_importance, sequence)
    frontier = [(start_type, 0, 1.0, (start_type,))]
    while frontier:
        nxt = []
        for cur_type, layer_idx, score, seq in frontier:
            if len(seq) > 1:
                paths.append(MetaPath(sequence=seq, importance=score))
            if layer_idx >= min(max_len - 1, len(per_layer_attn) - 1):
                continue
            for src, etype in incoming[cur_type]:
                w = _step_weight(per_layer_attn[layer_idx], src, cur_type, etype)
                if w <= 0:
                    continue
                nxt.append((src, layer_idx + 1, score * w, seq + (src,)))
        frontier = nxt

    paths.sort(key=lambda p: p.importance, reverse=True)
    return paths


def top_meta_paths(model: MedHGPS, k: int = 5,
                   max_len: int = 4) -> List[MetaPath]:
    return enumerate_meta_paths(model, max_len=max_len)[:k]


# =====================================================================
# SHAP via Captum GradientShap
# =====================================================================
def _enc_only_forward(
    model: MedHGPS, artifacts: GraphArtifacts, target_idx: int,
):
    """Returns a function f(X_enc) -> prob of readmission for
    `target_idx`, holding all other graph state fixed. Used by
    Captum (and by the LIME wrapper below)."""
    g  = artifacts.g
    fd = {nt: g.nodes[nt].data["h"] for nt in C.NODE_TYPES}

    def f(x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc has shape (batch_explain, n_enc_features). For each row
        # we substitute that row into position target_idx and forward.
        outs = []
        for row in x_enc:
            h_enc = fd[C.ENC_NTYPE].clone()
            h_enc[target_idx] = row
            local = dict(fd)
            local[C.ENC_NTYPE] = h_enc
            logits, _ = model(g, local)
            outs.append(torch.softmax(logits[target_idx], dim=-1)[1])
        return torch.stack(outs)

    return f


def explain_shap(
    model: MedHGPS, artifacts: GraphArtifacts,
    target_indices: List[int],
    n_baseline: int = 50,
    n_samples:  int = 5,
    seed: int = 0,
) -> np.ndarray:
    """SHAP-style attributions for encounter features on the listed
    target encounter nodes. Returns array (n_targets, n_enc_features)."""
    try:
        from captum.attr import GradientShap
    except ImportError as e:
        raise ImportError(
            "Install captum (`pip install captum`) to use explain_shap."
        ) from e

    g  = artifacts.g
    h_enc_all = g.nodes[C.ENC_NTYPE].data["h"]
    rng = np.random.default_rng(seed)

    # Baseline: random sample of encounter feature rows. The paper does
    # not specify the baseline distribution; the standard SHAP choice
    # is the marginal distribution of the training set.
    baseline_idx = rng.choice(h_enc_all.shape[0], size=n_baseline, replace=False)
    baselines = h_enc_all[baseline_idx]

    attribs = np.zeros((len(target_indices), h_enc_all.shape[1]), dtype=np.float32)
    for i, target in enumerate(target_indices):
        f = _enc_only_forward(model, artifacts, target)

        class _Wrap(torch.nn.Module):
            def forward(self_inner, x): return f(x)
        wrap = _Wrap()
        gs = GradientShap(wrap)
        # Attribute against the current encounter's feature row.
        inp = h_enc_all[target].unsqueeze(0).clone().requires_grad_(True)
        attr = gs.attribute(
            inputs=inp, baselines=baselines,
            n_samples=n_samples, stdevs=0.0,
        )
        attribs[i] = attr.detach().cpu().numpy().reshape(-1)
    return attribs


# =====================================================================
# LIME
# =====================================================================
def explain_lime(
    model: MedHGPS, artifacts: GraphArtifacts,
    target_indices: List[int],
    feature_names: List[str],
    n_samples: int = 1000,
    top_k:    int = 10,
    seed: int = 0,
) -> List[List[Tuple[str, float]]]:
    """LIME explanations - one (feature, weight) list per target node."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as e:
        raise ImportError(
            "Install lime (`pip install lime`) to use explain_lime."
        ) from e

    g = artifacts.g
    h_enc_all = g.nodes[C.ENC_NTYPE].data["h"].cpu().numpy()

    out: List[List[Tuple[str, float]]] = []
    for target in target_indices:
        explainer = LimeTabularExplainer(
            training_data=h_enc_all,
            feature_names=feature_names,
            class_names=["NoReadmit", "Readmit"],
            mode="classification",
            random_state=seed,
        )

        f_torch = _enc_only_forward(model, artifacts, target)

        def predict_proba(X: np.ndarray) -> np.ndarray:
            tx = torch.tensor(X, dtype=torch.float32,
                              device=g.nodes[C.ENC_NTYPE].data["h"].device)
            p1 = f_torch(tx).detach().cpu().numpy()
            return np.stack([1.0 - p1, p1], axis=1)

        exp = explainer.explain_instance(
            data_row=h_enc_all[target],
            predict_fn=predict_proba,
            num_samples=n_samples,
            num_features=top_k,
        )
        out.append(exp.as_list())
    return out
