"""
Training loop.

Mirrors the protocol described in Chen et al. (npj Health Systems 2025)
§Methods - The proposed framework and Experimental settings:

    * Adam optimiser with binary cross-entropy loss (their Eq 5):

        L = -(1/N) sum_{i=1..N} [ y_i log(yhat_i) + (1 - y_i) log(1 - yhat_i) ]

    * 8 : 1 : 1 train / validation / test split (Methods § Experimental
      settings), early stopping on validation AUROC.

    * Batch size 512 (Methods § Experimental settings).

    * Imbalanced-learn resampling experiments (Methods § Ablation):
      undersample, oversample, or SMOTE on the training encounters only.

    * Class-weighting fallback when resampling = "none" - this is what
      we use for our 30-day readmission cohort, which is heavily
      imbalanced just like the paper's mortality outcomes.

The graph is small enough (paper: 102 k patients, 3.2 M params,
75.85 M MACs, 0.03 s / patient on A100) that full-batch training over
the entire heterograph fits comfortably on a single GPU. We therefore
do not subgraph-sample and take a single full-batch gradient step per
epoch over the (optionally resampled) training encounters.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from . import config as C
from .graph import GraphArtifacts
from .model import MedHGPS


# =====================================================================
def set_seed(seed: int) -> None:
    """Seed every RNG that affects training so runs are reproducible.

    Without this only the numpy data split is deterministic; weight init
    (xavier), dropout, and torch.randperm draw from the unseeded global
    torch RNG, so two runs of the same config give different models.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import dgl
        dgl.seed(seed)
    except Exception:
        pass


# =====================================================================
@dataclass
class TrainResult:
    best_val_auroc: float
    best_epoch: int
    best_state_dict: dict
    history: list  # list[dict] per epoch


# =====================================================================
def _resample_train_indices(
    train_mask: torch.Tensor,
    y: torch.Tensor,
    method: str,
    seed: int = 0,
) -> torch.Tensor:
    """Returns indices into ENC nodes that form the training batch pool
    after resampling. Operates on the train split only."""
    rng = np.random.default_rng(seed)
    idx = torch.nonzero(train_mask, as_tuple=False).flatten().cpu().numpy()
    if method == "none":
        return torch.tensor(idx, dtype=torch.long)

    y_arr = y[idx].cpu().numpy()
    pos = idx[y_arr == 1]
    neg = idx[y_arr == 0]

    if method == "undersample":
        n = min(len(pos), len(neg))
        pos_s = rng.choice(pos, size=n, replace=False)
        neg_s = rng.choice(neg, size=n, replace=False)
        out = np.concatenate([pos_s, neg_s])
    elif method == "oversample":
        n = max(len(pos), len(neg))
        pos_s = rng.choice(pos, size=n, replace=len(pos) < n)
        neg_s = rng.choice(neg, size=n, replace=len(neg) < n)
        out = np.concatenate([pos_s, neg_s])
    elif method == "smote":
        # SMOTE on a graph is not meaningful (we cannot synthesise nodes
        # without inventing edges). Fall back to oversample with replace.
        return _resample_train_indices(train_mask, y, "oversample", seed)
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    rng.shuffle(out)
    return torch.tensor(out, dtype=torch.long)


# =====================================================================
def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


# =====================================================================
def train_model(
    artifacts: GraphArtifacts,
    cfg: C.TrainConfig = C.DEFAULTS_TRAIN,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[MedHGPS, TrainResult]:
    device = _resolve_device(cfg.device)
    set_seed(cfg.split_seed)   # deterministic weight init / dropout / shuffles
    g = artifacts.g.to(device)

    in_dims = {
        C.ENC_NTYPE:  artifacts.n_enc_features,
        C.PROV_NTYPE: artifacts.n_prov_features,
        C.UNIT_NTYPE: artifacts.n_unit_features,
    }
    model = MedHGPS(in_dims=in_dims, cfg=cfg).to(device)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.l2_reg,
    )

    # Initial node features stay on the GPU; the layers refresh them.
    def get_feature_dict() -> Dict[str, torch.Tensor]:
        return {nt: g.nodes[nt].data["h"] for nt in C.NODE_TYPES}

    y          = g.nodes[C.ENC_NTYPE].data["y"]
    train_mask = g.nodes[C.ENC_NTYPE].data["train_mask"]
    val_mask   = g.nodes[C.ENC_NTYPE].data["val_mask"]

    # Class weights when no resampling is used. Inverse-frequency
    # weights stabilise BCE on the heavily-imbalanced readmission task.
    if cfg.resampling == "none":
        n_pos = int((y[train_mask] == 1).sum())
        n_neg = int((y[train_mask] == 0).sum())
        w = torch.tensor(
            [n_pos + n_neg, (n_pos + n_neg)],
            dtype=torch.float32, device=device,
        ) / torch.tensor(
            [max(n_neg, 1) * 2, max(n_pos, 1) * 2],
            dtype=torch.float32, device=device,
        )
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_val      = -1.0
    best_state    = None
    best_epoch    = -1
    patience_left = cfg.early_stop_patience
    history       = []

    for epoch in range(cfg.max_epochs):
        # ---- Resample once per epoch (paper protocol) ----
        train_pool = _resample_train_indices(
            train_mask.cpu(), y.cpu(), cfg.resampling, seed=cfg.split_seed + epoch
        ).to(device)

        # ---- Full-batch gradient step over the (resampled) train pool ----
        # The heterograph is small enough to fit in one forward (see module
        # docstring), so we take a single full-batch step per epoch. The
        # previous code recomputed the whole-graph forward once per minibatch
        # slice but only backpropped that slice -- wasting compute and
        # silently coupling the number of gradient steps to batch_size.
        model.train()
        logits, _ = model(g, get_feature_dict())
        loss = loss_fn(logits[train_pool], y[train_pool])

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        epoch_loss = loss.item()

        # ---- Validation AUROC for early stopping ----
        model.eval()
        with torch.no_grad():
            logits, _ = model(g, get_feature_dict())
            probs     = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            val_y     = y[val_mask].cpu().numpy()
            val_p     = probs[val_mask.cpu().numpy()]
            try:
                val_auroc = roc_auc_score(val_y, val_p)
            except ValueError:
                val_auroc = 0.5  # single-class minibatch

        history.append({
            "epoch": epoch,
            "loss":  epoch_loss,
            "val_auroc": val_auroc,
        })

        if verbose and (epoch % 5 == 0 or epoch == cfg.max_epochs - 1):
            print(f"[epoch {epoch:3d}] loss={epoch_loss:.4f}"
                  f"  val AUROC={val_auroc:.4f}  best={best_val:.4f}")

        if val_auroc > best_val:
            best_val      = val_auroc
            best_state    = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
            best_epoch    = epoch
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"[epoch {epoch:3d}] early stop (no val AUROC improvement)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_dir / "medhg_ps_best.pt")

    return model, TrainResult(
        best_val_auroc=best_val,
        best_epoch=best_epoch,
        best_state_dict=best_state or {},
        history=history,
    )
