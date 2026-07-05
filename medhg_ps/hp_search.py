"""
Hyperparameter search.

Tree-structured Parzen Estimator (TPE) search via Optuna. Optuna's
TPESampler is the canonical implementation of TPE (Bergstra et al.
NeurIPS 2011, reference 63 in the paper), which matches Chen et al.
(npj Health Systems 2025) §Experimental settings:

    "tree-structured parzen estimator algorithm was used to optimize
     the learning rate, L2 regularization, dropout rate, batch
     normalization, the dimension of the first hidden layer, and
     attention vector size."

Search ranges from paper Table 2 row "MedHG-PS":

    Learning rate                10^-7 .. 10^-1      log-uniform
    L2 regularisation            10^-7 .. 10^-1      log-uniform
    Dropout rate                 0.0   .. 0.8        uniform
    Batch normalisation          {True, False}       categorical
    First hidden dimension       {32, 64, 128, 256}  categorical
    Attention vector dim         {8, 16, 32, 64, 128} categorical

Objective: validation AUROC. Best configuration is retrained from
scratch and returned along with its TrainResult and the final model.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

from . import config as C
from .graph import GraphArtifacts
from .model import MedHGPS
from .train import TrainResult, train_model


# =====================================================================
def run_hp_search(
    artifacts: GraphArtifacts,
    base_cfg: C.TrainConfig = C.DEFAULTS_TRAIN,
    space:    C.HPSearchSpace = C.DEFAULTS_SEARCH,
    save_dir: Optional[Path] = None,
    verbose:  bool = True,
) -> Tuple[C.TrainConfig, TrainResult, MedHGPS]:
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as e:
        raise ImportError(
            "Install optuna (`pip install optuna`) to use hp_search. "
            "Optuna's TPESampler is the canonical TPE implementation "
            "referenced by the paper."
        ) from e

    # Silence Optuna's per-trial INFO chatter unless verbose=True; we
    # do our own one-line-per-trial printing below.
    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )

    lr_lo, lr_hi   = space.learning_rate
    l2_lo, l2_hi   = space.l2_reg
    do_lo, do_hi   = space.dropout

    def objective(trial: "optuna.Trial") -> float:
        learning_rate = trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True)
        l2_reg        = trial.suggest_float("l2_reg",        l2_lo, l2_hi, log=True)
        dropout       = trial.suggest_float("dropout",       do_lo, do_hi)
        batch_norm    = trial.suggest_categorical("batch_norm",
                                                  list(space.batch_norm))
        hidden_dim_1  = trial.suggest_categorical("hidden_dim_1",
                                                  list(space.hidden_dim_1))
        attn_dim      = trial.suggest_categorical("attn_dim",
                                                  list(space.attn_dim))

        cfg = replace(
            base_cfg,
            learning_rate=float(learning_rate),
            l2_reg=float(l2_reg),
            dropout=float(dropout),
            batch_norm=bool(batch_norm),
            hidden_dim_1=int(hidden_dim_1),
            hidden_dim_2=max(int(hidden_dim_1) // 2, 8),
            hidden_dim_3=max(int(hidden_dim_1) // 4, 8),
            attn_dim=int(attn_dim),
            max_epochs=min(base_cfg.max_epochs, 80),  # cheaper inner loop
            early_stop_patience=10,
        )
        if verbose:
            print(f"[HP trial {trial.number:3d}] "
                  f"lr={cfg.learning_rate:.2e}, l2={cfg.l2_reg:.2e}, "
                  f"do={cfg.dropout:.2f}, bn={cfg.batch_norm}, "
                  f"h1={cfg.hidden_dim_1}, attn={cfg.attn_dim}")
        _, result = train_model(artifacts, cfg=cfg, verbose=False)
        if verbose:
            print(f"[HP trial {trial.number:3d}]   -> val AUROC = "
                  f"{result.best_val_auroc:.4f}")
        return float(result.best_val_auroc)

    sampler = TPESampler(seed=42)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=space.n_calls, show_progress_bar=False)

    best_params = study.best_params
    if verbose:
        print(f"[HP] best val AUROC = {study.best_value:.4f}")
        print(f"[HP] best params    = {best_params}")

    best_cfg = replace(
        base_cfg,
        learning_rate=float(best_params["learning_rate"]),
        l2_reg=float(best_params["l2_reg"]),
        dropout=float(best_params["dropout"]),
        batch_norm=bool(best_params["batch_norm"]),
        hidden_dim_1=int(best_params["hidden_dim_1"]),
        hidden_dim_2=max(int(best_params["hidden_dim_1"]) // 2, 8),
        hidden_dim_3=max(int(best_params["hidden_dim_1"]) // 4, 8),
        attn_dim=int(best_params["attn_dim"]),
    )
    final_model, final_result = train_model(
        artifacts, cfg=best_cfg, save_dir=save_dir, verbose=verbose,
    )
    return best_cfg, final_result, final_model
