"""5-fold CV of the ORIGINAL MedHG-PS method (paper's protocol) on the
corrected cohort.

- Uses medhg_ps package UNCHANGED (build_graph, train_model, softmax head).
- Encounter features = paper's 40-feature NSQIP allow-list + pre-op
  trajectory + calendar (NOT the enriched Blocks A+B+C+H+dispo).
- Graph = A3 care-unit graph (encounter+provider+care-unit heterograph),
  NOT orders-as-A3.
- Prediction from ie-HGCN classifier head via softmax — no downstream
  RF / LightGBM / XGBoost ensemble.
- Wrapped in 5-fold StratifiedKFold with seed 42 to produce pooled OOF
  for direct comparison to the other Table 2 rows.

Env-var overrides used to point at the corrected-cohort parquet:
    MEDHG_PS_ENC_FEATURES_CSV, MEDHG_PS_LABELS
"""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import medhg_ps.config as C
from medhg_ps.data import (
    add_calendar_features, apply_preprocess, build_preop_trajectory_features,
    build_provider_features, build_unit_features, fit_preprocess, load_raw,
)
from medhg_ps.graph import build_graph
from medhg_ps.train import set_seed, train_model
from medhg_ps.evaluate import _bootstrap_ci

OUT_RES = Path("artifacts/newdata/medhg_ps_5fold_original_results.csv")
OUT_OOF = Path("artifacts/newdata/medhg_ps_5fold_original_oof.npz")
LOG = Path("artifacts/newdata/medhg_ps_5fold_original.log")


def log(msg):
    line = msg if msg.startswith("[") else f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def main():
    OUT_RES.parent.mkdir(parents=True, exist_ok=True)
    log("=== MedHG-PS ORIGINAL METHOD, 5-fold CV, corrected cohort ===")

    # Step 1: load raw + merge (same as run.py)
    raw = load_raw()
    enc_features_no_dupes = raw.enc_features.drop(
        columns=([c for c in raw.encounters.columns
                  if c != "LogID" and c in raw.enc_features.columns]
                 + ["ReadmittedWithin30Days"]),
        errors="ignore")
    merged_all = (raw.encounters
                  .merge(enc_features_no_dupes, on="LogID", how="inner")
                  .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]],
                         on="LogID", how="inner")
                  .reset_index(drop=True))
    ss = merged_all[["LogID"]].copy()
    ss["_ss"] = pd.to_datetime(merged_all.get("Procedure/Surgery Start"),
                               errors="coerce")
    traj = build_preop_trajectory_features(raw.enc_unit_edges, ss)
    merged_all = merged_all.merge(traj, on="LogID", how="left")
    for c_ in C.TRAJECTORY_FEATURE_COLUMNS:
        merged_all[c_] = merged_all[c_].fillna(0)
    merged_all = add_calendar_features(merged_all)
    N = len(merged_all)
    y = merged_all["ReadmittedWithin30Days"].astype(int).values
    log(f"N={N}  base rate {y.mean()*100:.2f}%")

    # Step 2: build feature block (paper's 40-feature allow-list + derived)
    nsqip_cols = [c for c in C.MODEL_FEATURE_COLUMNS if c in merged_all.columns]
    derived_cols = [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged_all.columns]
    feat_cols = nsqip_cols + derived_cols
    log(f"features: {len(nsqip_cols)} NSQIP + {len(derived_cols)} derived = {len(feat_cols)}")
    feat_all = merged_all[feat_cols].copy()

    # Provider/unit features (only need to build once — no leakage)
    prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
    unit_ids, X_unit, _ = build_unit_features(raw.unit_attrs)

    # Step 3: 5-fold pooled OOF
    p_oof = np.full(N, np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for fi, (tr_full, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi + 1}/5 ---")
        # Inner val split from train indices (paper uses 8:1:1; we use
        # ~9:1 within the 80% train fold to get 80:10:10 overall)
        rng = np.random.default_rng(fi + 42)
        idx = tr_full.copy(); rng.shuffle(idx)
        n_val = int(0.111 * len(idx))    # 0.111 * 0.8 ≈ 0.089 of total
        va = np.sort(idx[:n_val])
        tr = np.sort(idx[n_val:])
        train_mask = np.zeros(N, bool); train_mask[tr] = True
        val_mask   = np.zeros(N, bool); val_mask[va]  = True
        test_mask  = np.zeros(N, bool); test_mask[te] = True
        log(f"  tr={train_mask.sum()}  va={val_mask.sum()}  te={test_mask.sum()}")

        # Fit encounter preprocessing on train rows only (fold-local)
        _, enc_state = fit_preprocess(
            feat_all.loc[train_mask].reset_index(drop=True), id_cols=[])
        X_enc = apply_preprocess(feat_all, enc_state)

        # Build A3 care-unit heterograph with fold masks
        artifacts = build_graph(
            raw=raw, encounters_merged=merged_all, enc_features=X_enc,
            prov_ids=prov_ids, prov_features=X_prov,
            unit_ids=unit_ids, unit_features=X_unit,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        )
        set_seed(C.DEFAULTS_TRAIN.split_seed)
        model, _ = train_model(artifacts, cfg=C.DEFAULTS_TRAIN,
                                save_dir=None, verbose=False)

        # Softmax head prediction on val + test; per-fold isotonic
        # calibration fit on val to bring cross-fold outputs onto a common
        # scale (matches the CalibratedClassifierCV protocol used by the
        # other Table 2 rows). No downstream tree ensemble — the softmax
        # head remains the sole predictor; isotonic only rescales
        # probabilities monotonically.
        from sklearn.isotonic import IsotonicRegression
        model.eval()
        g_dev = artifacts.g.to(C.DEFAULTS_TRAIN.device)
        with torch.no_grad():
            logits, _ = model.to(C.DEFAULTS_TRAIN.device)(
                g_dev,
                {nt: g_dev.nodes[nt].data["h"] for nt in C.NODE_TYPES})
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        iso = IsotonicRegression(out_of_bounds="clip",
                                  y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(probs[val_mask], y[val_mask])
        p_oof[te] = iso.transform(probs[te])
        log(f"  fold AUROC (softmax head, test slice only) "
            f"= {roc_auc_score(y[te], probs[te]):.4f}")

    # Pooled OOF metrics
    au = roc_auc_score(y, p_oof)
    ap = average_precision_score(y, p_oof)
    au_ci = _bootstrap_ci(y, p_oof, roc_auc_score, n_boot=2000, seed=0)
    ap_ci = _bootstrap_ci(y, p_oof, average_precision_score, n_boot=2000, seed=1)
    br = brier_score_loss(y, p_oof)
    best_f1 = best_pr = best_rc = 0
    for t in np.linspace(0.02, 0.35, 80):
        yh = (p_oof >= t).astype(int)
        if yh.sum() < 5: continue
        f = f1_score(y, yh)
        if f > best_f1:
            best_f1 = f; best_pr = precision_score(y, yh); best_rc = recall_score(y, yh)

    log("\n=== MedHG-PS ORIGINAL METHOD (A3 unit graph, no downstream ensemble) ===")
    log(f"AUROC {au:.3f} ({au_ci[0]:.3f}-{au_ci[1]:.3f})")
    log(f"AUPRC {ap:.3f} ({ap_ci[0]:.3f}-{ap_ci[1]:.3f})")
    log(f"F1 {best_f1:.3f} P {best_pr:.3f} R {best_rc:.3f}  Brier {br:.3f}")

    pd.DataFrame([dict(model="MedHG-PS (original method, A3 care-unit, softmax head)",
                        auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                        auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1],
                        brier=br, f1=best_f1,
                        precision=best_pr, recall=best_rc)]).to_csv(OUT_RES, index=False)
    np.savez(OUT_OOF, y=y, p_oof=p_oof)
    log(f"saved {OUT_RES}, {OUT_OOF}")


if __name__ == "__main__":
    main()
