"""
End-to-end orchestrator for MedHG-PS training and embedding extraction.

Sequence:

    1. Load A1..A5 + tabular encounter features + 30-day readmission labels.
    2. Stratified 8 : 1 : 1 split on the encounter LogIDs.
    3. Fit preprocessing on the train rows, apply to val/test, build
       provider and unit feature matrices.
    4. Build the DGL hetero graph.
    5. Run hyperparameter search (TPE) over Table 2 ranges; retrain
       the best configuration on train + val.
    6. Evaluate on the held-out test split (6-metric panel + bootstrap
       CIs).
    7. Extract H^PROV_3, H^UNIT_3, H^ENC_3 -> parquet for the deployment
       script to consume.

Run as:
    python -m medhg_ps.run
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from . import config as C
from .data import (
    add_calendar_features,
    apply_preprocess,
    build_preop_trajectory_features,
    build_provider_features,
    build_unit_features,
    fit_preprocess,
    load_raw,
    save_preprocess,
    stratified_split,
)
from .evaluate import evaluate, pick_threshold
from .extract_embeddings import extract_embeddings, save_embeddings
from .graph import build_graph, save_graph
from .hp_search import run_hp_search
from .train import set_seed, train_model


def main(args: argparse.Namespace) -> None:
    set_seed(C.DEFAULTS_TRAIN.split_seed)   # global determinism for the run
    artifact_dir = Path(args.artifact_dir)
    embed_dir    = Path(args.embed_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    print("[1/7] Loading SQL extracts + tabular features + labels...")
    raw = load_raw()

    # -----------------------------------------------------------------
    # CRITICAL ORDERING: build merged_all ONCE, then compute the split
    # against merged_all's row order. We previously called stratified_split
    # twice (once on encounters + labels, once on encounters + features +
    # labels) which gave two DIFFERENT partitions and let some
    # preprocess-fit rows leak into the eval splits used downstream.
    print("[2/7] Building merged encounter frame + stratified 8:1:1 split...")
    # bulk_features_with_label.parquet carries its own SurgeryDate /
    # PAT_ID / EncounterCSN that duplicate A1's, which would otherwise
    # produce *_x / *_y suffixes and break downstream lookups of plain
    # `SurgeryDate`. A1 is authoritative for encounter metadata, so drop
    # the dupes from enc_features before merging. The LogID merge key
    # itself stays.
    #
    # Also strip the label column from enc_features so it comes solely
    # from the labels merge -- bulk_features and labels point at the
    # same file by default, so without this, the label gets merged
    # twice and becomes ReadmittedWithin30Days_x / _y.
    enc_features_no_dupes = raw.enc_features.drop(
        columns=([c for c in raw.encounters.columns
                  if c != "LogID" and c in raw.enc_features.columns]
                 + ["ReadmittedWithin30Days"]),
        errors="ignore",
    )
    merged_all = (raw.encounters
                  .merge(enc_features_no_dupes, on="LogID", how="inner")
                  .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]],
                         on="LogID", how="inner")
                  .reset_index(drop=True))   # fix ordering so masks align

    # Augment encounter features with the paper's H^ENC "transfer
    # history" (pre-op care-unit trajectory from A3, split at surgery
    # start) + calendar block (from the surgery timestamp). These are
    # the columns derived from the A1-A5 graph extracts that the NSQIP
    # tabular block does not carry.
    ss = merged_all[["LogID"]].copy()
    ss["_ss"] = pd.to_datetime(merged_all.get("Procedure/Surgery Start"),
                               errors="coerce")
    traj = build_preop_trajectory_features(raw.enc_unit_edges, ss)
    merged_all = merged_all.merge(traj, on="LogID", how="left")
    for c in C.TRAJECTORY_FEATURE_COLUMNS:        # encounters w/ no A3 visit
        merged_all[c] = merged_all[c].fillna(0)
    merged_all = add_calendar_features(merged_all)

    train_mask, val_mask, test_mask = stratified_split(merged_all)

    # -----------------------------------------------------------------
    print("[3/7] Preprocessing encounter / provider / unit features...")
    # Restrict to the explicit 40-feature pre-operative allow-list
    # (config.MODEL_FEATURE_COLUMNS). The raw Surgery_RVA export carries
    # ~480 columns including the label source ('Case'), unplanned-
    # readmission counts, and the full post-operative outcome block --
    # feeding those leaks the target (val AUROC -> ~1.0). The allow-list
    # is the set the model is actually meant to see at the discharge
    # prediction point.
    nsqip_cols = [c for c in C.MODEL_FEATURE_COLUMNS
                  if c in merged_all.columns]
    derived_cols = [c for c in C.ENCOUNTER_DERIVED_COLUMNS
                    if c in merged_all.columns]
    feat_cols = nsqip_cols + derived_cols
    missing = [c for c in C.MODEL_FEATURE_COLUMNS
               if c not in merged_all.columns]
    if missing:
        print(f"      WARNING: {len(missing)} allow-list features absent "
              f"from the merged frame: {missing}")
    print(f"      using {len(nsqip_cols)}/{len(set(C.MODEL_FEATURE_COLUMNS))} "
          f"NSQIP pre-op features + {len(derived_cols)} A1-A5-derived "
          f"(pre-op trajectory + calendar); leakage columns excluded")
    feat_all = merged_all[feat_cols].copy()
    # Fit ONLY on the training subset of feat_all to avoid leakage.
    X_train_only, enc_state = fit_preprocess(
        feat_all.loc[train_mask].reset_index(drop=True), id_cols=[]
    )
    # Apply the train-fit transform to ALL rows so the graph carries
    # features for every node; val/test masks select held-out rows.
    X_enc = apply_preprocess(feat_all, enc_state)

    save_preprocess(enc_state, artifact_dir / "encounter_preprocess.pkl")

    prov_ids, X_prov, prov_state = build_provider_features(raw.prov_attrs)
    unit_ids, X_unit, unit_state = build_unit_features(raw.unit_attrs)
    save_preprocess(prov_state, artifact_dir / "provider_preprocess.pkl")
    save_preprocess(unit_state, artifact_dir / "unit_preprocess.pkl")

    # -----------------------------------------------------------------
    print("[4/7] Building DGL heterograph...")
    artifacts = build_graph(
        raw=raw,
        encounters_merged=merged_all,
        enc_features=X_enc,
        prov_ids=prov_ids, prov_features=X_prov,
        unit_ids=unit_ids, unit_features=X_unit,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )
    save_graph(artifacts, artifact_dir / "graph.bin")
    print(f"      nodes: ENC={artifacts.g.num_nodes(C.ENC_NTYPE)}, "
          f"PROV={artifacts.g.num_nodes(C.PROV_NTYPE)}, "
          f"UNIT={artifacts.g.num_nodes(C.UNIT_NTYPE)}")
    print(f"      edges: ENC-PROV={artifacts.g.num_edges(C.ETYPE_ENC_PROV)}, "
          f"ENC-UNIT={artifacts.g.num_edges(C.ETYPE_ENC_UNIT)}")

    # -----------------------------------------------------------------
    if args.skip_hp_search:
        print("[5/7] Training final model with default hyperparameters...")
        model, train_result = train_model(
            artifacts, cfg=C.DEFAULTS_TRAIN,
            save_dir=artifact_dir, verbose=True,
        )
        best_cfg = C.DEFAULTS_TRAIN
    else:
        print("[5/7] TPE hyperparameter search...")
        best_cfg, train_result, model = run_hp_search(
            artifacts, save_dir=artifact_dir, verbose=True,
        )

    # -----------------------------------------------------------------
    print("[6/7] Evaluating on held-out test split...")
    model.eval()
    g_dev = artifacts.g.to(best_cfg.device)
    with torch.no_grad():
        logits, _ = model.to(best_cfg.device)(
            g_dev,
            {nt: g_dev.nodes[nt].data["h"] for nt in C.NODE_TYPES},
        )
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    y = g_dev.nodes[C.ENC_NTYPE].data["y"].cpu().numpy()
    val_mask_np  = g_dev.nodes[C.ENC_NTYPE].data["val_mask"].cpu().numpy()
    test_mask_np = g_dev.nodes[C.ENC_NTYPE].data["test_mask"].cpu().numpy()

    # Pick the F1-maximising threshold on validation, then freeze it for test.
    op_thr = pick_threshold(y[val_mask_np], probs[val_mask_np])
    test_metrics = evaluate(y[test_mask_np], probs[test_mask_np],
                            threshold=op_thr, n_boot=args.bootstrap)
    print(f"      test AUROC = {test_metrics.auroc:.4f}  "
          f"(95% CI {test_metrics.auroc_ci[0]:.3f}, {test_metrics.auroc_ci[1]:.3f})")
    print(f"      test AUPRC = {test_metrics.auprc:.4f}  "
          f"(95% CI {test_metrics.auprc_ci[0]:.3f}, {test_metrics.auprc_ci[1]:.3f})")
    print(f"      test F1    = {test_metrics.f1:.4f}  "
          f"prec={test_metrics.precision:.3f}  "
          f"rec={test_metrics.recall:.3f}  "
          f"spec={test_metrics.specificity:.3f}")

    with open(artifact_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics.to_dict(), f, indent=2)
    with open(artifact_dir / "best_config.json", "w") as f:
        json.dump(asdict(best_cfg), f, indent=2)

    # -----------------------------------------------------------------
    print("[7/7] Extracting embeddings for downstream RF integration...")
    tables = extract_embeddings(
        model, artifacts,
        raw_prov_attrs=raw.prov_attrs,
        raw_unit_attrs=raw.unit_attrs,
        device=best_cfg.device,
    )
    save_embeddings(tables, embed_dir)
    print(f"      saved provider / unit / encounter embeddings to {embed_dir}")

    print("\nDONE.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", default=str(C.ARTIFACT_DIR))
    p.add_argument("--embed-dir",    default=str(C.EMBED_DIR))
    p.add_argument("--skip-hp-search", action="store_true",
                   help="Skip TPE and use TrainConfig defaults instead.")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap samples for AUROC/AUPRC CIs.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
