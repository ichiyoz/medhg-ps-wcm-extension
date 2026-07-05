# medhg-ps-wcm-extension

**WCM extension of MedHG-PS: 30-day unplanned readmission prediction after minimally invasive general surgery.**

This repository contains the analysis pipeline for the manuscript *"MEDHG-PS validation and extension at Weill Cornell Medicine"*. It extends the original MedHG-PS heterogeneous-graph framework (Liu et al. 2023) with a corrected NSQIP-anchored cohort, an enriched tabular feature stack, a flow-first sequence model, several graph-augmented tree architectures, outcome-driven phenotypes, and a deployment-sensitivity analysis.

## Cohort

- WCM MIS general-surgery encounters 2022–2026
- N = 13,858 after SQL corrections + standard readmission-modeling exclusions
  (Expired, Hospice, Acute-transfer, Left Against Medical Advice)
- Outcome: 30-day unplanned readmission, NSQIP-anchored gold label
- Base rate: 7.61% · no-skill AUPRC: 0.076

## Models included

Tabular / deployable
- `analysis/final_push_080.py` — enriched tabular feature-stack pipeline
- `analysis/tabular_leaders_v10.py` — LACE, HOSPITAL, rf_clin, rf_big enriched, LightGBM tuned, XGBoost tuned, Ensemble
- `analysis/tune_winning_learners.py` — nested-CV Bayesian Optuna tuning of RF/LGBM/XGB
- `analysis/lace_baseline.py`, `analysis/hospital_score.py` — clinical-score baselines

Flow-first (sequence)
- `analysis/final_push_080_v9b.py` — multi-attribute flow GRU on order stream + graph context + lean tabular
- `analysis/deployment_sensitivity_flow_v13b.py` — perturbation robustness for the Flow model

Graph-based
- `analysis/final_push_080_v2.py`, `v3.py` — MedHG-PS ie-HGCN with orders-as-A3 substitution
- `analysis/final_push_080_v5.py` — clock-time temporal-edge graph with bundle awareness
- `analysis/final_push_080_v6.py`, `v7.py` — hand-crafted graph features + Node2Vec walks
- `analysis/final_push_080_v8.py` — rich-node design (5 provider roles + 5 order categories + 17 Charlson diagnoses + top-200 order sets)
- `analysis/graph_aware_forest_v12.py` — custom Random Forest with genuine graph-query split rules
- `analysis/graph_models_v10.py` — head-to-head evaluation of the graph family on the corrected cohort

Phenotypes
- `analysis/phenotypes_v14.py` — unified hospital-operations graph + DICE
- `analysis/phenotypes_v15.py` — Flow-GRU + ie-HGCN substrates + DICE (final Table 5 pipeline)
- `analysis/dice.py` — tabular Deep Significance Clustering implementation (SMD + IRLS-LR constraints)
- `analysis/dice_smd_3substrates.py` — 3-substrate DICE comparison

Deployment sensitivity
- `analysis/deployment_sensitivity_v13.py` — realistic MISSING and WRONG failure scenarios for the Ensemble
- `analysis/deployment_sensitivity_flow_v13b.py` — same protocol for the Flow model

Node2Vec (with proper biased walks)
- `analysis/node2vec_v11.py` — biased Node2Vec (p, q parameters) + Optuna tuning

## Headline results (5-fold CV, isotonic-calibrated pooled OOF, bootstrap n=2000)

| Model | AUROC (95% CI) | AUPRC (95% CI) | Note |
|---|---|---|---|
| **Ensemble (RF + LGBM + XGB, tuned)** | **0.746 (0.730–0.762)** | **0.229 (0.206–0.256)** | Deployable candidate |
| XGBoost (tuned) | 0.742 (0.727–0.758) | 0.223 (0.201–0.249) | |
| Flow model (v9b) | 0.733 (0.718–0.748) | 0.219 (0.197–0.244) | Lean 28-col tabular + GRU |
| RF enriched (untuned) | 0.742 (0.727–0.758) | 0.216 (0.194–0.243) | |
| LightGBM (tuned) | 0.740 (0.724–0.756) | 0.215 (0.194–0.241) | |
| MedHG-PS (orders as A3) | 0.727 (0.711–0.743) | 0.197 (0.177–0.221) | ie-HGCN + tree ensemble |
| rf_clin (clinical features only) | 0.721 (0.706–0.737) | 0.171 (0.155–0.190) | Baseline reference |
| LACE score (van Walraven 2010) | 0.702 (0.685–0.717) | 0.142 (0.130–0.155) | Interpretable clinical baseline |
| HOSPITAL score (Donzé 2013) | 0.642 (0.626–0.658) | 0.115 (0.106–0.125) | Interpretable clinical baseline |

## Repository layout

```
analysis/     - experimental / paper-final analysis scripts
medhg_ps/     - shared package (data loading, preprocessing, model wrappers, evaluation)
sql/          - Clarity-side SQL extracts and helper queries
requirements.txt - Python dependencies
```

## Not included

- **Data files** (patient EHR extracts, NSQIP linkage tables, cluster assignments, model artifacts) are excluded due to PHI/data-use constraints.
- **Deployment artifacts** (trained model pickles, calibration transforms) are excluded for the same reason.

The scripts are designed to run against a local `~/Downloads/medhg_ps_data/` directory containing the corresponding parquet / CSV extracts. See `sql/` for the extraction queries.

## Reproducibility notes

- 5-fold StratifiedKFold with `random_state=42` throughout
- Isotonic-calibrated pooled OOF for all reported metrics
- Bootstrap 95% CIs with `n=2000` per metric
- Nested CV for hyperparameter tuning (outer 5-fold × inner 3-fold Optuna TPE)
- DICE consensus phenotypes: 8 independent seeds with per-patient majority vote

## Citation

If you use this pipeline, please cite the manuscript once available and the original MedHG-PS paper (Liu et al. 2023).

## License

Code released under the MIT License. Data are not included.
