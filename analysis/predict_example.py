"""Minimal inference example for the exported readmission model.

   PYTHONPATH=. python analysis/predict_example.py

Loads artifacts/readmission_model.joblib and scores a couple of hand-built
records. In production you would pass rows produced by the same feature ETL
(sql/Bulk_Features_From_Cohort.sql) plus the PrimaryCPT code and the ordered
post-op care-unit buckets observed up to the prediction time (pre-discharge).
"""
import joblib

MODEL_PATH = "artifacts/readmission_model.joblib"


def main():
    model = joblib.load(MODEL_PATH)
    print(f"model v{model.version}  estimator={model.estimator_name}  "
          f"n_features={model.n_features}  base_rate={model.base_rate:.4f}")
    print(f"CV: AUROC={model.cv_metrics.get('auroc'):.3f}  "
          f"AUPRC={model.cv_metrics.get('auprc'):.3f}  "
          f"Brier={model.cv_metrics.get('brier'):.4f}")

    # A record = NSQIP feature fields (same names as the training schema) +
    # PrimaryCPT + postop_units. Any missing tabular field is imputed to the
    # train mean / "Unknown", so a partial dict still scores.
    records = [
        {  # higher-acuity example
            "Age": 71, "ASA Class": "4", "Functional Status": "Independent",
            "Diabetes Mellitus": "Yes", "Hypertension requiring medication": "Yes",
            "Heart Failure": "Yes", "Disseminated Cancer": "No",
            "PrimaryCPT": "44143",
            "postop_units": ["OR", "Intensive", "Acute"],
        },
        {  # lower-acuity example
            "Age": 38, "ASA Class": "2", "Functional Status": "Independent",
            "Diabetes Mellitus": "No", "Hypertension requiring medication": "No",
            "PrimaryCPT": "43775",
            "postop_units": ["OR", "Acute"],
        },
    ]
    probs = model.predict_proba(records)
    flags = model.predict(records)  # uses model.threshold
    for i, (p, f) in enumerate(zip(probs, flags)):
        print(f"  record {i}: P(readmit)={p:.3f}  flag@{model.threshold:.2f}={f}")


if __name__ == "__main__":
    main()
