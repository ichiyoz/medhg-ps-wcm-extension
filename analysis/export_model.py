"""Export the best CV-validated readmission model for deployment.

  1. Assemble the enriched feature frame (NSQIP + CPT + post-op sequence).
  2. 5-fold CV both tree estimators (RF, HGB), raw and isotonic-calibrated, to
     report honest deployment performance (AUROC / AUPRC / Brier) and to pick.
  3. Refit the winner on 100% of the cohort, wrap in an isotonic calibrator.
  4. Serialize a self-contained medhg_ps.deploy.ReadmissionModel to
     artifacts/readmission_model.joblib and write a model card.

   PYTHONPATH=. python analysis/export_model.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import fit_preprocess, apply_preprocess
from medhg_ps.deploy import (MIN_SUP, ReadmissionModel, _make_estimator,
                             assemble_training_frame, fit_full)

K, SEED = 5, 42
OUT_DIR = Path("artifacts"); OUT_DIR.mkdir(exist_ok=True)


def cv_eval(merged, feat_cols, cpt_arr, Fseq, seq_all, y, name, calibrate):
    """Fold-honest CV of one estimator; returns dict of mean metrics."""
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    N = len(merged)
    au, ap, br = [], [], []
    for tr, te in skf.split(np.zeros(N), y):
        Xtab_tr, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
        Xtab = apply_preprocess(merged[feat_cols], st)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
        Xcpt = ohe.transform(cpt_arr)
        keep = [c for c in seq_all if (Fseq.iloc[tr][c] > 0).sum() >= MIN_SUP]
        Xseq = Fseq[keep].values.astype(float)
        Xall = np.hstack([Xtab, Xcpt, Xseq])
        sc = StandardScaler().fit(Xall[tr]); Xall = sc.transform(Xall)
        est = _make_estimator(name)
        if calibrate:
            est = CalibratedClassifierCV(est, method="isotonic", cv=3)
        est.fit(Xall[tr], y[tr])
        p = est.predict_proba(Xall[te])[:, 1]
        au.append(roc_auc_score(y[te], p)); ap.append(average_precision_score(y[te], p))
        br.append(brier_score_loss(y[te], p))
    return dict(auroc=float(np.mean(au)), auroc_sd=float(np.std(au)),
                auprc=float(np.mean(ap)), brier=float(np.mean(br)))


def main():
    print("[export] assembling enriched training frame...", flush=True)
    merged, feat_cols, cpt_arr, Fseq, seq_all, y = assemble_training_frame()
    N = len(merged)
    print(f"[export] cohort={N:,}  base={y.mean()*100:.2f}%  "
          f"tabular={len(feat_cols)} CPT=1 seq_candidates={len(seq_all)}", flush=True)

    print("[export] 5-fold CV (honest deployment estimate)...", flush=True)
    report = {}
    for name in ["hgb", "rf"]:
        for cal in [False, True]:
            tag = f"{name}{'_cal' if cal else ''}"
            report[tag] = cv_eval(merged, feat_cols, cpt_arr, Fseq, seq_all, y, name, cal)
            r = report[tag]
            print(f"  {tag:8s} AUROC {r['auroc']:.3f}±{r['auroc_sd']:.3f}  "
                  f"AUPRC {r['auprc']:.3f}  Brier {r['brier']:.4f}", flush=True)

    # pick on the CALIBRATED metrics (what actually deploys): AUPRC first
    # (emphasis at 9.6% prevalence), AUROC as tiebreak.
    best_name = max(["hgb", "rf"],
                    key=lambda n: (report[f"{n}_cal"]["auprc"], report[f"{n}_cal"]["auroc"]))
    chosen = report[f"{best_name}_cal"]
    print(f"[export] chosen estimator: {best_name} (isotonic-calibrated)", flush=True)

    print("[export] refitting on 100% of cohort + calibrating...", flush=True)
    model = fit_full(merged, feat_cols, cpt_arr, Fseq, seq_all, y, best_name)
    # wrap a calibrated classifier fit on all data (internal 5-fold isotonic)
    # rebuild the full design matrix once to fit the calibrator
    Xtab, st = fit_preprocess(merged[feat_cols].reset_index(drop=True), id_cols=[])
    Xcpt = model.cpt_encoder.transform(cpt_arr)
    Xseq = Fseq[model.seq_keep].values.astype(float)
    Xall = model.scaler.transform(np.hstack([Xtab, Xcpt, Xseq]))
    cal = CalibratedClassifierCV(_make_estimator(best_name), method="isotonic", cv=5)
    cal.fit(Xall, y)
    model.clf = cal
    model.cv_metrics = chosen
    model.version = "1.0"

    # sanity: self-prediction runs end to end on a few records
    sample = []
    for i in range(3):
        rec = {c: merged[c].iloc[i] for c in feat_cols}
        rec["PrimaryCPT"] = cpt_arr[i, 0]
        rec["postop_units"] = []
        sample.append(rec)
    probs = model.predict_proba(sample)
    print(f"[export] self-test predict_proba (3 rows): {np.round(probs, 4)}", flush=True)

    out = OUT_DIR / "readmission_model.joblib"
    joblib.dump(model, out)
    meta = dict(version=model.version, estimator=best_name, calibrated=True,
                base_rate=model.base_rate, n_features=model.n_features,
                n_seq_features=len(model.seq_keep), cohort_n=int(N),
                cv=report, chosen_cv=chosen)
    (OUT_DIR / "model_card.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"[export] wrote {OUT_DIR/'model_card.json'}", flush=True)


if __name__ == "__main__":
    main()
