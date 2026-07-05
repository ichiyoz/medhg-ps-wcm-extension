"""Alert triggering: overall risk threshold OR clinically-meaningful single
feature, tracked separately.

A deployed readmission alert usually fires on one number -- the model's
overall risk crossing a threshold p_t. But a clinician may also want to act on
an individual indicator that is clinically alarming on its own (e.g. albumin
2.5 g/dL, ASA IV, a post-operative ICU stay) even when the aggregate risk is
unremarkable. This script implements that two-path alert and, per the design
request, tracks and measures the two paths SEPARATELY:

    risk_flag    : calibrated overall risk >= p_t            (model path)
    feat_flag    : any clinically-meaningful red flag present (rule path)
    union_flag   : risk_flag OR feat_flag                    (deployed alert)
    feature-only : feat_flag AND NOT risk_flag  (what the rule path ADDS)

For each action threshold we report:
  - how many encounters each path flags, and their overlap;
  - the readmission YIELD (observed rate / PPV) of the feature-only additions
    -- i.e. are the rule-triggered, low-model-risk patients actually higher
    risk? -- the test of whether the individual-feature path is meaningful;
  - the net benefit (Vickers & Elkin) of risk-only vs the union, so the
    clinical value of the rule path is on the same decision-analytic scale as
    the decision curve.
Plus a per-flag breakdown: prevalence, readmission rate, and how many each
flag catches that the model alone misses at a reference threshold.

Overall risk = out-of-fold isotonic-calibrated probability from the deployable
gradient-boosted tree (clinical features + post-op care path), the Table-2
model. Red flags use fixed clinical cutoffs, NOT learned thresholds.

Prints aggregate numbers only. Writes artifacts/alert_triggers.{csv,json}.

    PYTHONPATH=. python analysis/cv_alert_triggers.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import apply_preprocess, fit_preprocess
from medhg_ps.deploy import MIN_SUP, _make_estimator, assemble_training_frame
from medhg_ps.evaluate import net_benefit

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED = (2 if SMOKE else 5), 42
THRESHOLDS = np.round(np.arange(0.02, 0.31, 0.01), 2)
KEY_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
REF_T = 0.10                                   # reference operating point for per-flag table
OUT_DIR = Path("artifacts"); OUT_DIR.mkdir(exist_ok=True)

_NEG = {"no", "0", "nan", "none", "", "0.0"}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pos(s: pd.Series) -> pd.Series:
    """A yes/no comorbidity flag is positive unless it is a negative token."""
    return ~s.astype(str).str.strip().str.lower().isin(_NEG)


def build_red_flags(m: pd.DataFrame, Fseq: pd.DataFrame) -> "dict[str, np.ndarray]":
    """Clinically-meaningful single-feature triggers (fixed cutoffs).

    Each value is a boolean array aligned to `m`. A missing lab is NOT a flag
    (fillna False): the rule path only fires on observed, alarming values.
    """
    f: dict[str, np.ndarray] = {}

    def lab(col, fn):
        return fn(_num(m[col])).fillna(False).to_numpy() if col in m.columns else np.zeros(len(m), bool)

    f["Albumin <3.0 g/dL"]        = lab("ALB",  lambda s: s < 3.0)
    f["Hematocrit <27%"]          = lab("HCT",  lambda s: s < 27.0)
    f["Creatinine >2.0 mg/dL"]    = lab("Creat", lambda s: s > 2.0)
    f["Sodium <130 or >150"]      = lab("NA",   lambda s: (s < 130) | (s > 150))
    f["INR >1.5"]                 = lab("INR",  lambda s: s > 1.5)
    f["WBC >15 x10^3/uL"]         = lab("WBC",  lambda s: s > 15.0)
    f["ASA class >=IV"]           = lab("ASAClass", lambda s: s >= 4.0)
    for col, name in [("Preop Dialysis", "Preoperative dialysis"),
                      ("Preop Acute Kidney Injury", "Preoperative AKI"),
                      ("Disseminated Cancer", "Disseminated cancer"),
                      ("Heart Failure", "Heart failure"),
                      ("Ventilator Dependent", "Ventilator dependent")]:
        f[name] = (_pos(m[col]).to_numpy() if col in m.columns else np.zeros(len(m), bool))
    f["Post-op ICU stay"] = (Fseq["has_ICU"].to_numpy().astype(bool)
                             if "has_ICU" in Fseq.columns else np.zeros(len(m), bool))
    return f


def oof_risk(merged, feat_cols, cpt_arr, Fseq, seq_all, y) -> np.ndarray:
    """Out-of-fold isotonic-calibrated risk from the deployable model
    (clinical features + post-op care path)."""
    N = len(merged)
    risk = np.full(N, np.nan)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        Xtab_tr, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
        Xtab = apply_preprocess(merged[feat_cols], st)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
        Xcpt = ohe.transform(cpt_arr)
        keep = [c for c in seq_all if (Fseq.iloc[tr][c] > 0).sum() >= MIN_SUP]
        Xseq = Fseq[keep].values.astype(float)
        Xall = StandardScaler().fit(np.hstack([Xtab, Xcpt, Xseq])[tr]).transform(
            np.hstack([Xtab, Xcpt, Xseq]))
        est = CalibratedClassifierCV(_make_estimator("hgb"), method="isotonic", cv=3)
        est.fit(Xall[tr], y[tr])
        risk[te] = est.predict_proba(Xall[te])[:, 1]
        print(f"[alert] fold {fold + 1}/{K} scored", flush=True)
    assert not np.isnan(risk).any()
    return risk


def main() -> None:
    print(f"[alert] {'SMOKE ' if SMOKE else ''}assembling frame...", flush=True)
    merged, feat_cols, cpt_arr, Fseq, seq_all, y = assemble_training_frame()
    if SMOKE:
        rng = np.random.default_rng(SEED); idx = rng.choice(len(merged), 4000, replace=False)
        merged = merged.iloc[idx].reset_index(drop=True); cpt_arr = cpt_arr[idx]
        Fseq = Fseq.iloc[idx].reset_index(drop=True); y = y[idx]
    N = len(merged); prev = float(y.mean())
    print(f"[alert] cohort={N:,}  base rate={prev*100:.2f}%", flush=True)

    risk = oof_risk(merged, feat_cols, cpt_arr, Fseq, seq_all, y)
    flags = build_red_flags(merged, Fseq)
    any_flag = np.zeros(N, bool)
    for v in flags.values():
        any_flag |= v

    # ---- per-flag breakdown at the reference operating point -------------
    print(f"\n=== Per-flag breakdown (readmit base rate {prev*100:.1f}%; "
          f"reference risk threshold p_t={REF_T:.2f}) ===")
    print(f"  {'flag':26s} {'n':>7s} {'%coh':>6s} {'readmit%':>9s} "
          f"{'miss-by-model':>13s} {'their readmit%':>15s}")
    low_risk = risk < REF_T
    per_flag = {}
    for name, v in flags.items():
        n = int(v.sum())
        rr = float(y[v].mean() * 100) if n else 0.0
        extra = v & low_risk                       # flagged by rule, missed by model at REF_T
        ne = int(extra.sum())
        rre = float(y[extra].mean() * 100) if ne else 0.0
        per_flag[name] = dict(n=n, pct_cohort=100 * n / N, readmit_pct=rr,
                              missed_by_model=ne, missed_readmit_pct=rre)
        print(f"  {name:26s} {n:>7,d} {100*n/N:>5.1f}% {rr:>8.1f}% "
              f"{ne:>13,d} {rre:>14.1f}%")

    nf = int(any_flag.sum())
    print(f"\n  {'ANY red flag':26s} {nf:>7,d} {100*nf/N:>5.1f}% "
          f"{float(y[any_flag].mean()*100):>8.1f}%")

    # ---- two-path tracking + net benefit across thresholds ---------------
    print(f"\n=== Risk path vs rule path vs union, by action threshold ===")
    print(f"  {'p_t':>5s} {'risk_n':>8s} {'union_n':>8s} {'feat_only':>10s} "
          f"{'fo_readmit%':>12s} {'NB_risk':>9s} {'NB_union':>9s} {'dNB':>8s}")
    rows = []
    for t in THRESHOLDS:
        risk_flag = risk >= t
        union_flag = risk_flag | any_flag
        feat_only = any_flag & ~risk_flag
        nb_risk = net_benefit(y, risk.astype(float), float(t))
        # union is a hard rule set; score it at the same p_t exchange rate
        tp = int(np.sum(union_flag & (y == 1))); fp = int(np.sum(union_flag & (y == 0)))
        w = t / (1 - t)
        nb_union = tp / N - (fp / N) * w
        fo_rr = float(y[feat_only].mean() * 100) if feat_only.sum() else 0.0
        rows.append(dict(threshold=float(t), risk_n=int(risk_flag.sum()),
                         union_n=int(union_flag.sum()), feat_only_n=int(feat_only.sum()),
                         feat_only_readmit_pct=fo_rr, nb_risk=nb_risk, nb_union=nb_union,
                         dnb=nb_union - nb_risk))
        if round(float(t), 2) in KEY_THRESHOLDS:
            print(f"  {t:>5.2f} {int(risk_flag.sum()):>8,d} {int(union_flag.sum()):>8,d} "
                  f"{int(feat_only.sum()):>10,d} {fo_rr:>11.1f}% {nb_risk:>9.4f} "
                  f"{nb_union:>9.4f} {nb_union-nb_risk:>+8.4f}")

    # ---- write artifacts -------------------------------------------------
    cols = ["threshold", "risk_n", "union_n", "feat_only_n",
            "feat_only_readmit_pct", "nb_risk", "nb_union", "dnb"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    (OUT_DIR / "alert_triggers.csv").write_text("\n".join(lines) + "\n")
    (OUT_DIR / "alert_triggers.json").write_text(json.dumps(
        dict(cohort_n=N, prevalence=prev, reference_threshold=REF_T,
             any_flag_n=nf, any_flag_readmit_pct=float(y[any_flag].mean() * 100),
             per_flag=per_flag, by_threshold=rows), indent=2))
    print(f"\n[alert] wrote {OUT_DIR/'alert_triggers.csv'} and .json", flush=True)
    print("Note: feature-only = rule-flagged but below the model threshold; its "
          "readmit% vs base rate shows whether the rule path is clinically meaningful.")


if __name__ == "__main__":
    main()
