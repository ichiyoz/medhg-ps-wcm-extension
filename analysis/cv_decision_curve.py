"""Decision curve analysis for the deployable readmission models.

AUROC/AUPRC say how well a model ranks; they do not say whether acting on it
helps. Decision curve analysis (Vickers & Elkin 2006) closes that gap: it plots
net benefit against the threshold probability p_t at which a clinician would act,
on the same scale as the treat-all and treat-none defaults.

We score the two DEPLOYABLE tree models from Table 2 on identical fold-honest
5-fold splits, collect each encounter's out-of-fold CALIBRATED probability
(isotonic, fit on train only -- DCA is meaningful only on calibrated risks),
and build a cross-validated net-benefit curve for each:

  gbt_clin_seq : HGB on clinical features (NSQIP + pre-op trajectory + calendar
                 + PrimaryCPT) PLUS the post-op care-unit sequence  [deployed]
  gbt_clin     : HGB on clinical features only (care-path sequence removed)

The contrast isolates the clinical value of the post-op care path -- the signal
that raised AUPRC (0.235 vs 0.225) but not AUROC (0.703 vs 0.703) in Table 2.
Treat-all and treat-none are drawn as references.

Writes, to artifacts/:
  - decision_curve.csv   net benefit per model per threshold (+ treat-all/none)
  - decision_curve.png   the curves over the plausible threshold range
  - decision_curve.json  summary: NB at key thresholds, sanity AUROC/AUPRC/Brier

Prints only aggregate numbers; no record-level data is emitted.

    PYTHONPATH=. python analysis/cv_decision_curve.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import apply_preprocess, fit_preprocess
from medhg_ps.deploy import MIN_SUP, _make_estimator, assemble_training_frame
from medhg_ps.evaluate import decision_curve

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED = (2 if SMOKE else 5), 42
EST = "hgb"                                  # the Table-2 winner (calibrated)
# Clinically plausible action thresholds for 30-day readmission. Base rate is
# ~9.6%, so the informative range sits low; treat-all goes negative past it.
THRESHOLDS = np.round(np.arange(0.02, 0.31, 0.01), 2)
KEY_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
OUT_DIR = Path("artifacts"); OUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "gbt_clin_seq": True,    # include post-op care-unit sequence block
    "gbt_clin":     False,   # clinical features only
}


def _design(merged, feat_cols, cpt_arr, Fseq, seq_all, tr, with_seq):
    """Fold-honest design matrix; all transforms fit on `tr` rows only."""
    Xtab_tr, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    Xcpt = ohe.transform(cpt_arr)
    blocks = [Xtab, Xcpt]
    if with_seq:
        keep = [c for c in seq_all if (Fseq.iloc[tr][c] > 0).sum() >= MIN_SUP]
        blocks.append(Fseq[keep].values.astype(float))
    Xall = np.hstack(blocks)
    sc = StandardScaler().fit(Xall[tr])
    return sc.transform(Xall)


def main() -> None:
    print(f"[dca] {'SMOKE ' if SMOKE else ''}assembling enriched frame...", flush=True)
    merged, feat_cols, cpt_arr, Fseq, seq_all, y = assemble_training_frame()
    N = len(merged)

    if SMOKE:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(N, size=min(4000, N), replace=False)
        merged = merged.iloc[idx].reset_index(drop=True)
        cpt_arr = cpt_arr[idx]; Fseq = Fseq.iloc[idx].reset_index(drop=True)
        y = y[idx]; N = len(merged)

    print(f"[dca] cohort={N:,}  base rate={y.mean()*100:.2f}%  "
          f"estimator={EST} (isotonic-calibrated)  folds={K}", flush=True)

    # out-of-fold calibrated probabilities: every encounter scored once, on the
    # fold where it is held out -> a single cross-validated risk per patient.
    oof = {m: np.full(N, np.nan) for m in MODELS}
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        for m, with_seq in MODELS.items():
            X = _design(merged, feat_cols, cpt_arr, Fseq, seq_all, tr, with_seq)
            est = CalibratedClassifierCV(_make_estimator(EST), method="isotonic", cv=3)
            est.fit(X[tr], y[tr])
            oof[m][te] = est.predict_proba(X[te])[:, 1]
        print(f"[dca] fold {fold + 1}/{K} scored", flush=True)

    assert all(not np.isnan(oof[m]).any() for m in MODELS), "unscored rows remain"

    # sanity: out-of-fold discrimination/calibration should reproduce Table 2.
    sanity = {}
    for m in MODELS:
        sanity[m] = dict(
            auroc=float(roc_auc_score(y, oof[m])),
            auprc=float(average_precision_score(y, oof[m])),
            brier=float(brier_score_loss(y, oof[m])),
        )
        s = sanity[m]
        print(f"[dca] {m:13s} OOF  AUROC {s['auroc']:.3f}  "
              f"AUPRC {s['auprc']:.3f}  Brier {s['brier']:.4f}", flush=True)

    # cross-validated net-benefit curves on the pooled OOF risks
    curves = {m: decision_curve(y, oof[m], THRESHOLDS) for m in MODELS}
    ref = curves["gbt_clin_seq"]            # treat-all/none identical across models

    # ---- CSV -------------------------------------------------------------
    csv = OUT_DIR / "decision_curve.csv"
    cols = ["threshold", "treat_all", "treat_none"] + list(MODELS)
    lines = [",".join(cols)]
    for i, t in enumerate(THRESHOLDS):
        row = [f"{t:.2f}", f"{ref.treat_all[i]:.6f}", f"{ref.treat_none[i]:.6f}"]
        row += [f"{curves[m].net_benefit[i]:.6f}" for m in MODELS]
        lines.append(",".join(row))
    csv.write_text("\n".join(lines) + "\n")

    # ---- summary at key thresholds + JSON --------------------------------
    summary = {"cohort_n": int(N), "prevalence": float(y.mean()),
               "estimator": EST, "folds": int(K), "sanity": sanity,
               "net_benefit_at": {}}
    print("\n=== Net benefit at key action thresholds ===")
    print(f"  {'p_t':>5s} {'treat_all':>10s} "
          + " ".join(f"{m:>14s}" for m in MODELS)
          + f" {'seq-vs-clin':>12s}")
    for t in KEY_THRESHOLDS:
        j = int(np.argmin(np.abs(THRESHOLDS - t)))
        nb = {m: float(curves[m].net_benefit[j]) for m in MODELS}
        delta = nb["gbt_clin_seq"] - nb["gbt_clin"]
        summary["net_benefit_at"][f"{t:.2f}"] = {
            "treat_all": float(ref.treat_all[j]), **nb,
            "seq_minus_clin": delta,
        }
        print(f"  {t:>5.2f} {ref.treat_all[j]:>10.4f} "
              + " ".join(f"{nb[m]:>14.4f}" for m in MODELS)
              + f" {delta:>+12.4f}")
    (OUT_DIR / "decision_curve.json").write_text(json.dumps(summary, indent=2))

    # ---- plot ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(THRESHOLDS, ref.treat_none, color="0.6", lw=1, ls=":", label="Treat none")
        ax.plot(THRESHOLDS, ref.treat_all, color="0.4", lw=1, ls="--", label="Treat all")
        styles = {"gbt_clin_seq": ("#1f77b4", "-", "GBT: clinical + care-path"),
                  "gbt_clin":     ("#d62728", "-.", "GBT: clinical only")}
        for m in MODELS:
            c, ls, lab = styles[m]
            ax.plot(THRESHOLDS, curves[m].net_benefit, color=c, ls=ls, lw=1.8, label=lab)
        ax.axvline(y.mean(), color="0.8", lw=1, zorder=0)
        ax.set_xlabel("Threshold probability $p_t$")
        ax.set_ylabel("Net benefit")
        ax.set_xlim(THRESHOLDS[0], THRESHOLDS[-1])
        lo = min(curves[m].net_benefit.min() for m in MODELS)
        ax.set_ylim(min(-0.01, lo * 1.1), float(y.mean()) * 1.05)
        ax.set_title(f"Decision curve, 30-day readmission (5-fold CV, n={N:,})")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        png = OUT_DIR / "decision_curve.png"
        fig.savefig(png, dpi=150)
        print(f"\n[dca] wrote {png}", flush=True)
    except Exception as e:                       # plotting is non-essential
        print(f"[dca] plot skipped: {type(e).__name__}: {e}", flush=True)

    print(f"[dca] wrote {csv}", flush=True)
    print(f"[dca] wrote {OUT_DIR / 'decision_curve.json'}", flush=True)


if __name__ == "__main__":
    main()
