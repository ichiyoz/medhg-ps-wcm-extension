"""Wrapper: run an analysis script with HistGradientBoosting transparently
REPLACED by the tuned RandomForest (RF is the better tabular learner; drop HGB).
Patches sklearn.ensemble.HistGradientBoostingClassifier BEFORE the target script
imports it, so every `from sklearn.ensemble import HistGradientBoostingClassifier`
picks up the RF. HGB-specific constructor kwargs are ignored; random_state kept.
Usage: python analysis/run_rf.py <script.py>
"""
import sys, runpy
import sklearn.ensemble as _ens
from sklearn.ensemble import RandomForestClassifier as _RF

# Canonical RF (matches the repo's native rf_tab / the manuscript's "500-tree RF";
# ~0.714-0.715 AUROC). The Optuna "tuned" leaf=3 config was protocol-overfit and
# generalized worse, so we use the robust, established config.
_RF_HP = dict(n_estimators=500, min_samples_leaf=10, max_features="sqrt",
              class_weight="balanced", n_jobs=-1)

def HistGradientBoostingClassifier(*args, random_state=42, **kwargs):
    """Factory returning a genuine tuned RandomForestClassifier (so sklearn
    clone/calibration work); HGB-specific kwargs are ignored."""
    return _RF(random_state=random_state, **_RF_HP)

_ens.HistGradientBoostingClassifier = HistGradientBoostingClassifier
print(f"[rf] HistGradientBoostingClassifier -> RandomForest "
      f"({_RF_HP['n_estimators']} trees, min_leaf {_RF_HP['min_samples_leaf']}, "
      f"class_weight {_RF_HP['class_weight']})", flush=True)
runpy.run_path(sys.argv[1], run_name="__main__")
