"""Final Table 2 with bootstrap 95% CIs + redone decision-curve analysis.

All five rows scored under one consistent protocol on the SQL-fixed clean
41-feature cohort:
    - 5-fold stratified CV (seed 42), pooled out-of-fold predictions.
    - Isotonic-calibrated (CalibratedClassifierCV method='isotonic', cv=3),
      each patient scored once on the held-out fold.
    - RandomForest is the single downstream learner (HGB dropped).
    - GRU + GNN embeddings are trained per fold on TRAIN rows only.

Models:
    1) rf_tuned_tab   per-fold-TPE-tuned RF on [Xtab] -- REPORTED BASELINE
    2) rf_clin        canonical RF on [Xtab]
    3) rf_seq         canonical RF on [Xtab + hand-crafted care path]
    4) rf_gru         canonical RF on [Xtab + GRU care-path encoder] -- DEPLOYABLE
    5) rf_gnn_emb     canonical RF on [Xtab + GNN encounter embedding]

Outputs (artifacts/newdata/):
    table2_final_dca.log            stdout tee
    table2_final_dca_results.csv    AUROC/AUPRC 95% CI + Brier per model
    table2_final_dca.png            5-model decision-curve plot
    table2_final_dca.csv/.json      per-threshold net benefit
    table2_final_dca_oof.npz        pooled OOF predictions (for reuse)

    PYTHONPATH=. python analysis/table2_final_dca.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import medhg_ps.config as C
from medhg_ps.data import (add_calendar_features, apply_preprocess,
                           build_preop_trajectory_features, build_provider_features,
                           build_unit_features, fit_preprocess, load_raw)
from medhg_ps.deploy import MIN_SUP, UNIT_BUCKET, _load_cpt_map, seq_feature_dict
from medhg_ps.evaluate import _bootstrap_ci, decision_curve
from medhg_ps.extract_embeddings import extract_embeddings
from medhg_ps.graph import build_graph
from medhg_ps.train import _resolve_device, set_seed, train_model

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
MAX_EP, PAT = (15, 4) if SMOKE else (120, 12)
N_TPE_TRIALS = 5 if SMOKE else 18
N_BOOT = 500 if SMOKE else 2000
THRESHOLDS = np.round(np.arange(0.02, 0.31, 0.01), 2)
KEY = [0.05, 0.10, 0.15, 0.20]
MAXLEN, EMB_DIM, HID, NUMF = 40, 16, 32, 4
UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]
U2I = {u: i for i, u in enumerate(UNITS)}
PAD = len(UNITS)
OUT_DIR = Path("artifacts/newdata")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "table2_final_dca.log"

optuna.logging.set_verbosity(optuna.logging.WARNING)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = __import__("sys").stdout
    def write(self, s): self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.stdout.flush(); self.f.flush()
    def close(self): self.f.close()


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def canon_rf():
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )


def cal_oof(base_est_factory, X, tr, te):
    """Isotonic-calibrated: fit on TRAIN fold, predict on TEST fold."""
    est = CalibratedClassifierCV(base_est_factory(), method="isotonic", cv=3)
    est.fit(X[tr], y[tr])
    return est.predict_proba(X[te])[:, 1]


# ================= assemble frame + sequences =================
import sys
tee = Tee(LOG_PATH); sys.stdout = tee
print(f"[t2] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
cpt_map = _load_cpt_map()
raw = load_raw()
enc_nodupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
merged = (raw.encounters.merge(enc_nodupes, on="LogID", how="inner")
          .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]], on="LogID", how="inner")
          .reset_index(drop=True))
ss = merged[["LogID"]].copy()
ss["_ss"] = pd.to_datetime(merged.get("Procedure/Surgery Start"), errors="coerce")
merged = merged.merge(build_preop_trajectory_features(raw.enc_unit_edges, ss),
                     on="LogID", how="left")
for c in C.TRAJECTORY_FEATURE_COLUMNS:
    merged[c] = merged[c].fillna(0)
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)
nsqip_cols = [c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
derived_cols = [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged.columns]
feat_cols = nsqip_cols + derived_cols
y = merged["ReadmittedWithin30Days"].astype(int).values
N = len(merged)
row_of = {l: i for i, l in enumerate(merged["LogID"])}
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)

# hand-crafted care-path sequence features (mirrors cv_seq_gru)
u = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
u["cid"] = u["Clarity_ID"].astype("Int64").astype(str)
u["g"] = u["UnitType"].map(UNIT_BUCKET).fillna("Other")
ud = u.dropna(subset=["Clarity_ID"]).drop_duplicates("cid", keep="first").set_index("cid")
a3 = raw.enc_unit_edges.copy()
a3["UnitType"] = _norm(a3["DepartmentID"]).map(ud["g"]).fillna(a3["UnitType"])
a3["InTime"] = pd.to_datetime(a3["InTime"], errors="coerce")
a3["Hours"] = pd.to_numeric(a3.get("Hours"), errors="coerce").fillna(0.0)
a3["LogID"] = a3["LogID"].astype(str)
a3 = a3.sort_values(["LogID", "InTime"])

seq_idx = np.full((N, MAXLEN), PAD, dtype=np.int64)
seq_num = np.zeros((N, MAXLEN, NUMF), dtype=np.float32)
lengths = np.ones(N, dtype=np.int64)
stat_rows = []
for lid, grp in a3.groupby("LogID", sort=False):
    r = row_of.get(lid)
    if r is None:
        continue
    units = grp["UnitType"].tolist()
    stat_rows.append(dict(seq_feature_dict(units), LogID=lid))
    steps = list(zip(units, grp["Hours"].tolist(), grp["InTime"].tolist()))[:MAXLEN]
    lengths[r] = max(len(steps), 1)
    for j, (un, h, t) in enumerate(steps):
        seq_idx[r, j] = U2I.get(un, U2I["Other"])
        hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        seq_num[r, j] = [np.log1p(max(h, 0.0)),
                         np.sin(2 * np.pi * hour / 24.0),
                         np.cos(2 * np.pi * hour / 24.0),
                         (j + 1) / MAXLEN]
Fstat = merged[["LogID"]].merge(pd.DataFrame(stat_rows).astype({"LogID": str}),
                                on="LogID", how="left").fillna(0)
stat_cols = [c for c in Fstat.columns if c != "LogID"]

# provider/unit features for the base heterograph (rf_gnn_emb)
prov_ids, X_prov, _ = build_provider_features(raw.prov_attrs)
unit_ids, X_unit, _ = build_unit_features(raw.unit_attrs)

print(f"[t2] cohort={N:,}  base={y.mean()*100:.2f}%  "
      f"tab feats={len(nsqip_cols)}  +derived={len(derived_cols)}  "
      f"static seq cands={len(stat_cols)}", flush=True)


# ================= GRU care-path encoder =================
class SeqGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(PAD + 1, EMB_DIM, padding_idx=PAD)
        self.gru = nn.GRU(EMB_DIM + NUMF, HID, batch_first=True)
        self.head = nn.Linear(HID, 2)

    def encode(self, idx, num, lens):
        x = torch.cat([self.emb(idx), num], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]

    def forward(self, idx, num, lens):
        return self.head(self.encode(idx, num, lens))


def train_gru(tr, va):
    set_seed(SEED); net = SeqGRU().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    npos, nneg = float((y[tr] == 1).sum()), float((y[tr] == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)],
                     dtype=torch.float32, device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    Xi = torch.tensor(seq_idx, device=dev); Xn = torch.tensor(seq_num, device=dev)
    Ln = torch.tensor(lengths, device=dev); Y = torch.tensor(y, device=dev)
    best, state, pat = -1.0, None, PAT
    rng = np.random.default_rng(SEED)
    for ep in range(MAX_EP):
        net.train(); perm = rng.permutation(tr)
        for s in range(0, len(perm), 4096):
            b = perm[s:s + 4096]
            opt.zero_grad()
            lf(net(Xi[b], Xn[b], Ln[b]), Y[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.softmax(net(Xi[va], Xn[va], Ln[va]), -1)[:, 1].cpu().numpy()
        try: vauc = roc_auc_score(y[va], pv)
        except ValueError: vauc = 0.5
        if vauc > best:
            best, state, pat = vauc, {k: v.detach().cpu().clone()
                                      for k, v in net.state_dict().items()}, PAT
        else:
            pat -= 1
            if pat <= 0: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return net.encode(Xi, Xn, Ln).cpu().numpy()


# ================= per-fold pooled-OOF loop =================
MODELS = ["rf_tuned_tab", "rf_clin", "rf_seq", "rf_gru", "rf_gnn_emb"]
oof = {m: np.full(N, np.nan) for m in MODELS}

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
    rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]
    print(f"[t2] === fold {fi+1}/{K} tr={len(tr):,} va={len(va):,} te={len(te):,} ===", flush=True)

    # ---- shared tabular design (fit on TRAIN only) ----
    _, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    Xcpt = ohe.transform(cpt_arr)
    X_base = np.hstack([Xtab, Xcpt])

    # ---- rf_clin: canonical RF on tabular ----
    oof["rf_clin"][te] = cal_oof(canon_rf, X_base, tr, te)
    print(f"[t2] fold {fi+1} rf_clin done", flush=True)

    # ---- rf_seq: canonical RF on tabular + hand-crafted care-path seq ----
    keep = [c for c in stat_cols if (Fstat.iloc[tr][c] > 0).sum() >= MIN_SUP]
    Xseq = Fstat[keep].values.astype(float)
    X_seq = np.hstack([X_base, Xseq])
    oof["rf_seq"][te] = cal_oof(canon_rf, X_seq, tr, te)
    print(f"[t2] fold {fi+1} rf_seq done ({len(keep)} seq cols)", flush=True)

    # ---- rf_gru: canonical RF on tabular + GRU embedding ----
    gemb = train_gru(tr, va)
    ge_std = (gemb - gemb[tr].mean(0)) / (gemb[tr].std(0) + 1e-8)
    X_gru = np.hstack([X_base, ge_std])
    oof["rf_gru"][te] = cal_oof(canon_rf, X_gru, tr, te)
    print(f"[t2] fold {fi+1} rf_gru done", flush=True)

    # ---- rf_gnn_emb: canonical RF on tabular + GNN encounter embedding ----
    train_mask = np.zeros(N, bool); train_mask[tr] = True
    val_mask   = np.zeros(N, bool); val_mask[va] = True
    test_mask  = np.zeros(N, bool); test_mask[te] = True
    artifacts = build_graph(
        raw=raw, encounters_merged=merged, enc_features=Xtab,
        prov_ids=prov_ids, prov_features=X_prov,
        unit_ids=unit_ids, unit_features=X_unit,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )
    set_seed(SEED + fi)
    gmodel, _ = train_model(artifacts, cfg=C.DEFAULTS_TRAIN, save_dir=None, verbose=False)
    tables = extract_embeddings(gmodel, artifacts,
                                raw_prov_attrs=raw.prov_attrs,
                                raw_unit_attrs=raw.unit_attrs, device=dev)
    enc_emb = tables.encounter.copy()
    enc_emb["LogID"] = enc_emb["LogID"].astype(str)
    emb_cols = [c for c in enc_emb.columns if c.startswith("emb_")]
    emb_df = merged[["LogID"]].merge(enc_emb, on="LogID", how="left")
    Emb = emb_df[emb_cols].fillna(0).values.astype(float)
    Emb_std = (Emb - Emb[tr].mean(0)) / (Emb[tr].std(0) + 1e-8)
    X_gnn = np.hstack([X_base, Emb_std])
    oof["rf_gnn_emb"][te] = cal_oof(canon_rf, X_gnn, tr, te)
    print(f"[t2] fold {fi+1} rf_gnn_emb done ({len(emb_cols)} emb cols)", flush=True)

    # ---- rf_tuned_tab: per-fold TPE-tuned RF on X_base ----
    # Objective: median 3-fold inner-CV AUROC on TRAIN, keep isotonic ext consistent.
    Xtr, ytr = X_base[tr], y[tr]

    def objective(trial):
        hp = dict(
            n_estimators=trial.suggest_categorical("n_estimators", [200, 500, 800]),
            min_samples_leaf=trial.suggest_categorical("min_samples_leaf", [2, 5, 10, 20]),
            max_features=trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
            class_weight=trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]),
        )
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fi)
        aucs = []
        for i_tr, i_va in inner.split(np.zeros(len(ytr)), ytr):
            m = RandomForestClassifier(random_state=SEED, n_jobs=-1, **hp).fit(
                Xtr[i_tr], ytr[i_tr])
            p = m.predict_proba(Xtr[i_va])[:, 1]
            aucs.append(roc_auc_score(ytr[i_va], p))
        return float(np.median(aucs))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED + fi))
    study.optimize(objective, n_trials=N_TPE_TRIALS, show_progress_bar=False)
    best_hp = study.best_params
    print(f"[t2] fold {fi+1} rf_tuned_tab best HP={best_hp}  "
          f"inner-med AUROC={study.best_value:.3f}", flush=True)

    def tuned_rf():
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **best_hp)

    oof["rf_tuned_tab"][te] = cal_oof(tuned_rf, X_base, tr, te)
    print(f"[t2] fold {fi+1} rf_tuned_tab done", flush=True)

# =================== metrics + CIs ===================
assert all(not np.isnan(oof[m]).any() for m in MODELS)
np.savez(OUT_DIR / "table2_final_dca_oof.npz", y=y, **{m: oof[m] for m in MODELS})

print("\n[t2] ==== Table 2 (5-fold pooled OOF, isotonic-calibrated) ====", flush=True)
print(f"     N={N:,}  base rate={y.mean()*100:.2f}%  bootstrap n_boot={N_BOOT}", flush=True)
rows = []
for m in MODELS:
    au = float(roc_auc_score(y, oof[m])); ap = float(average_precision_score(y, oof[m]))
    au_ci = _bootstrap_ci(y, oof[m], roc_auc_score, n_boot=N_BOOT, seed=0)
    ap_ci = _bootstrap_ci(y, oof[m], average_precision_score, n_boot=N_BOOT, seed=1)
    br = float(brier_score_loss(y, oof[m]))
    rows.append({"model": m, "AUROC": au, "AUROC_lo": au_ci[0], "AUROC_hi": au_ci[1],
                 "AUPRC": ap, "AUPRC_lo": ap_ci[0], "AUPRC_hi": ap_ci[1], "Brier": br})
    print(f"[t2] {m:14s}  AUROC {au:.3f} ({au_ci[0]:.3f}-{au_ci[1]:.3f})  "
          f"AUPRC {ap:.3f} ({ap_ci[0]:.3f}-{ap_ci[1]:.3f})  Brier {br:.4f}", flush=True)

pd.DataFrame(rows).to_csv(OUT_DIR / "table2_final_dca_results.csv", index=False)

# =================== decision-curve analysis ===================
curves = {m: decision_curve(y, oof[m], THRESHOLDS) for m in MODELS}
ref = curves["rf_gru"]

cols = ["threshold", "treat_all", "treat_none"] + MODELS
lines = [",".join(cols)]
for i, t in enumerate(THRESHOLDS):
    row = [f"{t:.2f}", f"{ref.treat_all[i]:.6f}", "0.000000"] + [
        f"{curves[m].net_benefit[i]:.6f}" for m in MODELS]
    lines.append(",".join(row))
(OUT_DIR / "table2_final_dca.csv").write_text("\n".join(lines) + "\n")

print("\n[t2] === Net benefit at key thresholds ===", flush=True)
print(f"       {'p_t':>5s} {'treat_all':>10s} " +
      " ".join(f"{m:>13s}" for m in MODELS), flush=True)
nb_at = {}
for t in KEY:
    j = int(np.argmin(np.abs(THRESHOLDS - t)))
    row_nb = {"treat_all": float(ref.treat_all[j])}
    row_nb.update({m: float(curves[m].net_benefit[j]) for m in MODELS})
    nb_at[f"{t:.2f}"] = row_nb
    print(f"       {t:>5.2f} {ref.treat_all[j]:>10.4f} " +
          " ".join(f"{curves[m].net_benefit[j]:>13.4f}" for m in MODELS),
          flush=True)

(OUT_DIR / "table2_final_dca.json").write_text(json.dumps(
    dict(cohort_n=N, prevalence=float(y.mean()),
         results=rows, net_benefit_at=nb_at), indent=2))

# =================== plot ===================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(THRESHOLDS, ref.treat_none, color="0.6", lw=1, ls=":", label="Treat none")
    ax.plot(THRESHOLDS, ref.treat_all, color="0.4", lw=1, ls="--", label="Treat all")
    style = {
        "rf_tuned_tab": ("#1f77b4", "-",  "RF tabular (per-fold tuned) — reported baseline"),
        "rf_clin":      ("#d62728", ":",  "RF: clinical only"),
        "rf_seq":       ("#2ca02c", "-.", "RF: clinical + hand-crafted care path"),
        "rf_gru":       ("#ff7f0e", "-",  "RF + GRU care-path encoder — deployable"),
        "rf_gnn_emb":   ("#9467bd", "--", "RF + GNN encounter embedding"),
    }
    for m in MODELS:
        c, ls, lab = style[m]
        ax.plot(THRESHOLDS, curves[m].net_benefit, color=c, ls=ls, lw=1.8, label=lab)
    ax.axvline(y.mean(), color="0.85", lw=1, zorder=0)
    ax.set_xlabel("Threshold probability $p_t$"); ax.set_ylabel("Net benefit")
    ax.set_xlim(THRESHOLDS[0], THRESHOLDS[-1])
    lo = min(curves[m].net_benefit.min() for m in MODELS)
    ax.set_ylim(min(-0.01, lo * 1.1), float(y.mean()) * 1.05)
    ax.set_title(f"Decision curve — five Table 2 models (5-fold CV, N={N:,})")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table2_final_dca.png", dpi=150)
    print(f"\n[t2] wrote {OUT_DIR/'table2_final_dca.png'}", flush=True)
except Exception as e:
    print(f"[t2] plot skipped: {type(e).__name__}: {e}", flush=True)

print("[t2] wrote results.csv, dca.csv, dca.json, oof.npz", flush=True)
tee.close()
