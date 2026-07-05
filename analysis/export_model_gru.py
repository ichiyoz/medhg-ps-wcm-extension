"""Train the Table-2 GRU care-path model on the full cohort and serialize it.

Produces a deployable medhg_ps.deploy_gru.ReadmissionGRUModel:
  1. Build the GRU sequence inputs (unit type, hours, arrival time-of-day,
     position) for every encounter.
  2. Train the GRU on 100% of the cohort (early-stopped on a carved val split).
  3. Extract the encounter embeddings, concatenate to tabular + CPT features,
     fit an isotonic-calibrated histogram gradient-boosted tree.
  4. Serialize the bundle to artifacts/readmission_model_gru.joblib (+ a torch
     state_dict and a model card).

CV performance is taken from cv_seq_gru.py (AUROC 0.703, AUPRC 0.242,
Brier 0.081); this script reports the full-cohort fit, not a fresh CV.

    PYTHONPATH=. python analysis/export_model_gru.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import fit_preprocess, apply_preprocess, load_raw
from medhg_ps.deploy import UNIT_BUCKET, assemble_training_frame
from medhg_ps.deploy_gru import (HID, MAXLEN, NUMF, PAD, U2I, ReadmissionGRUModel,
                                 SeqGRU, visit_step_features)
from medhg_ps.train import set_seed, _resolve_device

SEED, VAL_FRAC = 42, 0.10
MAX_EPOCHS, PATIENCE = 120, 12
# CV metrics from cv_seq_gru.py (calibrated tab+gru), reported in the model card.
CV_METRICS = {"auroc": 0.703, "auprc": 0.242, "brier": 0.081,
              "auroc_ci": [0.695, 0.711], "auprc_ci": [0.230, 0.254], "folds": 5,
              "source": "analysis/cv_seq_gru.py + cv_decision_curve_all4.py (bootstrap CI)"}
OUT_DIR = Path("artifacts"); OUT_DIR.mkdir(exist_ok=True)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def build_sequences(merged):
    """Per-encounter (idx[MAXLEN], num[MAXLEN,NUMF], length) + the raw visit
    lists (for the model-card self-test), in merged-row order."""
    raw = load_raw()
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

    row_of = {l: i for i, l in enumerate(merged["LogID"].astype(str))}
    N = len(merged)
    seq_idx = np.full((N, MAXLEN), PAD, dtype=np.int64)
    seq_num = np.zeros((N, MAXLEN, NUMF), dtype=np.float32)
    lengths = np.ones(N, dtype=np.int64)
    visit_lists = [[] for _ in range(N)]
    for lid, grp in a3.groupby("LogID", sort=False):
        r = row_of.get(lid)
        if r is None:
            continue
        steps = list(zip(grp["UnitType"], grp["Hours"], grp["InTime"]))[:MAXLEN]
        lengths[r] = max(len(steps), 1)
        for j, (un, h, t) in enumerate(steps):
            hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
            seq_idx[r, j] = U2I.get(un, U2I["Other"])
            seq_num[r, j] = visit_step_features(h, hour, j)
            visit_lists[r].append({"unit": un, "hours": float(h), "arrival_hour": float(hour)})
    return seq_idx, seq_num, lengths, visit_lists


def train_full_gru(seq_idx, seq_num, lengths, y):
    """Train the GRU on all rows (early-stopped on a carved val split)."""
    set_seed(SEED)
    N = len(y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    nv = int(round(VAL_FRAC * N))
    va, tr = perm[:nv], perm[nv:]
    net = SeqGRU().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    npos, nneg = float((y[tr] == 1).sum()), float((y[tr] == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)],
                     dtype=torch.float32, device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    Xi = torch.tensor(seq_idx, device=dev); Xn = torch.tensor(seq_num, device=dev)
    Ln = torch.tensor(lengths, device=dev); Y = torch.tensor(y, device=dev)
    from sklearn.metrics import roc_auc_score
    best, state, pat = -1.0, None, PATIENCE
    for ep in range(MAX_EPOCHS):
        net.train(); bperm = rng.permutation(tr)
        for s in range(0, len(bperm), 4096):
            b = bperm[s:s + 4096]; opt.zero_grad()
            lf(net(Xi[b], Xn[b], Ln[b]), Y[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.softmax(net(Xi[va], Xn[va], Ln[va]), -1)[:, 1].cpu().numpy()
        try: vauc = roc_auc_score(y[va], pv)
        except ValueError: vauc = 0.5
        if vauc > best:
            best, state, pat = vauc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0: break
    net.load_state_dict(state); net.eval()
    print(f"[gru-export] GRU trained, best val AUROC {best:.3f}", flush=True)
    with torch.no_grad():
        emb = net.encode(Xi, Xn, Ln).cpu().numpy()
    return net, emb


def main():
    print("[gru-export] assembling frame...", flush=True)
    merged, feat_cols, cpt_arr, Fseq, seq_all, y = assemble_training_frame()
    N = len(merged)
    print(f"[gru-export] cohort={N:,}  base={y.mean()*100:.2f}%", flush=True)

    seq_idx, seq_num, lengths, visit_lists = build_sequences(merged)
    gru, emb = train_full_gru(seq_idx, seq_num, lengths, y)

    # tabular + CPT + GRU embedding -> calibrated HGB, fit on all rows
    Xtab, st = fit_preprocess(merged[feat_cols].reset_index(drop=True), id_cols=[])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr)
    Xcpt = ohe.transform(cpt_arr)
    scaler = StandardScaler().fit(np.hstack([Xtab, Xcpt, emb]))
    Xall = scaler.transform(np.hstack([Xtab, Xcpt, emb]))
    clf = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                       l2_regularization=1.0, random_state=42),
        method="isotonic", cv=5).fit(Xall, y)

    model = ReadmissionGRUModel(
        gru=gru.cpu().eval(), preprocess_state=st, tab_feat_cols=list(feat_cols),
        cpt_encoder=ohe, scaler=scaler, clf=clf, n_features=Xall.shape[1],
        base_rate=float(y.mean()), cv_metrics=CV_METRICS, version="1.0")

    # self-test: end-to-end predict on 3 encounters (raw records)
    sample = []
    for i in range(3):
        rec = {c: merged[c].iloc[i] for c in feat_cols}
        rec["PrimaryCPT"] = cpt_arr[i, 0]
        rec["postop_visits"] = visit_lists[i]
        sample.append(rec)
    probs = model.predict_proba(sample)
    print(f"[gru-export] self-test predict_proba (3 rows): {np.round(probs, 4)}", flush=True)
    assert np.all((probs >= 0) & (probs <= 1))

    out = OUT_DIR / "readmission_model_gru.joblib"
    joblib.dump(model, out)
    torch.save(gru.state_dict(), OUT_DIR / "readmission_gru_encoder.pt")
    (OUT_DIR / "model_card_gru.json").write_text(json.dumps(dict(
        version=model.version, estimator="hgb+gru", calibrated=True,
        base_rate=model.base_rate, n_features=model.n_features, cohort_n=int(N),
        seq=dict(units=list(U2I), maxlen=MAXLEN, hidden=HID),
        cv=CV_METRICS, default_threshold=model.threshold), indent=2))
    print(f"[gru-export] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"[gru-export] wrote {OUT_DIR/'readmission_gru_encoder.pt'} and model_card_gru.json", flush=True)


if __name__ == "__main__":
    main()
