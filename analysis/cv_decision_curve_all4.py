"""Decision curve analysis for all four Table-2 models.

Extends cv_decision_curve.py (which covered the two tree models) to the full
Table 2: the gradient-boosted tree on clinical features + hand-crafted care
path, the same on clinical features only, the GBT with the GRU care-path
sequence encoder, and the GNN-embedding + GBT. Each model's out-of-fold
isotonic-calibrated risk is collected on identical fold-honest 5-fold splits
(GRU and concept-graph HGT trained per fold, embeddings re-extracted), then
net benefit (Vickers & Elkin) is computed across the action-threshold range.

  gbt_gru   : HGB on clinical features + GRU care-path encoder   [Table 2 best]
  gbt_seq   : HGB on clinical features + hand-crafted care path
  gbt_clin  : HGB on clinical features only
  gnn_emb   : HGB on clinical features + care path + concept-graph HGT embedding

Writes artifacts/decision_curve_all4.{png,csv,json}. Aggregate output only.

    PYTHONPATH=. python analysis/cv_decision_curve_all4.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import dgl
from dgl.nn import HGTConv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (apply_preprocess, fit_preprocess, load_raw,
                           build_preop_trajectory_features, add_calendar_features)
from medhg_ps.deploy import MIN_SUP, UNIT_BUCKET, seq_feature_dict, _load_cpt_map
from medhg_ps.evaluate import decision_curve
from medhg_ps.train import set_seed, _resolve_device

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
MAX_EP, PAT = (15, 4) if SMOKE else (120, 12)
GMAX_EP, GPAT = (20, 5) if SMOKE else (150, 15)
THRESHOLDS = np.round(np.arange(0.02, 0.31, 0.01), 2)
KEY = [0.05, 0.10, 0.15, 0.20]
MAXLEN, EMB_DIM, HID, NUMF = 40, 16, 32, 4
UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]
U2I = {u: i for i, u in enumerate(UNITS)}; PAD = len(UNITS)
TOPK = 50
GHP = dict(hid=64, nl=2, lr=5e-3, dp=0.3)
COMORB = ["Diabetes Mellitus", "Hypertension requiring medication", "Heart Failure",
          "History of Severe COPD", "Ascites", "Disseminated Cancer", "Bleeding Disorder",
          "Preop Acute Kidney Injury", "Preop Dialysis", "Ventilator Dependent",
          "Immunosuppressive Therapy", "Current Smoker within 1 year",
          "Preop RBC Transfusions (72h)"]
OUT_DIR = Path("artifacts"); OUT_DIR.mkdir(exist_ok=True)
HGB = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=42)
dev = _resolve_device(C.DEFAULTS_TRAIN.device)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


# ================= assemble frame, sequences, concept graph =================
print(f"[dca4] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
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
merged = merged.merge(build_preop_trajectory_features(raw.enc_unit_edges, ss), on="LogID", how="left")
for c in C.TRAJECTORY_FEATURE_COLUMNS:
    merged[c] = merged[c].fillna(0)
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)
feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
             + [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged.columns])
y = merged["ReadmittedWithin30Days"].astype(int).values
N = len(merged)
row_of = {l: i for i, l in enumerate(merged["LogID"])}
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)

# bucketed, time-ordered A3 -> static features + GRU sequences
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
        seq_num[r, j] = [np.log1p(max(h, 0.0)), np.sin(2 * np.pi * hour / 24),
                         np.cos(2 * np.pi * hour / 24), (j + 1) / MAXLEN]
Fstat = merged[["LogID"]].merge(pd.DataFrame(stat_rows).astype({"LogID": str}),
                                on="LogID", how="left").fillna(0)
stat_cols = [c for c in Fstat.columns if c != "LogID"]

# five-node-type clinical-concept graph (enc, prov, unit, dx, cpt) -- as cv_emb_to_tree
P_ids = sorted({p for p in _norm(raw.enc_prov_edges["ProvID"]) if p and p != "nan"})
U_ids = sorted({d for d in _norm(raw.enc_unit_edges["DepartmentID"]) if d and d != "nan"})
DX_ids = [c for c in COMORB if c in merged.columns]
CPT_ids = sorted({v for v in cpt_arr.ravel() if v and v != "UNK"})
pidx = {p: N + i for i, p in enumerate(P_ids)}
o = N + len(P_ids); uidx = {d: o + i for i, d in enumerate(U_ids)}
o += len(U_ids); dxidx = {d: o + i for i, d in enumerate(DX_ids)}
o += len(DX_ids); cidx = {c: o + i for i, c in enumerate(CPT_ids)}
TOT = o + len(CPT_ids)
src, dst, et = [], [], []
ap = raw.enc_prov_edges[["LogID", "ProvID"]].copy(); ap["LogID"] = _norm(ap["LogID"]); ap["ProvID"] = _norm(ap["ProvID"])
for lid, pv in zip(ap["LogID"], ap["ProvID"]):
    r, pp = row_of.get(lid), pidx.get(pv)
    if r is not None and pp is not None:
        src += [r, pp]; dst += [pp, r]; et += [0, 1]
au = raw.enc_unit_edges[["LogID", "DepartmentID"]].copy(); au["LogID"] = _norm(au["LogID"]); au["DepartmentID"] = _norm(au["DepartmentID"])
for lid, dp in zip(au["LogID"], au["DepartmentID"]):
    r, uu = row_of.get(lid), uidx.get(dp)
    if r is not None and uu is not None:
        src += [r, uu]; dst += [uu, r]; et += [2, 3]
for dx in DX_ids:
    present = ~merged[dx].astype(str).str.strip().str.lower().isin(["no", "0", "nan", "none", ""])
    dn = dxidx[dx]
    for r in np.nonzero(present.values)[0]:
        src += [int(r), dn]; dst += [dn, int(r)]; et += [4, 5]
for r in range(N):
    cn = cidx.get(cpt_arr[r, 0])
    if cn is not None:
        src += [r, cn]; dst += [cn, r]; et += [6, 7]
g = dgl.add_self_loop(dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=TOT)).to(dev)
etype = torch.cat([torch.tensor(et, dtype=torch.long), torch.full((TOT,), 8, dtype=torch.long)]).to(dev)
ntype = torch.zeros(TOT, dtype=torch.long, device=dev)
ntype[N:N + len(P_ids)] = 1
ntype[N + len(P_ids):N + len(P_ids) + len(U_ids)] = 2
ntype[N + len(P_ids) + len(U_ids):N + len(P_ids) + len(U_ids) + len(DX_ids)] = 3
ntype[N + len(P_ids) + len(U_ids) + len(DX_ids):] = 4
NREL, NNT = 9, 5
print(f"[dca4] cohort={N:,}  graph nodes={TOT:,} edges={g.num_edges():,}", flush=True)


class SeqGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(PAD + 1, EMB_DIM, padding_idx=PAD)
        self.gru = nn.GRU(EMB_DIM + NUMF, HID, batch_first=True)
        self.head = nn.Linear(HID, 2)

    def encode(self, idx, num, lens):
        x = torch.cat([self.emb(idx), num], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed); return h[-1]

    def forward(self, idx, num, lens):
        return self.head(self.encode(idx, num, lens))


class HGTNet(nn.Module):
    def __init__(self, ind):
        super().__init__()
        self.drop = nn.Dropout(GHP["dp"]); self.layers = nn.ModuleList()
        dims = [ind] + [GHP["hid"]] * GHP["nl"]
        for i in range(GHP["nl"]):
            self.layers.append(HGTConv(dims[i], dims[i + 1] // 4, 4, NNT, NREL, dropout=GHP["dp"]))
        self.cls = nn.Linear(GHP["hid"], 2)

    def embed(self, gg, x):
        h = x
        for conv in self.layers:
            h = self.drop(Fn.relu(conv(gg, h, ntype, etype)))
        return h

    def forward(self, gg, x):
        return self.cls(self.embed(gg, x))


def _classweight(yt, trm):
    npos, nneg = float((yt[trm] == 1).sum()), float((yt[trm] == 0).sum())
    return torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)], device=dev)


def train_gru(tr, va, yt):
    set_seed(SEED); net = SeqGRU().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    trm = np.zeros(N, bool); trm[tr] = True
    lf = nn.CrossEntropyLoss(weight=_classweight(yt, torch.tensor(trm, device=dev)))
    Xi = torch.tensor(seq_idx, device=dev); Xn = torch.tensor(seq_num, device=dev)
    Ln = torch.tensor(lengths, device=dev); Y = torch.tensor(y, device=dev)
    best, state, pat = -1.0, None, PAT; rng = np.random.default_rng(SEED)
    for ep in range(MAX_EP):
        net.train(); perm = rng.permutation(tr)
        for s in range(0, len(perm), 4096):
            b = perm[s:s + 4096]; opt.zero_grad()
            nn.CrossEntropyLoss(weight=lf.weight)(net(Xi[b], Xn[b], Ln[b]), Y[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.softmax(net(Xi[va], Xn[va], Ln[va]), -1)[:, 1].cpu().numpy()
        try: vauc = roc_auc_score(y[va], pv)
        except ValueError: vauc = 0.5
        if vauc > best:
            best, state, pat = vauc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, PAT
        else:
            pat -= 1
            if pat <= 0: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return net.encode(Xi, Xn, Ln).cpu().numpy()


def train_hgt(Xn_feat, yt, trm, vam):
    set_seed(SEED); net = HGTNet(Xn_feat.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=GHP["lr"], weight_decay=1e-4)
    lf = nn.CrossEntropyLoss(weight=_classweight(yt, trm))
    yv, vm = yt[vam].cpu().numpy(), vam.cpu().numpy()
    best, state, pat = -1.0, None, GPAT
    for ep in range(GMAX_EP):
        net.train(); loss = lf(net(g, Xn_feat)[:N][trm], yt[trm])
        opt.zero_grad(); loss.backward(); opt.step(); net.eval()
        with torch.no_grad():
            p = torch.softmax(net(g, Xn_feat)[:N], -1)[:, 1].cpu().numpy()
        try: va = roc_auc_score(yv, p[vm])
        except ValueError: va = 0.5
        if va > best:
            best, state, pat = va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, GPAT
        else:
            pat -= 1
            if pat <= 0: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return net.embed(g, Xn_feat)[:N].cpu().numpy()


def cal_oof(Xall, tr, te):
    """isotonic-calibrated HGB; return test probabilities."""
    est = CalibratedClassifierCV(HGB(), method="isotonic", cv=3).fit(Xall[tr], y[tr])
    return est.predict_proba(Xall[te])[:, 1]


def main():
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    MODELS = ["gbt_gru", "gbt_seq", "gbt_clin", "gnn_emb"]
    oof = {m: np.full(N, np.nan) for m in MODELS}

    for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
        rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
        nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]
        trm = torch.zeros(N, dtype=torch.bool, device=dev); trm[tr] = True
        vam = torch.zeros(N, dtype=torch.bool, device=dev); vam[va] = True

        # shared tabular + cpt + static-seq blocks (fold-honest)
        Xtab, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True), id_cols=[])
        Xtab = apply_preprocess(merged[feat_cols], st)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
        Xcpt = ohe.transform(cpt_arr)
        keep = [c for c in stat_cols if (Fstat.iloc[tr][c] > 0).sum() >= MIN_SUP]
        Xseq = Fstat[keep].values.astype(float)

        def scaled(blocks):
            X = np.hstack(blocks); return StandardScaler().fit(X[tr]).transform(X)

        oof["gbt_seq"][te]  = cal_oof(scaled([Xtab, Xcpt, Xseq]), tr, te)
        oof["gbt_clin"][te] = cal_oof(scaled([Xtab, Xcpt]), tr, te)

        # model 1: GRU care-path encoder
        gemb = train_gru(tr, va, yt)
        ge = StandardScaler().fit(gemb[tr]).transform(gemb)
        oof["gbt_gru"][te] = cal_oof(scaled([Xtab, Xcpt, ge]), tr, te)

        # model 4: concept-graph HGT embedding (node feats = RF top-50 of enriched)
        Xall = np.hstack([Xtab, Xcpt, Xseq])
        sel = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                                     class_weight="balanced", random_state=42, n_jobs=-1).fit(Xall[tr], y[tr])
        top = np.argsort(sel.feature_importances_)[::-1][:TOPK]
        Xsel = StandardScaler().fit(Xall[tr][:, top]).transform(Xall[:, top])
        Xnode = torch.zeros((TOT, TOPK), dtype=torch.float32, device=dev); Xnode[:N] = torch.tensor(Xsel, device=dev)
        hemb = train_hgt(Xnode, yt, trm, vam)
        he = StandardScaler().fit(hemb[tr]).transform(hemb)
        oof["gnn_emb"][te] = cal_oof(scaled([Xtab, Xcpt, Xseq, he]), tr, te)
        print(f"[dca4] fold {fi + 1}/{K} scored", flush=True)

    assert all(not np.isnan(oof[m]).any() for m in MODELS)

    np.savez(OUT_DIR / "oof_all4.npz", y=y, **{m: oof[m] for m in MODELS})  # reuse without rerun
    from medhg_ps.evaluate import _bootstrap_ci
    sanity = {}
    for m in MODELS:
        au = float(roc_auc_score(y, oof[m])); ap = float(average_precision_score(y, oof[m]))
        au_ci = _bootstrap_ci(y, oof[m], roc_auc_score, n_boot=2000, seed=0)
        ap_ci = _bootstrap_ci(y, oof[m], average_precision_score, n_boot=2000, seed=1)
        sanity[m] = dict(auroc=au, auroc_ci=list(au_ci), auprc=ap, auprc_ci=list(ap_ci),
                         brier=float(brier_score_loss(y, oof[m])))
        print(f"[dca4] {m:9s} AUROC {au:.3f} ({au_ci[0]:.3f}-{au_ci[1]:.3f})  "
              f"AUPRC {ap:.3f} ({ap_ci[0]:.3f}-{ap_ci[1]:.3f})  Brier {sanity[m]['brier']:.4f}", flush=True)

    curves = {m: decision_curve(y, oof[m], THRESHOLDS) for m in MODELS}
    ref = curves["gbt_gru"]

    cols = ["threshold", "treat_all", "treat_none"] + MODELS
    lines = [",".join(cols)]
    for i, t in enumerate(THRESHOLDS):
        row = [f"{t:.2f}", f"{ref.treat_all[i]:.6f}", "0.000000"] + [f"{curves[m].net_benefit[i]:.6f}" for m in MODELS]
        lines.append(",".join(row))
    (OUT_DIR / "decision_curve_all4.csv").write_text("\n".join(lines) + "\n")

    print("\n=== Net benefit at key thresholds (all four models) ===")
    print(f"  {'p_t':>5s} {'treat_all':>10s} " + " ".join(f"{m:>9s}" for m in MODELS))
    nb_at = {}
    for t in KEY:
        j = int(np.argmin(np.abs(THRESHOLDS - t)))
        nb_at[f"{t:.2f}"] = {"treat_all": float(ref.treat_all[j]),
                             **{m: float(curves[m].net_benefit[j]) for m in MODELS}}
        print(f"  {t:>5.2f} {ref.treat_all[j]:>10.4f} " + " ".join(f"{curves[m].net_benefit[j]:>9.4f}" for m in MODELS))
    (OUT_DIR / "decision_curve_all4.json").write_text(json.dumps(
        dict(cohort_n=N, prevalence=float(y.mean()), sanity=sanity, net_benefit_at=nb_at), indent=2))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.8, 4.7))
        ax.plot(THRESHOLDS, ref.treat_none, color="0.6", lw=1, ls=":", label="Treat none")
        ax.plot(THRESHOLDS, ref.treat_all, color="0.4", lw=1, ls="--", label="Treat all")
        style = {"gbt_gru": ("#1f77b4", "-", "GBT + GRU care-path encoder"),
                 "gbt_seq": ("#2ca02c", "-.", "GBT: clinical + care path"),
                 "gbt_clin": ("#d62728", ":", "GBT: clinical only"),
                 "gnn_emb": ("#9467bd", "--", "GNN embedding + GBT")}
        for m in MODELS:
            c, ls, lab = style[m]
            ax.plot(THRESHOLDS, curves[m].net_benefit, color=c, ls=ls, lw=1.8, label=lab)
        ax.axvline(y.mean(), color="0.85", lw=1, zorder=0)
        ax.set_xlabel("Threshold probability $p_t$"); ax.set_ylabel("Net benefit")
        ax.set_xlim(THRESHOLDS[0], THRESHOLDS[-1])
        lo = min(curves[m].net_benefit.min() for m in MODELS)
        ax.set_ylim(min(-0.01, lo * 1.1), float(y.mean()) * 1.05)
        ax.set_title(f"Decision curve, all four models (5-fold CV, n={N:,})")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout(); fig.savefig(OUT_DIR / "decision_curve_all4.png", dpi=150)
        print(f"\n[dca4] wrote {OUT_DIR/'decision_curve_all4.png'}", flush=True)
    except Exception as e:
        print(f"[dca4] plot skipped: {type(e).__name__}: {e}", flush=True)
    print(f"[dca4] wrote decision_curve_all4.csv and .json", flush=True)


if __name__ == "__main__":
    main()
