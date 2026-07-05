"""Ablation: does adding missingness / lab-count / care-structure signal beat
the current best RF and RF+GRU models? Data audit showed lab-missingness is a
2x readmit signal that mean-imputation discards; care-structure adds +0.004
AUROC by itself. Test each block singly and combined, on RF and RF+GRU."""
from __future__ import annotations
import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (add_calendar_features, apply_preprocess,
                           build_preop_trajectory_features, fit_preprocess,
                           load_raw, _read_table)
from medhg_ps.deploy import MIN_SUP, UNIT_BUCKET, seq_feature_dict, _load_cpt_map
from medhg_ps.evaluate import _bootstrap_ci
from medhg_ps.train import set_seed, _resolve_device

# ---- config (mirrors cv_seq_gru + table2_final_dca) -----------------------
SEED, K, VAL_FRAC = 42, 5, 0.10
MAX_EPOCHS, PATIENCE = 120, 12
MAXLEN = 40
UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]
U2I = {u: i for i, u in enumerate(UNITS)}
PAD = len(UNITS)
EMB_DIM, HID, NUMF = 16, 32, 4
LAB_COLS = ["NA", "BUN", "Creat", "ALB", "BT", "SGOT", "ALKPhos",
            "WBC", "HCT", "PLT", "INR", "APTT"]
INST_CATS = ["ICU", "PICU", "NICU", "Med/Surg", "Procedural Area", "ED",
             "Labor and Delivery", "Postpartum", "Antepartum", "Nursery",
             "Pediatrics", "Psych", "Rehab", "Hospice"]
dev = _resolve_device(C.DEFAULTS_TRAIN.device)

def RF():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                  max_features="sqrt", class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


# ---- assemble base frame + GRU sequences ---------------------------------
print("[abl] loading...", flush=True)
cpt_map = _load_cpt_map()
raw = load_raw()
enc_nodupes = raw.enc_features.drop(
    columns=([c for c in raw.encounters.columns
              if c != "LogID" and c in raw.enc_features.columns]
             + ["ReadmittedWithin30Days"]), errors="ignore")
merged = (raw.encounters.merge(enc_nodupes, on="LogID", how="inner")
          .merge(raw.labels[["LogID", "ReadmittedWithin30Days"]],
                 on="LogID", how="inner").reset_index(drop=True))
ss = merged[["LogID"]].copy()
ss["_ss"] = pd.to_datetime(merged.get("Procedure/Surgery Start"), errors="coerce")
merged = merged.merge(build_preop_trajectory_features(raw.enc_unit_edges, ss),
                     on="LogID", how="left")
for c in C.TRAJECTORY_FEATURE_COLUMNS:
    merged[c] = merged[c].fillna(0)
merged = add_calendar_features(merged)
merged["LogID"] = merged["LogID"].astype(str)
feat_cols = ([c for c in C.MODEL_FEATURE_COLUMNS if c in merged.columns]
             + [c for c in C.ENCOUNTER_DERIVED_COLUMNS if c in merged.columns])
y_all = merged["ReadmittedWithin30Days"].astype(int).values
N = len(merged)
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)
print(f"[abl] N={N:,}  base={y_all.mean()*100:.2f}%  feat_cols={len(feat_cols)}", flush=True)


# ---- BLOCK M: missingness (before imputation) + BLOCK L: lab count -------
print("[abl] extracting raw missingness (pre-imputation)...", flush=True)
raw_bulk = _read_table(C.ENC_FEATURES_CSV, C.BULK_FEATURES_COLUMNS).copy()
raw_bulk["LogID"] = raw_bulk["LogID"].astype(str)
miss = raw_bulk[["LogID"] + LAB_COLS].copy()
for col in LAB_COLS:
    miss[f"{col}__present"] = pd.to_numeric(miss[col], errors="coerce").notna().astype(int)
miss["lab_count"] = miss[[f"{c}__present" for c in LAB_COLS]].sum(axis=1)
miss = miss[["LogID"] + [f"{c}__present" for c in LAB_COLS] + ["lab_count"]]
miss = merged[["LogID"]].merge(miss, on="LogID", how="left")
# any missing joins -> 0 (safe default; won't happen in practice — same cohort)
miss = miss.fillna(0)
M_COLS = [f"{c}__present" for c in LAB_COLS]
L_COLS = ["lab_count"]
print(f"[abl]   M block: {len(M_COLS)} flags, mean present-rate per row = "
      f"{miss[M_COLS].values.mean():.2f}", flush=True)


# ---- BLOCK S: care structure (unit-type comp + team + surgeon volume) ----
print("[abl] extracting care-structure block...", flush=True)
a3s = raw.enc_unit_edges.copy()
a3s["LogID"] = a3s["LogID"].astype(str)
# InstitutionType (raw 14-bucket) is on A3 per your data audit
inst = a3s.get("InstitutionType")
if inst is None:
    # fall back to UnitType if InstitutionType isn't present
    inst = a3s["UnitType"]
a3s["_inst"] = inst.astype(str)
# per-encounter: did the encounter touch each of the 14 categories?
touched = (a3s.assign(one=1)
             .pivot_table(index="LogID", columns="_inst", values="one",
                         aggfunc="max", fill_value=0))
touched = touched.reindex(columns=INST_CATS, fill_value=0).reset_index()
touched.columns = ["LogID"] + [f"touched_{c}" for c in INST_CATS]

# Team composition per encounter from A2 + A4
a2 = raw.enc_prov_edges.copy()
a2["LogID"] = a2["LogID"].astype(str)
a4 = raw.prov_attrs.copy()
a4["ProvID"] = a4["ProvID"].astype(str)
a2["ProvID"] = a2["ProvID"].astype(str)
a2 = a2.merge(a4[["ProvID", "ProvType"]], on="ProvID", how="left")
a2["_pt"] = a2["ProvType"].astype(str).str.lower()
def _has(kw): return a2["_pt"].str.contains(kw, na=False).astype(int)
a2["is_attending"] = _has("attending")
a2["is_resident"]  = _has("resident")
a2["is_fellow"]    = _has("fellow")
a2["is_anesthesiologist"] = _has("anesthesiologist")
a2["is_nurse_anesth"] = _has("nurse anesth")
team = (a2.groupby("LogID")[["is_attending","is_resident","is_fellow",
                              "is_anesthesiologist","is_nurse_anesth"]]
          .sum().reset_index())
team.columns = ["LogID","n_attending","n_resident","n_fellow",
                "n_anesth","n_nurse_anesth"]

# Surgeon volume
enc = raw.encounters[["LogID","PrimarySurgID"]].copy()
enc["LogID"] = enc["LogID"].astype(str)
enc["PrimarySurgID"] = enc["PrimarySurgID"].astype(str)
vol = a4[["ProvID","CaseVolume2yr"]].rename(columns={"ProvID":"PrimarySurgID"})
vol["CaseVolume2yr"] = pd.to_numeric(vol["CaseVolume2yr"], errors="coerce")
surg_vol = enc.merge(vol, on="PrimarySurgID", how="left")[["LogID","CaseVolume2yr"]]
surg_vol["CaseVolume2yr"] = surg_vol["CaseVolume2yr"].fillna(0)

S_df = (merged[["LogID"]]
        .merge(touched, on="LogID", how="left")
        .merge(team,    on="LogID", how="left")
        .merge(surg_vol,on="LogID", how="left"))
S_df = S_df.fillna(0)
S_COLS = ([c for c in S_df.columns if c.startswith("touched_")]
          + ["n_attending","n_resident","n_fellow","n_anesth","n_nurse_anesth",
             "CaseVolume2yr"])
print(f"[abl]   S block: {len(S_COLS)} features "
      f"({sum(c.startswith('touched_') for c in S_COLS)} unit-comp + "
      f"5 team + 1 surg-vol)", flush=True)


# ---- GRU sequences (identical to cv_seq_gru) -----------------------------
u = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
u["cid"] = u["Clarity_ID"].astype("Int64").astype(str)
u["g"] = u["UnitType"].map(UNIT_BUCKET).fillna("Other")
ud = u.dropna(subset=["Clarity_ID"]).drop_duplicates("cid", keep="first").set_index("cid")
a3g = raw.enc_unit_edges.copy()
a3g["UnitType"] = (_norm(a3g["DepartmentID"]).map(ud["g"]).fillna(a3g["UnitType"]))
a3g["InTime"] = pd.to_datetime(a3g["InTime"], errors="coerce")
a3g["Hours"] = pd.to_numeric(a3g.get("Hours"), errors="coerce").fillna(0.0)
a3g["LogID"] = a3g["LogID"].astype(str)
a3g = a3g.sort_values(["LogID","InTime"])

row_of = {l: i for i, l in enumerate(merged["LogID"])}
seq_idx = np.full((N, MAXLEN), PAD, dtype=np.int64)
seq_num = np.zeros((N, MAXLEN, NUMF), dtype=np.float32)
lengths = np.ones(N, dtype=np.int64)
for lid, grp in a3g.groupby("LogID", sort=False):
    r = row_of.get(lid)
    if r is None:
        continue
    units = grp["UnitType"].tolist()
    hrs = grp["Hours"].tolist(); times = grp["InTime"].tolist()
    steps = [(U2I.get(un, U2I["Other"]), h, t) for un, h, t in zip(units, hrs, times)]
    steps = steps[:MAXLEN]
    L = len(steps); lengths[r] = max(L, 1)
    for j, (ui, h, t) in enumerate(steps):
        seq_idx[r, j] = ui
        hour = (t.hour + t.minute / 60.0) if pd.notna(t) else 0.0
        seq_num[r, j] = [np.log1p(max(h, 0.0)),
                         np.sin(2 * np.pi * hour / 24.0),
                         np.cos(2 * np.pi * hour / 24.0),
                         (j + 1) / MAXLEN]

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
    set_seed(SEED)
    net = SeqGRU().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    npos, nneg = float((y_all[tr] == 1).sum()), float((y_all[tr] == 0).sum())
    w = torch.tensor([(npos + nneg) / (2 * nneg), (npos + nneg) / (2 * npos)],
                     dtype=torch.float32, device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    Xi = torch.tensor(seq_idx, device=dev)
    Xn = torch.tensor(seq_num, device=dev)
    Ln = torch.tensor(lengths, device=dev)
    Y = torch.tensor(y_all, device=dev)
    best, state, pat = -1.0, None, PATIENCE
    rng = np.random.default_rng(SEED)
    for ep in range(MAX_EPOCHS):
        net.train(); perm = rng.permutation(tr)
        for s in range(0, len(perm), 4096):
            b = perm[s:s + 4096]
            opt.zero_grad()
            loss = lf(net(Xi[b], Xn[b], Ln[b]), Y[b])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.softmax(net(Xi[va], Xn[va], Ln[va]), -1)[:, 1].cpu().numpy()
        try:
            vauc = roc_auc_score(y_all[va], pv)
        except ValueError:
            vauc = 0.5
        if vauc > best:
            best, state, pat = (vauc,
                                {k: v.detach().cpu().clone()
                                 for k, v in net.state_dict().items()},
                                PATIENCE)
        else:
            pat -= 1
            if pat <= 0: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        emb = net.encode(Xi, Xn, Ln).cpu().numpy()
    return emb, best


# ---- design builder ------------------------------------------------------
def build_design(tr, blocks: str, gru_emb=None):
    """blocks is a str from {'', 'M','L','ML','S','MLS'} added to Xtab_base;
    if gru_emb is not None, also concatenate the standardized GRU embedding."""
    Xtab, st = fit_preprocess(merged[feat_cols].iloc[tr].reset_index(drop=True),
                               id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    parts = [Xtab, ohe.transform(cpt_arr)]
    if "M" in blocks:
        parts.append(miss[M_COLS].values.astype(float))
    if "L" in blocks:
        # standardize lab_count on train
        v = miss[L_COLS].values.astype(float)
        sc = StandardScaler().fit(v[tr])
        parts.append(sc.transform(v))
    if "S" in blocks:
        v = S_df[S_COLS].values.astype(float)
        sc = StandardScaler().fit(v[tr])
        parts.append(sc.transform(v))
    if gru_emb is not None:
        sc = StandardScaler().fit(gru_emb[tr])
        parts.append(sc.transform(gru_emb))
    X = np.hstack(parts)
    # final scaler (same as cv_seq_gru pattern)
    return StandardScaler().fit(X[tr]).transform(X)


# ---- fold-honest 5-fold CV with isotonic calibration --------------------
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
splits = list(skf.split(np.zeros(N), y_all))

# per fold: carve val slice from train for GRU
fold_val = []
for tr_all, te in splits:
    rng = np.random.default_rng(SEED)
    tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N))
    va, tr = tr_all[:nv], tr_all[nv:]
    fold_val.append((tr, va, te))

# Pre-train per-fold GRU once (reused for rows 7-9)
print("[abl] training per-fold GRU (for RF+GRU rows)...", flush=True)
gru_embs = []
for i, (tr, va, te) in enumerate(fold_val):
    emb, vb = train_gru(tr, va)
    gru_embs.append(emb)
    print(f"[abl]   fold {i+1}/{K} GRU val AUROC {vb:.3f}", flush=True)


def cv_oof(blocks: str, use_gru: bool) -> np.ndarray:
    p = np.full(N, np.nan)
    for i, (tr, va, te) in enumerate(fold_val):
        gru = gru_embs[i] if use_gru else None
        X = build_design(np.concatenate([tr, va]), blocks, gru_emb=gru)
        # calibration cv=3 on the train+val block; test remains held out
        full_tr = np.concatenate([tr, va])
        est = CalibratedClassifierCV(RF(), method="isotonic", cv=3)
        est.fit(X[full_tr], y_all[full_tr])
        p[te] = est.predict_proba(X[te])[:, 1]
    return p


CELLS = [
    # (label, blocks, use_gru)
    ("RF base",           "",    False),   # baseline
    ("RF +M",             "M",   False),
    ("RF +L",             "L",   False),
    ("RF +M+L",           "ML",  False),
    ("RF +S",             "S",   False),
    ("RF +M+L+S",         "MLS", False),
    ("RF+GRU base",       "",    True),
    ("RF+GRU +M+L",       "ML",  True),
    ("RF+GRU +M+L+S",     "MLS", True),
]

results = []
oofs = {}
for label, blocks, use_gru in CELLS:
    print(f"[abl] running: {label}  blocks='{blocks}'  use_gru={use_gru}", flush=True)
    p = cv_oof(blocks, use_gru)
    au = roc_auc_score(y_all, p)
    ap = average_precision_score(y_all, p)
    br = brier_score_loss(y_all, p)
    au_ci = _bootstrap_ci(y_all, p, roc_auc_score, n_boot=2000, seed=0)
    ap_ci = _bootstrap_ci(y_all, p, average_precision_score, n_boot=2000, seed=1)
    results.append(dict(model=label, blocks=blocks, use_gru=use_gru,
                        auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                        auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1],
                        brier=br))
    oofs[label] = p
    print(f"[abl]   -> AUROC {au:.3f} ({au_ci[0]:.3f}-{au_ci[1]:.3f})  "
          f"AUPRC {ap:.3f} ({ap_ci[0]:.3f}-{ap_ci[1]:.3f})  Brier {br:.4f}",
          flush=True)

# save
df = pd.DataFrame(results)
outdir = Path("artifacts/newdata"); outdir.mkdir(parents=True, exist_ok=True)
df.to_csv(outdir / "enrich_ablation.csv", index=False)
np.savez(outdir / "enrich_ablation_oof.npz",
         y=y_all, **{k.replace(" ","_").replace("+","p"): v for k, v in oofs.items()})

# paired deltas vs matching baseline
rf_base = df.query("model == 'RF base'").iloc[0]
gru_base = df.query("model == 'RF+GRU base'").iloc[0]
print("\n=== Ablation results (calibrated pooled OOF, 5-fold seed 42, bootstrap n=2000) ===")
print(f"{'model':16s}  {'blocks':6s}  AUROC (95% CI)               AUPRC (95% CI)               Brier   dAUROC   dAUPRC")
for _, r in df.iterrows():
    if r["use_gru"]:
        base = gru_base; b_lbl = "vs RF+GRU"
    else:
        base = rf_base; b_lbl = "vs RF"
    dau = r["auroc"] - base["auroc"]; dap = r["auprc"] - base["auprc"]
    print(f"{r['model']:16s}  {r['blocks']:6s}  {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f})   "
          f"{r['auprc']:.3f} ({r['auprc_lo']:.3f}-{r['auprc_hi']:.3f})   "
          f"{r['brier']:.4f}   {dau:+.4f}  {dap:+.4f}  ({b_lbl})")

# verdict summary
print("\n=== VERDICT ===")
def by(label): return df.query(f"model == '{label}'").iloc[0]
print(f"RF baseline:       AUROC {by('RF base').auroc:.3f} / AUPRC {by('RF base').auprc:.3f}")
best_rf = df[~df.use_gru].sort_values("auprc", ascending=False).iloc[0]
print(f"Best RF variant:   {best_rf.model:15s} AUROC {best_rf.auroc:.3f} / AUPRC {best_rf.auprc:.3f}  "
      f"(dAUROC {best_rf.auroc-by('RF base').auroc:+.3f}, dAUPRC {best_rf.auprc-by('RF base').auprc:+.3f})")
print(f"RF+GRU baseline:   AUROC {by('RF+GRU base').auroc:.3f} / AUPRC {by('RF+GRU base').auprc:.3f}")
best_gru = df[df.use_gru].sort_values("auprc", ascending=False).iloc[0]
print(f"Best RF+GRU:       {best_gru.model:15s} AUROC {best_gru.auroc:.3f} / AUPRC {best_gru.auprc:.3f}  "
      f"(dAUROC {best_gru.auroc-by('RF+GRU base').auroc:+.3f}, dAUPRC {best_gru.auprc-by('RF+GRU base').auprc:+.3f})")
print("[abl] done", flush=True)
