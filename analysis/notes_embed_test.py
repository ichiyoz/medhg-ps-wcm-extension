"""Test whether local clinical-text embeddings from Notes.csv add discrimination
on the notes-covered subgroup (N=12,454; base 8.73%). Honest to the informative-
missingness confound: OOS evaluation restricted to the subgroup where notes exist,
because full-cohort lift would just re-learn "no notes = ambulatory low-risk".

Encoder preference (first that loads without network / first found in cache):
  a) emilyalsentzer/Bio_ClinicalBERT
  b) UFNLP/gatortron-base
  c) distilbert-base-uncased
  d) sentence-transformers/all-MiniLM-L6-v2

Everything runs locally. No PHI leaves the box.
"""
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import medhg_ps.config as C
from medhg_ps.data import apply_preprocess, fit_preprocess
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.evaluate import _bootstrap_ci

# --------------------------------------------------------------------- config
NOTES_CSV = "/Users/yiyezhang/Downloads/medhg_ps_data/Notes.csv"
CACHE_NPZ = "artifacts/newdata/notes_embeddings_raw.npz"
LOG_PATH  = "artifacts/newdata/notes_embed.log"

NOTES_COLS = ["LogID","PAT_ID","EncounterCSN","SurgeryStart","DischargeTime",
              "NoteID","NoteTypeCode","NoteTypeName","NoteCreated","AuthorID",
              "MinutesAfterSurgeryStart","HoursBeforeDischarge","NoteText"]

CANDIDATES = [
    ("emilyalsentzer/Bio_ClinicalBERT", "cls",  512),
    ("UFNLP/gatortron-base",            "cls",  512),
    ("distilbert-base-uncased",         "cls",  512),
    ("sentence-transformers/all-MiniLM-L6-v2", "mean", 256),
]

SEED = 42
N_BOOT = 2000
BATCH = 24
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def RF():
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=10, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1)


def load_encoder():
    """Try HF models in order. First that loads (from cache or network) wins."""
    from transformers import AutoModel, AutoTokenizer
    for name, pool, maxlen in CANDIDATES:
        try:
            print(f"[enc] trying {name} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name).to(device).eval()
            print(f"[enc] LOADED {name}  pool={pool}  maxlen={maxlen}  device={device}", flush=True)
            return name, tok, mdl, pool, maxlen
        except Exception as e:
            print(f"[enc] {name} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
    raise RuntimeError("no encoder loaded")


def embed_notes(notes: pd.DataFrame) -> np.ndarray:
    """Encode NoteText per row; return [n_notes, dim] float32."""
    if os.path.exists(CACHE_NPZ):
        z = np.load(CACHE_NPZ, allow_pickle=True)
        if z["note_id"].shape[0] == len(notes):
            print(f"[enc] cache hit: {CACHE_NPZ} ({z['E'].shape})", flush=True)
            return z["E"].astype(np.float32)
    name, tok, mdl, pool, maxlen = load_encoder()
    texts = notes["NoteText"].fillna("").astype(str).tolist()
    n = len(texts)
    with torch.no_grad():
        sample = tok(texts[:1], truncation=True, max_length=maxlen,
                     padding=True, return_tensors="pt").to(device)
        out = mdl(**sample)
        dim = (out.last_hidden_state[:, 0]).shape[-1] if pool == "cls" else out.last_hidden_state.mean(1).shape[-1]
    E = np.zeros((n, dim), dtype=np.float32)
    t0 = time.time()
    for s in range(0, n, BATCH):
        batch = texts[s:s + BATCH]
        try:
            enc = tok(batch, truncation=True, max_length=maxlen,
                      padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**enc)
                if pool == "cls":
                    v = out.last_hidden_state[:, 0]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    v = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
            E[s:s + len(batch)] = v.detach().cpu().numpy()
        except RuntimeError as e:                          # OOM -> CPU fallback
            print(f"[enc] batch {s} err {e.__class__.__name__}, retry on CPU", flush=True)
            mdl_cpu = mdl.cpu()
            enc = tok(batch, truncation=True, max_length=maxlen,
                      padding=True, return_tensors="pt")
            with torch.no_grad():
                out = mdl_cpu(**enc)
                v = out.last_hidden_state[:, 0] if pool == "cls" \
                    else (out.last_hidden_state * enc["attention_mask"].unsqueeze(-1).float()).sum(1)
            E[s:s + len(batch)] = v.detach().numpy()
            mdl.to(device)
        if s % (BATCH * 50) == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(s + BATCH, 1) * (n - s - BATCH)
            print(f"[enc] {s + len(batch):>6d}/{n}  elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m", flush=True)
    np.savez_compressed(CACHE_NPZ,
                        note_id=notes["NoteID"].values.astype(str),
                        log_id=notes["LogID"].values.astype(str),
                        note_type_code=notes["NoteTypeCode"].values.astype(str),
                        E=E, encoder_name=name)
    print(f"[enc] wrote {CACHE_NPZ}", flush=True)
    return E


def pool_per_case(notes: pd.DataFrame, E: np.ndarray,
                  case_index: pd.Index, by_type: bool = False) -> np.ndarray:
    """Mean-pool per LogID; optionally per (LogID, NoteTypeCode) concatenated."""
    n = len(case_index); dim = E.shape[1]
    idx = notes.groupby("LogID").indices
    if not by_type:
        out = np.zeros((n, dim), dtype=np.float32)
        for i, lid in enumerate(case_index):
            r = idx.get(str(lid))
            if r is not None and len(r):
                out[i] = E[r].mean(0)
        return out
    # by type: concatenate one mean-pool per (type in TYPES)
    TYPES = ["1000000", "27", "1000004", "4", "3", "1000012", "26"]
    out = np.zeros((n, dim * len(TYPES)), dtype=np.float32)
    by = notes.assign(_r=np.arange(len(notes))).groupby(["LogID", "NoteTypeCode"])["_r"].apply(list).to_dict()
    for i, lid in enumerate(case_index):
        for t, ty in enumerate(TYPES):
            r = by.get((str(lid), ty))
            if r is not None and len(r):
                out[i, t * dim:(t + 1) * dim] = E[r].mean(0)
    return out


def cv_oof(X: np.ndarray, y: np.ndarray, name: str):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = np.full(len(y), np.nan)
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        est = CalibratedClassifierCV(RF(), method="isotonic", cv=3).fit(X[tr], y[tr])
        p[te] = est.predict_proba(X[te])[:, 1]
        print(f"[cv] {name} fold {fi+1}/5", flush=True)
    au = roc_auc_score(y, p); ap = average_precision_score(y, p); br = brier_score_loss(y, p)
    au_ci = _bootstrap_ci(y, p, roc_auc_score, n_boot=N_BOOT, seed=0)
    ap_ci = _bootstrap_ci(y, p, average_precision_score, n_boot=N_BOOT, seed=1)
    return dict(name=name, auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1],
                brier=br, oof=p)


def build_xtab(feat_df: pd.DataFrame, feat_cols, cpt_arr: np.ndarray,
               tr_mask: np.ndarray) -> np.ndarray:
    """Design matrix built on subgroup (all rows), preprocessing fit only on train."""
    _, st = fit_preprocess(feat_df[feat_cols].iloc[tr_mask].reset_index(drop=True), id_cols=[])
    X = apply_preprocess(feat_df[feat_cols], st)
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr_mask])
    return np.hstack([X, oh.transform(cpt_arr)])


def cv_oof_designed(y: np.ndarray, feat_df: pd.DataFrame, feat_cols,
                    cpt_arr: np.ndarray, extra: Optional[np.ndarray], name: str):
    """Refit tabular design per fold on train rows (avoids leakage from
    fit_preprocess seeing test rows). extra is optionally the notes block."""
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = np.full(len(y), np.nan)
    for fi, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        X = build_xtab(feat_df, feat_cols, cpt_arr, tr)
        if extra is not None:
            X = np.hstack([X, extra])
        est = CalibratedClassifierCV(RF(), method="isotonic", cv=3).fit(X[tr], y[tr])
        p[te] = est.predict_proba(X[te])[:, 1]
        print(f"[cv] {name} fold {fi+1}/5", flush=True)
    au = roc_auc_score(y, p); ap = average_precision_score(y, p); br = brier_score_loss(y, p)
    au_ci = _bootstrap_ci(y, p, roc_auc_score, n_boot=N_BOOT, seed=0)
    ap_ci = _bootstrap_ci(y, p, average_precision_score, n_boot=N_BOOT, seed=1)
    return dict(name=name, auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1],
                brier=br, oof=p)


def main():
    Path("artifacts/newdata").mkdir(parents=True, exist_ok=True)

    print("[t] loading cohort + notes", flush=True)
    merged, feat_cols, cpt_arr, Fseq, seq_all, y_all = assemble_training_frame()
    merged = merged.copy()
    merged["LogID"] = merged["LogID"].astype(str)
    y_all = np.asarray(y_all).astype(int)

    notes = pd.read_csv(NOTES_CSV, header=None, names=NOTES_COLS, low_memory=False,
                        dtype={"LogID": "string", "PAT_ID": "string",
                               "EncounterCSN": "string", "NoteID": "string",
                               "NoteTypeCode": "string"})
    notes["LogID"] = notes["LogID"].astype(str).str.replace(r"\.0+$", "", regex=True)
    notes = notes[notes["LogID"].isin(set(merged["LogID"]))].reset_index(drop=True)
    print(f"[t] cohort N={len(merged):,}  notes rows kept={len(notes):,}", flush=True)

    E = embed_notes(notes)
    print(f"[t] embedding shape {E.shape}", flush=True)

    # Notes-covered subgroup
    has_notes = merged["LogID"].isin(set(notes["LogID"]))
    sub_idx = np.where(has_notes.values)[0]
    print(f"[t] subgroup with notes: n={len(sub_idx):,}  base={y_all[sub_idx].mean()*100:.2f}%", flush=True)

    # per-case pools (indexed to the SUBGROUP order)
    case_ids = merged.iloc[sub_idx]["LogID"].values
    Emean   = pool_per_case(notes, E, pd.Index(case_ids), by_type=False)
    Ebytype = pool_per_case(notes, E, pd.Index(case_ids), by_type=True)
    print(f"[t] pooled per case: mean={Emean.shape}  bytype={Ebytype.shape}", flush=True)

    # PCA-32 on the mean-pool (fit will be re-fit per fold below in cv_oof_designed)
    # For simplicity we global-PCA here for the 'notes only' baseline; per-fold PCA
    # would be tighter but ~identical because notes are only weakly modeled.
    pca = PCA(n_components=min(32, Emean.shape[1])).fit(Emean)
    Emean_pca = pca.transform(Emean).astype(np.float32)

    # subgroup slices of tab inputs
    sub_feat_df = merged.iloc[sub_idx].reset_index(drop=True)
    sub_cpt = cpt_arr[sub_idx]
    y_sub = y_all[sub_idx]

    print("\n[t] subgroup CV: 12,454 notes-covered cases", flush=True)
    r_base    = cv_oof_designed(y_sub, sub_feat_df, feat_cols, sub_cpt, None,        "rf_tab_subset")
    r_mean    = cv_oof_designed(y_sub, sub_feat_df, feat_cols, sub_cpt, Emean,       "rf_tab_notes_mean")
    r_bytype  = cv_oof_designed(y_sub, sub_feat_df, feat_cols, sub_cpt, Ebytype,     "rf_tab_notes_bytype")
    # notes-only: use PCA-reduced (cheaper and typically better than raw 768-d)
    r_only    = cv_oof(Emean_pca, y_sub, "rf_notes_only_pca32")

    rows = [r_base, r_mean, r_bytype, r_only]

    # paired deltas vs baseline
    def d(row):
        return (row["auroc"] - r_base["auroc"], row["auprc"] - r_base["auprc"])

    print("\n=== SUBGROUP results (n=12,454, base 8.73%) — calibrated pooled OOF, bootstrap n=2000 ===", flush=True)
    print(f"{'model':22s}  AUROC (95% CI)         AUPRC (95% CI)         Brier    dAUROC   dAUPRC")
    for r in rows:
        dau, dap = d(r) if r is not r_base else (0.0, 0.0)
        print(f"{r['name']:22s}  {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f})  "
              f"{r['auprc']:.3f} ({r['auprc_lo']:.3f}-{r['auprc_hi']:.3f})  "
              f"{r['brier']:.4f}   {dau:+.4f}  {dap:+.4f}", flush=True)

    # full-cohort confounded sanity
    print("\n[t] confounded sanity on FULL 14,009 cohort (zero-embedding for missing cases + missing flag)", flush=True)
    Emean_full = np.zeros((len(merged), Emean.shape[1]), dtype=np.float32)
    Emean_full[sub_idx] = Emean
    miss = (~has_notes).astype(int).values.reshape(-1, 1)
    Efull = np.hstack([Emean_full, miss])
    r_full_base  = cv_oof_designed(y_all, merged, feat_cols, cpt_arr, None,  "rf_tab_full")
    r_full_notes = cv_oof_designed(y_all, merged, feat_cols, cpt_arr, Efull, "rf_tab_notes_full")
    print(f"\n[FULL COHORT sanity] rf_tab={r_full_base['auroc']:.3f}/{r_full_base['auprc']:.3f}  "
          f"rf_tab_notes={r_full_notes['auroc']:.3f}/{r_full_notes['auprc']:.3f}  "
          f"dAUROC {r_full_notes['auroc']-r_full_base['auroc']:+.4f}  "
          f"dAUPRC {r_full_notes['auprc']-r_full_base['auprc']:+.4f}   "
          f"(confounded by 'no notes = low-risk' — do not report as notes contribution)", flush=True)

    # save
    df = pd.DataFrame([{k: r[k] for k in ("name","auroc","auroc_lo","auroc_hi",
                                            "auprc","auprc_lo","auprc_hi","brier")}
                       for r in rows + [r_full_base, r_full_notes]])
    df.to_csv("artifacts/newdata/notes_embed_results.csv", index=False)
    np.savez_compressed("artifacts/newdata/notes_embed_oof.npz",
                        y_sub=y_sub, y_all=y_all, sub_idx=sub_idx,
                        oof_tab_subset=r_base["oof"],
                        oof_tab_notes_mean=r_mean["oof"],
                        oof_tab_notes_bytype=r_bytype["oof"],
                        oof_notes_only_pca32=r_only["oof"],
                        oof_tab_full=r_full_base["oof"],
                        oof_tab_notes_full=r_full_notes["oof"])
    print("\n[t] wrote artifacts/newdata/notes_embed_results.csv + notes_embed_oof.npz", flush=True)


if __name__ == "__main__":
    main()
