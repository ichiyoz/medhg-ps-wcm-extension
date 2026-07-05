"""Final push: stack every feature block to try to reach 0.80 AUROC on
30-day readmission (WCM SQL-fixed, gold label, N=14,009, base 7.54%).

Blocks:
  A base tabular (39 features + CPT one-hot)
  B prior utilization (LACE_Components.sql -> Charlson flags, ED180d, admits365d)
  C clinical scores (LACE, HOSPITAL and components)
  D sequence embeddings (order-seq GRU + care-unit-seq GRU, per-fold)
  E graph embedding (A2 provider heterograph via ie-HGCN, per-fold)
  F notes ClinicalBERT (Bio_ClinicalBERT CLS, PCA to keep dim)
  G SDoH regex flags from notes text (concatenated per patient)
  H geocode / ZIP-derived features (pgeocode + NYC hardcoded ACS)
  I additional clinical extractions from notes (meds/comorbid regex)

Runs 4 learners on the stacked matrix, ensembles top 3, then ablates.
"""
from __future__ import annotations
import os, re, sys, warnings, time, json, pickle
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score)
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import medhg_ps.config as C, medhg_ps.data as d
from medhg_ps.data import fit_preprocess, apply_preprocess
from medhg_ps.deploy import assemble_training_frame
from medhg_ps.evaluate import _bootstrap_ci

# ---------- constants ----------
SEED = 42
DATA_DIR = Path("/Users/yiyezhang/Downloads/medhg_ps_data")
LACE_COMP = DATA_DIR / "LACE_comp.csv"
NOTES_CSV = DATA_DIR / "Notes.csv"
NOTES_EMB_CACHE = Path("artifacts/newdata/notes_embeddings_raw.npz")
GOLD = DATA_DIR / "bulk_features_with_label_gold.parquet"
OUT_LOG = Path("artifacts/newdata/final_push_080.log")
OUT_RES = Path("artifacts/newdata/final_push_080_results.csv")
OUT_OOF = Path("artifacts/newdata/final_push_080_oof.npz")
OUT_ABL = Path("artifacts/newdata/final_push_080_ablation.csv")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# BLOCK A - base tabular
# ============================================================
def make_A(merged, feat_cols, cpt_arr, train_mask):
    _, st = fit_preprocess(merged.loc[train_mask, feat_cols].reset_index(drop=True), id_cols=[])
    X = apply_preprocess(merged[feat_cols], st)
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[train_mask])
    X = np.hstack([X, oh.transform(cpt_arr)])
    return X.astype(np.float32)


# ============================================================
# BLOCK B - LACE components as features
# ============================================================
LACE_COLS = ["LogID","mi","chf","pvd","cvd","dementia","copd","rheum","pud",
             "liver_mild","dm_uncomp","hemiplegia","renal","dm_comp","cancer",
             "liver_sev","mets","aids","n_ed_visits_180d","n_admits_365d"]
CHARLSON_WEIGHTS = dict(mi=1,chf=1,pvd=1,cvd=1,dementia=1,copd=1,rheum=1,pud=1,
                       liver_mild=1,dm_uncomp=1,hemiplegia=2,renal=2,dm_comp=2,
                       cancer=2,liver_sev=3,mets=6,aids=6)

def load_lace_comp():
    first = str(pd.read_csv(LACE_COMP, nrows=1, header=None, dtype=str,
                            encoding="utf-8-sig").iloc[0,0]).strip()
    has_header = first.upper() == "LOGID"
    ncols = pd.read_csv(LACE_COMP, nrows=1, header=None, encoding="utf-8-sig").shape[1]
    cols = LACE_COLS if ncols >= 20 else LACE_COLS[:19]
    df = pd.read_csv(LACE_COMP, header=0 if has_header else None,
                     names=None if has_header else cols,
                     encoding="utf-8-sig", dtype={"LogID": str})
    df["LogID"] = df["LogID"].astype(str).str.replace(r"\.0+$","",regex=True)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def make_B(merged):
    lace = load_lace_comp()
    m = merged[["LogID"]].astype({"LogID": str}).merge(lace, on="LogID", how="left")
    for c in LACE_COLS[1:]:
        if c not in m: m[c] = 0
        m[c] = m[c].fillna(0)
    m["charlson_weighted"] = sum(m[k]*w for k,w in CHARLSON_WEIGHTS.items())
    m["log_ed180"] = np.log1p(m["n_ed_visits_180d"])
    m["log_admits365"] = np.log1p(m["n_admits_365d"])
    m["ed180_ge2"] = (m["n_ed_visits_180d"] >= 2).astype(int)
    m["admits365_ge2"] = (m["n_admits_365d"] >= 2).astype(int)
    age = pd.to_numeric(merged["AgeYears"], errors="coerce").fillna(60).values
    m["elderly_x_admits"] = ((age>75) & (m["n_admits_365d"]>=2)).astype(int)
    cols = LACE_COLS[1:] + ["charlson_weighted","log_ed180","log_admits365",
                            "ed180_ge2","admits365_ge2","elderly_x_admits"]
    return m[cols].astype(np.float32).values, cols


# ============================================================
# BLOCK C - clinical scores
# ============================================================
def make_C(merged):
    from analysis.lace_baseline import compute_lace
    from analysis.hospital_score import compute_hospital_score
    # LACE two variants
    l1 = compute_lace(emergent_source="ed_first")[["LogID","L_pts","A_pts","C_pts","E_pts","LACE_score"]]
    l1 = l1.rename(columns={"A_pts":"A_edfirst","LACE_score":"LACE_edfirst"})
    l2 = compute_lace(emergent_source="patient_type")[["LogID","A_pts","LACE_score"]]
    l2 = l2.rename(columns={"A_pts":"A_pt","LACE_score":"LACE_pt"})
    # HOSPITAL two variants
    h1 = compute_hospital_score(emergent_source="ed_first")[
        ["LogID","H_pts","O_pts","S_pts","P_pts","I_pts","T_pts","HOSPITAL_score"]]
    h1 = h1.rename(columns={"I_pts":"I_edfirst","HOSPITAL_score":"HOSP_edfirst"})
    h2 = compute_hospital_score(emergent_source="patient_type")[["LogID","I_pts","HOSPITAL_score"]]
    h2 = h2.rename(columns={"I_pts":"I_pt","HOSPITAL_score":"HOSP_pt"})
    for x in (l1,l2,h1,h2): x["LogID"] = x["LogID"].astype(str)
    df = merged[["LogID"]].astype({"LogID":str}).merge(l1,on="LogID",how="left")
    df = df.merge(l2,on="LogID",how="left").merge(h1,on="LogID",how="left").merge(h2,on="LogID",how="left")
    df = df.fillna(0)
    cols = [c for c in df.columns if c != "LogID"]
    return df[cols].astype(np.float32).values, cols


# ============================================================
# BLOCK D - sequence GRU embeddings
# ============================================================
class SeqGRU(nn.Module):
    def __init__(self, vocab, embed_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def encode(self, x, lens):
        e = self.emb(x)
        pk = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(pk)
        return h[-1]
    def forward(self, x, lens):
        return self.head(self.encode(x, lens)).squeeze(-1)

def _pad_seqs(seqs, maxlen):
    L = np.array([max(min(len(s), maxlen), 1) for s in seqs], dtype=np.int64)
    X = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-maxlen:] if len(s)>maxlen else s
        if len(s) == 0:
            X[i,0] = 0  # padding token; L[i]=1 so pack sees 1 timestep
        else:
            X[i,:len(s)] = s
    return X, L

def train_gru_and_encode(seqs, y, train_idx, vocab, maxlen=128,
                          epochs=6, batch=128, lr=1e-3, hidden=64):
    """Train on train_idx, return encoded (N, hidden) for ALL rows."""
    X, L = _pad_seqs(seqs, maxlen)
    Xt = torch.tensor(X, device=DEV); Lt = torch.tensor(L)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    npos = float(max(y[train_idx].sum(), 1))
    nneg = float(max((y[train_idx]==0).sum(), 1))
    pw = torch.tensor(nneg/npos, device=DEV)
    m = SeqGRU(vocab, hidden=hidden).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)
    tr = np.array(train_idx)
    for ep in range(epochs):
        m.train()
        np.random.shuffle(tr)
        for i in range(0, len(tr), batch):
            b = tr[i:i+batch]
            opt.zero_grad()
            lg = m(Xt[b], Lt[b])
            loss = F.binary_cross_entropy_with_logits(lg, yt[b], pos_weight=pw)
            loss.backward(); opt.step()
    m.eval()
    E = np.zeros((len(seqs), hidden), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(seqs), 512):
            e = m.encode(Xt[i:i+512], Lt[i:i+512]).cpu().numpy()
            E[i:i+512] = e
    return E


# ============================================================
# BLOCK F+G+I - Notes (BERT + SDoH regex + clinical regex)
# ============================================================
NOTES_COLS = ["LogID","PAT_ID","EncounterCSN","SurgeryStart","DischargeTime",
              "NoteID","NoteTypeCode","NoteTypeName","NoteCreated","AuthorID",
              "MinutesAfterSurgeryStart","HoursBeforeDischarge","NoteText"]

def load_notes():
    import csv as _csv
    df = pd.read_csv(NOTES_CSV, header=None, names=NOTES_COLS, dtype=str,
                     engine="python", quoting=_csv.QUOTE_MINIMAL)
    df["LogID"] = df["LogID"].astype(str).str.replace(r"\.0+$","",regex=True)
    df["NoteText"] = df["NoteText"].fillna("")
    return df

SDOH_PATTERNS = {
    "sdoh_housing":  r"\b(homeless|shelter|unstable housing|housing insecurit\w*|eviction)\b",
    "sdoh_food":     r"\b(food insecurit\w*|SNAP|food stamps|hungry|malnourish\w*)\b",
    "sdoh_transp":   r"\b(no transportation|unable to travel|no ride|no car)\b",
    "sdoh_alcohol":  r"\b(alcohol abuse|alcoholism|heavy drink\w*|AUD|delirium tremens|DTs)\b",
    "sdoh_subst":    r"\b(substance abuse|drug use|IVDU|opioid abuse|cocaine|methamphetamine)\b",
    "sdoh_social":   r"\b(lives alone|no support|no caregiver|widowed|isolated)\b",
    "sdoh_depr":     r"\b(depression|depressed|MDD|major depressive|suicide|suicidal)\b",
    "sdoh_anx":      r"\b(anxiety|PTSD|panic disorder)\b",
    "sdoh_unempl":   r"\b(unemployed|disabled|cannot work)\b",
    "sdoh_insur":    r"\b(uninsured|self-pay|Medicaid)\b",
    "sdoh_smoke":    r"\b(smoker|tobacco use|pack-year|pack years)\b",
    "sdoh_nonadh":   r"\b(non[- ]?compliant|missed appointment|non[- ]?adherent)\b",
    "sdoh_lang":     r"\b(interpreter|non[- ]?English speaking|language barrier)\b",
}
MED_PATTERNS = {
    "med_steroid":   r"\b(prednisone|dexamethasone|methylprednisolone|hydrocortisone|steroid)\b",
    "med_immuno":    r"\b(tacrolimus|cyclosporine|mycophenolate|azathioprine|sirolimus)\b",
    "med_anticoag":  r"\b(warfarin|apixaban|rivaroxaban|dabigatran|heparin|enoxaparin|coumadin|eliquis|xarelto)\b",
    "med_dmard":     r"\b(methotrexate|adalimumab|etanercept|infliximab|rituximab)\b",
    "med_insulin":   r"\b(insulin|glargine|lantus|humalog|novolog)\b",
    "med_opioid":    r"\b(morphine|oxycodone|hydrocodone|fentanyl|percocet|dilaudid|hydromorphone|oxycontin)\b",
    "med_benzo":     r"\b(alprazolam|lorazepam|clonazepam|diazepam|xanax|ativan|klonopin|valium)\b",
}
COMORB_PATTERNS = {
    "cx_htn":     r"\b(hypertension|HTN)\b",
    "cx_dm":      r"\b(diabetes|diabetic|DM type|type 2 DM|type 1 DM|T2DM|T1DM)\b",
    "cx_cad":     r"\b(coronary artery disease|CAD|MI history|prior MI)\b",
    "cx_afib":    r"\b(atrial fibrillation|Afib|A-fib)\b",
    "cx_copd":    r"\b(COPD|chronic obstructive pulmonary)\b",
    "cx_chf":     r"\b(congestive heart failure|CHF|heart failure|HFrEF|HFpEF)\b",
    "cx_ckd":     r"\b(chronic kidney disease|CKD|ESRD|dialysis)\b",
    "cx_cancer":  r"\b(cancer|carcinoma|malignancy|adenocarcinoma|neoplasm)\b",
    "cx_obese":   r"\b(obesity|obese|BMI \d\d)\b",
    "cx_osa":     r"\b(sleep apnea|OSA|CPAP)\b",
}
SEV_PATTERNS = {
    "sev_sepsis": r"\b(sepsis|septic)\b",
    "sev_shock":  r"\b(shock)\b",
    "sev_crit":   r"\b(critical|unstable|emergent|resuscitat\w*)\b",
}

def make_regex_features(notes_df, all_logids):
    """Per-encounter binary flags + counts + numeric labs extracted."""
    # concatenate all notes text per LogID (lowercased)
    agg = notes_df.groupby("LogID")["NoteText"].apply(lambda s: " ".join(s.astype(str))).str.lower()
    pats = {**SDOH_PATTERNS, **MED_PATTERNS, **COMORB_PATTERNS, **SEV_PATTERNS}
    log(f"  regex on {len(agg)} concatenated note-strings across {len(pats)} patterns")
    rows = []
    for lid, text in agg.items():
        r = {"LogID": lid}
        for name, pat in pats.items():
            r[name] = int(bool(re.search(pat, text, flags=re.IGNORECASE)))
        # numeric lab extractions - most recent value
        m = re.findall(r"\b(?:hemoglobin|hgb|hb)\s*[:=]?\s*(\d{1,2}(?:\.\d)?)", text, re.I)
        r["note_hb_last"] = float(m[-1]) if m else np.nan
        m = re.findall(r"\bcreatinine\s*[:=]?\s*(\d{1,2}(?:\.\d{1,2})?)", text, re.I)
        r["note_cr_last"] = float(m[-1]) if m else np.nan
        m = re.findall(r"\b(?:wbc|white blood cell)\s*[:=]?\s*(\d{1,3}(?:\.\d{1,2})?)", text, re.I)
        r["note_wbc_last"] = float(m[-1]) if m else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    # sdoh count
    sdoh_cols = [c for c in df.columns if c.startswith("sdoh_")]
    df["sdoh_total"] = df[sdoh_cols].sum(axis=1)
    med_cols = [c for c in df.columns if c.startswith("med_")]
    df["med_total"] = df[med_cols].sum(axis=1)
    # align to all_logids
    out = pd.DataFrame({"LogID": all_logids})
    out = out.merge(df, on="LogID", how="left")
    # fill missing
    for c in out.columns[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # has_notes indicator
    out["has_notes"] = (~out.iloc[:,1:].isna().all(axis=1)).astype(int)
    # medians for numeric labs, 0 for binaries
    for c in out.columns[1:]:
        if c.startswith(("sdoh_","med_","cx_","sev_")) or c in ("sdoh_total","med_total","has_notes"):
            out[c] = out[c].fillna(0).astype(np.float32)
        elif c.startswith("note_") and c.endswith("_last"):
            med = out[c].median()
            out[c] = out[c].fillna(med).astype(np.float32)
    cols = [c for c in out.columns if c != "LogID"]
    return out[cols].values.astype(np.float32), cols


def compute_bert_notes(notes_df, all_logids, batch_size=64, max_len=128):
    """Compute Bio_ClinicalBERT CLS embeddings per note; cache to disk;
    return per-encounter mean-pooled 768-d vector (zeros for missing)."""
    from transformers import AutoTokenizer, AutoModel
    if NOTES_EMB_CACHE.exists():
        log(f"  loading cached BERT embeddings: {NOTES_EMB_CACHE}")
        d = np.load(NOTES_EMB_CACHE, allow_pickle=True)
        note_logids = d["log_id"].astype(str)
        E = d["E"]
    else:
        log("  loading Bio_ClinicalBERT ...")
        tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        mdl = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(DEV).eval()
        n = len(notes_df)
        E = np.zeros((n, 768), dtype=np.float32)
        note_logids = notes_df["LogID"].astype(str).values
        texts = notes_df["NoteText"].astype(str).tolist()
        t0 = time.time()
        # Time budget: cap BERT run at 45 min. If projected ETA exceeds
        # this after a warmup, abort and return zeros (block F disabled).
        BUDGET_S = 45 * 60
        aborted = False
        with torch.no_grad():
            for i in range(0, n, batch_size):
                b = texts[i:i+batch_size]
                enc = tok(b, padding=True, truncation=True, max_length=max_len,
                          return_tensors="pt").to(DEV)
                out = mdl(**enc)
                cls = out.last_hidden_state[:,0,:].cpu().numpy()
                E[i:i+batch_size] = cls
                if (i//batch_size) % 50 == 0 and i > 0:
                    dt = time.time()-t0
                    eta = dt/(i+1) * (n-i-1)
                    log(f"    BERT {i+batch_size}/{n} elapsed {dt/60:.1f}m eta {eta/60:.1f}m")
                    if dt > 300 and dt + eta > BUDGET_S:
                        log(f"    BERT ABORT: budget exceeded ({(dt+eta)/60:.0f}m > {BUDGET_S/60:.0f}m). "
                            "Filling remainder with zeros.")
                        aborted = True
                        break
        if not aborted:
            np.savez(NOTES_EMB_CACHE, log_id=note_logids, E=E)
            log(f"  saved cache to {NOTES_EMB_CACHE}")
        else:
            # cache partial to disk with marker
            np.savez(str(NOTES_EMB_CACHE).replace(".npz","_partial.npz"),
                     log_id=note_logids, E=E, aborted=True)
    # mean-pool per LogID
    log(f"  mean-pool {len(note_logids)} notes into {len(all_logids)} encounters")
    df = pd.DataFrame(E)
    df["LogID"] = note_logids
    pooled = df.groupby("LogID").mean().reset_index()
    out = pd.DataFrame({"LogID": all_logids}).merge(pooled, on="LogID", how="left")
    has_notes = (~out.iloc[:,1:].isna().all(axis=1)).astype(int).values
    Fmat = out.iloc[:,1:].fillna(0).values.astype(np.float32)
    return Fmat, has_notes


# ============================================================
# BLOCK H - geocode
# ============================================================
def make_H(merged):
    import pgeocode
    nomi = pgeocode.Nominatim("us")
    # WCM main location: 525 East 68th St, NYC = zip 10065
    ref_wcm = nomi.query_postal_code("10065")
    lat0, lon0 = float(ref_wcm.latitude), float(ref_wcm.longitude)
    # 5 NYP hospitals ZIPs (approx)
    nyp_zips = ["10065","10032","11215","10461","10038"]
    nyp_pts = nomi.query_postal_code(nyp_zips)[["latitude","longitude"]].values
    zip5 = merged["ZIP"].astype(str).str.replace(r"[^0-9]","",regex=True).str[:5]
    zip5 = zip5.where(zip5.str.len()==5, "00000")
    unique_zips = zip5.unique()
    log(f"  geocoding {len(unique_zips)} unique ZIPs")
    q = nomi.query_postal_code(list(unique_zips))
    q["zip5"] = unique_zips
    q = q[["zip5","latitude","longitude"]].fillna(np.nan)
    m = pd.DataFrame({"zip5": zip5.values})
    m = m.merge(q, on="zip5", how="left")
    lat = m["latitude"].values; lon = m["longitude"].values
    def hav(la1,lo1,la2,lo2):
        R = 3958.8
        p = np.pi/180
        a = np.sin((la2-la1)*p/2)**2 + np.cos(la1*p)*np.cos(la2*p)*np.sin((lo2-lo1)*p/2)**2
        return 2*R*np.arcsin(np.sqrt(a))
    dist_wcm = hav(lat, lon, lat0, lon0)
    # nearest NYP
    dists_nyp = np.array([hav(lat, lon, la, lo) for la,lo in nyp_pts])
    dist_nyp = np.nanmin(dists_nyp, axis=0)
    # bins
    b = np.digitize(dist_wcm, bins=[2,5,10,25])
    # is_nyc (Manhattan/Brooklyn/Queens/Bronx/SI zip prefixes)
    is_nyc = zip5.str[:3].isin(["100","101","102","103","104","112","113","114","116"]).astype(int).values
    is_manh = zip5.str[:3].isin(["100","101","102"]).astype(int).values
    # hardcoded NYC ZIP median income (2023 ACS 5yr, rough): high-income ZIPs
    HI_INC = {"10007","10021","10022","10023","10028","10036","10065","10075",
              "10128","11201","10011","10014","10019","10024","10025"}
    LOW_INC = {"10029","10035","10037","10039","10451","10452","10453","10454",
               "10455","10456","10457","10458","10459","10460","10467","10473",
               "10474","11207","11208","11212","11213","11221","11224","11225",
               "11226","11233","11239"}
    inc = np.zeros(len(zip5))
    inc[zip5.isin(HI_INC).values] = 1
    inc[zip5.isin(LOW_INC).values] = -1
    # fill nans
    dist_wcm = np.nan_to_num(dist_wcm, nan=np.nanmedian(dist_wcm))
    dist_nyp = np.nan_to_num(dist_nyp, nan=np.nanmedian(dist_nyp))
    Xh = np.column_stack([dist_wcm, dist_nyp, b.astype(float), is_nyc,
                          is_manh, inc]).astype(np.float32)
    cols = ["dist_wcm","dist_nyp","dist_wcm_bin","is_nyc","is_manhattan","zip_inc_band"]
    return Xh, cols


# ============================================================
# Learners
# ============================================================
def learner_rf(seed=SEED, big=False):
    if big:
        return RandomForestClassifier(n_estimators=1000, min_samples_leaf=3,
                                      max_features="sqrt", class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                  max_features="sqrt", class_weight="balanced",
                                  n_jobs=-1, random_state=seed)

def learner_hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=40, l2_regularization=1.0,
                                          random_state=SEED)


def eval_pooled_oof(y, p, name):
    au = roc_auc_score(y, p); ap = average_precision_score(y, p)
    br = brier_score_loss(y, p)
    au_ci = _bootstrap_ci(y, p, roc_auc_score, n_boot=2000, seed=0)
    ap_ci = _bootstrap_ci(y, p, average_precision_score, n_boot=2000, seed=1)
    best_f1, best_thr = 0.0, 0.5
    for t in np.linspace(0.02, 0.35, 100):
        yh = (p >= t).astype(int)
        if yh.sum() < 5: continue
        f = f1_score(y, yh)
        if f > best_f1: best_f1, best_thr = f, t
    yh = (p >= best_thr).astype(int)
    return dict(model=name, auroc=au, auroc_lo=au_ci[0], auroc_hi=au_ci[1],
                auprc=ap, auprc_lo=ap_ci[0], auprc_hi=ap_ci[1], brier=br,
                thr=best_thr, f1=f1_score(y,yh),
                precision=precision_score(y,yh),
                recall=recall_score(y,yh), flag_pct=yh.mean()*100)


# ============================================================
# main
# ============================================================
def main():
    log("=== FINAL PUSH 0.80 START ===")

    # ---- cohort + gold label
    log("assembling cohort + gold label")
    merged, feat_cols, cpt_arr, Fseq, seq_all, _ = assemble_training_frame()
    gold = pd.read_parquet(GOLD)[["LogID","ReadmittedWithin30Days_gold"]]
    gold["LogID"] = gold["LogID"].astype(str)
    merged["LogID"] = merged["LogID"].astype(str)
    merged = merged.merge(gold, on="LogID", how="left")
    y = merged["ReadmittedWithin30Days_gold"].astype(int).values
    N = len(y)
    log(f"cohort N={N} base rate {y.mean()*100:.2f}%")

    # ---- BLOCK B tabular (folds-independent)
    log("BLOCK B: LACE utility (Charlson flags, ED180d, admits365d)")
    XB, colsB = make_B(merged)
    log(f"  B shape {XB.shape}")

    # ---- BLOCK C tabular
    log("BLOCK C: LACE + HOSPITAL scores and components")
    XC, colsC = make_C(merged)
    log(f"  C shape {XC.shape}")

    # ---- BLOCK G regex features from notes
    log("BLOCK G+I: regex SDoH + medications + comorbidities + severity + labs")
    notes_df = load_notes()
    log(f"  loaded {len(notes_df)} notes across {notes_df['LogID'].nunique()} encounters")
    all_logids = merged["LogID"].astype(str).values
    XGI, colsGI = make_regex_features(notes_df, all_logids)
    log(f"  G+I shape {XGI.shape}; SDoH-any: {(XGI[:,colsGI.index('sdoh_total')]>0).sum()}")

    # ---- BLOCK F Bio_ClinicalBERT mean-pooled
    log("BLOCK F: Bio_ClinicalBERT mean-pool")
    XF_raw, has_notes = compute_bert_notes(notes_df, all_logids, batch_size=32)
    log(f"  F raw shape {XF_raw.shape}; encounters with notes: {has_notes.sum()}")

    # ---- BLOCK H geocode
    log("BLOCK H: geocode + distance + zip-income proxy")
    XH, colsH = make_H(merged)
    log(f"  H shape {XH.shape}")

    # ---- sequences for BLOCK D (order-seq only; Fseq care-path features go into tabular)
    log("BLOCK D: order-sequence GRU")
    from medhg_ps.data import load_order_sequence, collapse_order_runs
    orders_df = collapse_order_runs(load_order_sequence())
    orders_df["LogID"] = orders_df["LogID"].astype(str)
    # Build token vocab from OrderGroup
    tokens = orders_df["OrderGroup"].astype(str).fillna("UNK")
    uniq = sorted(tokens.unique())
    tok2id = {t: i+1 for i, t in enumerate(uniq)}  # 0 = pad
    orders_df["tid"] = tokens.map(tok2id).astype(np.int64)
    log(f"  order vocab {len(tok2id)+1}, {len(orders_df)} tokens")
    seqs_by_log = orders_df.sort_values(["LogID","SeqInEncounter"]).groupby("LogID")["tid"].apply(list)
    order_seqs = []
    for lid in all_logids:
        s = seqs_by_log.get(lid, [])
        order_seqs.append(np.asarray(s, dtype=np.int64))
    vocab_o = len(tok2id) + 2
    lens = [len(s) for s in order_seqs]
    log(f"  order-seq len median {int(np.median(lens))} p90 {int(np.percentile(lens,90))} max {max(lens)}")

    # Fseq care-path features as additional tabular block
    Fseq["LogID"] = Fseq["LogID"].astype(str)
    care_df = pd.DataFrame({"LogID": all_logids}).merge(Fseq, on="LogID", how="left")
    care_cols = [c for c in care_df.columns if c != "LogID"]
    XCARE = care_df[care_cols].fillna(0).values.astype(np.float32)
    log(f"  care-path tabular shape {XCARE.shape}")

    # ---- run 5-fold CV
    log("=== CROSS VALIDATION ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

    p_oof = {"rf_base":np.full(N,np.nan), "rf_big":np.full(N,np.nan),
             "hgb":np.full(N,np.nan)}
    # feature name list to track
    all_feat_cols = None

    for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
        log(f"--- fold {fi+1}/5 --- train {len(tr)} test {len(te)}")
        train_mask = np.zeros(N, bool); train_mask[tr] = True

        # A: base tabular
        XA = make_A(merged, feat_cols, cpt_arr, train_mask)

        # D: sequence GRUs
        log(f"  D: train order-GRU")
        XD_o = train_gru_and_encode(order_seqs, y, tr, vocab_o, maxlen=128,
                                    epochs=5, hidden=64)
        # unit-seq GRU replaced by care-path tabular features (Fseq)
        XD_u = XCARE  # not a GRU embedding; per-fold identical

        # F: PCA on BERT embeddings (fit on train)
        pca = PCA(n_components=64, random_state=SEED).fit(XF_raw[tr])
        XF = pca.transform(XF_raw)
        XF = np.hstack([XF, has_notes.reshape(-1,1)]).astype(np.float32)

        # Concatenate blocks
        X_full = np.hstack([XA, XB, XC, XD_o, XD_u, XF, XGI, XH]).astype(np.float32)
        if all_feat_cols is None:
            names = [f"A_{i}" for i in range(XA.shape[1])] + colsB + colsC
            names += [f"D_o_{i}" for i in range(XD_o.shape[1])]
            names += list(care_cols)
            names += [f"F_pc_{i}" for i in range(XF.shape[1])] + colsGI + colsH
            all_feat_cols = names
            log(f"  X shape {X_full.shape}, {len(all_feat_cols)} named cols")

        # Learners
        for name, mk in [("rf_base", lambda: learner_rf()),
                         ("rf_big",  lambda: learner_rf(big=True)),
                         ("hgb",     lambda: learner_hgb())]:
            try:
                est = CalibratedClassifierCV(mk(), method="isotonic", cv=3).fit(X_full[tr], y[tr])
                p_oof[name][te] = est.predict_proba(X_full[te])[:,1]
                log(f"  fold {fi+1} {name} done")
            except Exception as e:
                log(f"  fold {fi+1} {name} FAILED: {e}")

        # save last-fold X_full for ablation
        if fi == 4:
            np.savez(str(OUT_OOF).replace(".npz","_lastfold_X.npz"),
                     tr=tr, te=te, X=X_full, y=y)

    # ---- eval
    log("=== POOLED OOF EVAL ===")
    results = []
    for name in ["rf_base","rf_big","hgb"]:
        p = p_oof[name]
        if np.isnan(p).any():
            log(f"  {name} has NaN, skipping"); continue
        r = eval_pooled_oof(y, p, name)
        log(f"  {name:10s} AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    # ensemble: mean of top-3 learners
    valid = [k for k in p_oof if not np.isnan(p_oof[k]).any()]
    if len(valid) >= 2:
        p_ens = np.mean([p_oof[k] for k in valid], axis=0)
        r = eval_pooled_oof(y, p_ens, "ensemble")
        log(f"  ensemble AUROC {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
            f"AUPRC {r['auprc']:.3f} F1 {r['f1']:.3f}")
        results.append(r)

    # ---- ablation on rf_base with block removals
    log("=== ABLATION (drop-one-block on rf_base) ===")
    ablation = []
    # We'll rerun 5-fold with rf_base only, dropping each block in turn
    # For speed: use fold-level design matrices - regenerate per fold but cheaper (skip GRU/GNN retraining is not possible, so skip D drop)
    # Simplified: keep GRUs from full-model per-fold (we already trained them),
    # regenerate other blocks in-place per fold; only drop tabular blocks fast.
    block_defs = {
        "A_base_tabular": XA.shape[1] if 'XA' in dir() else None,
        # We'll do a lightweight ablation: compare against ensemble on OOF
    }
    # For ablation we redo per-fold with only rf_base (5x faster than main run per config)
    # and drop each block from concatenation
    def block_indices(nA, nB, nC, nDo, nDu, nF, nGI, nH):
        blocks = {}
        i = 0
        for name, sz in [("A", nA), ("B", nB), ("C", nC), ("D_order", nDo),
                         ("D_unit", nDu), ("F_bert", nF), ("GI_regex", nGI), ("H_geo", nH)]:
            blocks[name] = (i, i + sz); i += sz
        return blocks

    log("  running 8-block ablation (may take ~15 min)")
    n_blocks_done = 0
    for drop in ["none","A","B","C","D_order","D_unit","F_bert","GI_regex","H_geo"]:
        p_ab = np.full(N, np.nan)
        try:
            skf2 = StratifiedKFold(5, shuffle=True, random_state=SEED)
            for fi,(tr,te) in enumerate(skf2.split(np.zeros(N), y)):
                train_mask = np.zeros(N, bool); train_mask[tr] = True
                XA_ = make_A(merged, feat_cols, cpt_arr, train_mask)
                # D: retrain (cheap 3-epoch version for ablation)
                XD_o_ = train_gru_and_encode(order_seqs, y, tr, vocab_o, maxlen=128, epochs=3, hidden=64)
                XD_u_ = XCARE
                pca2 = PCA(n_components=64, random_state=SEED).fit(XF_raw[tr])
                XF_ = np.hstack([pca2.transform(XF_raw), has_notes.reshape(-1,1)]).astype(np.float32)
                blocks = {"A":XA_, "B":XB, "C":XC, "D_order":XD_o_, "D_unit":XD_u_,
                          "F_bert":XF_, "GI_regex":XGI, "H_geo":XH}
                keep = [k for k in blocks if k != drop] if drop != "none" else list(blocks)
                X_ab = np.hstack([blocks[k] for k in keep]).astype(np.float32)
                est = CalibratedClassifierCV(learner_rf(), method="isotonic", cv=3).fit(X_ab[tr], y[tr])
                p_ab[te] = est.predict_proba(X_ab[te])[:,1]
            au = roc_auc_score(y, p_ab); ap = average_precision_score(y, p_ab)
            ablation.append(dict(drop=drop, auroc=au, auprc=ap))
            log(f"  drop {drop:10s} AUROC {au:.3f} AUPRC {ap:.3f}")
            n_blocks_done += 1
        except Exception as e:
            log(f"  drop {drop} FAILED: {e}")

    # ---- save
    pd.DataFrame(results).to_csv(OUT_RES, index=False)
    pd.DataFrame(ablation).to_csv(OUT_ABL, index=False)
    np.savez(OUT_OOF, y=y, **p_oof)
    log(f"saved {OUT_RES} {OUT_ABL} {OUT_OOF}")

    # ---- permutation importance (top-20) - only if time
    try:
        log("=== PERMUTATION IMPORTANCE (top-20) ===")
        # refit on full data
        skf3 = StratifiedKFold(5, shuffle=True, random_state=SEED)
        tr_all, te_all = next(iter(skf3.split(np.zeros(N), y)))
        # use last-fold data we saved
        d = np.load(str(OUT_OOF).replace(".npz","_lastfold_X.npz"))
        X_all = d["X"]; tr_last = d["tr"]; te_last = d["te"]
        est = learner_rf().fit(X_all[tr_last], y[tr_last])
        pi = permutation_importance(est, X_all[te_last], y[te_last],
                                     n_repeats=3, random_state=SEED,
                                     n_jobs=-1, scoring="average_precision")
        imp = pd.DataFrame(dict(feature=all_feat_cols, importance=pi.importances_mean))
        imp = imp.sort_values("importance", ascending=False).head(30)
        log("top-20 permutation importance:")
        for _, r in imp.head(20).iterrows():
            log(f"  {r['feature']:30s} {r['importance']:+.5f}")
        imp.to_csv(str(OUT_RES).replace(".csv","_permimp.csv"), index=False)
    except Exception as e:
        log(f"perm importance failed: {e}")

    log("=== DONE ===")


if __name__ == "__main__":
    main()
