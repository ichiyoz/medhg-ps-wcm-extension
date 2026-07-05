"""Kitchen-sink GNN sweep: GraphSAGE / GAT / R-GCN / HGT, feature-selected nodes,
nested TPE inside each CV fold. (Multi-hour run.)

Node graph: encounters + providers + units, bidirectional encounter-provider and
encounter-unit edges, with node-type and edge-type ids. Encounter node features
= top-50 (RF-importance, fold-honest) of the enriched feature set (tabular + CPT
+ post-op care-unit sequence); provider/unit nodes get zero features.

Per OUTER fold (5): an Optuna TPE search (N_TRIALS) over {hidden, n_layers, lr,
dropout} maximizes INNER-validation AUROC; the selected config is retrained and
scored on the untouched outer test. Compared against the enriched tree baseline.

Robust: each architecture is wrapped so a failure does not kill the others;
results print incrementally. Set MEDHG_SMOKE=1 for a tiny validation run.

    PYTHONPATH=. python analysis/cv_gnn_zoo.py
"""
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import dgl
from dgl.nn import SAGEConv, GATConv, RelGraphConv, HGTConv
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import medhg_ps.config as C
from medhg_ps.data import (load_raw, fit_preprocess, apply_preprocess,
                           build_preop_trajectory_features, add_calendar_features)
from medhg_ps.train import set_seed, _resolve_device

optuna.logging.set_verbosity(optuna.logging.WARNING)
SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
K, SEED, VAL_FRAC = (2 if SMOKE else 5), 42, 0.10
N_TRIALS = 2 if SMOKE else 8
MAX_EPOCHS, PATIENCE = (25, 6) if SMOKE else (120, 12)
TOPK = 50
ARCHS = ["sage", "gat", "rgcn", "hgt"]
UNITS = ["ED", "Acute", "OR", "Intensive", "Intermediate", "Other"]
MIN_SUP = 50
COHORT_COLS = ["LogID", "EncounterCSN", "PAT_ID", "SurgeryDate", "SurgeryYear", "PrimaryCPT", "AgeYears"]
dev = _resolve_device(C.DEFAULTS_TRAIN.device)


def _norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def load_cpt():
    for d in [str(C.DATA_DIR), "/Users/yiyezhang/Dropbox/Surgery"]:
        for p in glob.glob(os.path.join(d, "*.csv")):
            try:
                h = pd.read_csv(p, nrows=1, header=None, dtype=str)
            except Exception:
                continue
            if str(h.iloc[0, 0]).strip() == "LogID":
                df = pd.read_csv(p, low_memory=False)
                if {"LogID", "PrimaryCPT"} <= set(df.columns):
                    return dict(zip(_norm(df["LogID"]), _norm(df["PrimaryCPT"])))
            elif h.shape[1] == len(COHORT_COLS):
                df = pd.read_csv(p, header=None, names=COHORT_COLS, low_memory=False)
                return dict(zip(_norm(df["LogID"]), _norm(df["PrimaryCPT"])))
    raise FileNotFoundError("cohort CSV with PrimaryCPT not found")


print(f"[zoo] {'SMOKE ' if SMOKE else ''}loading...", flush=True)
cpt_map = load_cpt()
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
row_of = {lid: i for i, lid in enumerate(merged["LogID"])}
cpt_arr = np.array([cpt_map.get(l, "UNK") for l in merged["LogID"]]).reshape(-1, 1)

# sequence features
gnnb = {"ICU": "Intensive", "PICU": "Intensive", "NICU": "Intensive", "Med/Surg": "Acute",
        "Procedural Area": "OR", "ED": "ED", "Recovery Area": "Intermediate"}
u = pd.read_excel(Path(__file__).parent / "Unit_Names.xlsx")
u["cid"] = u["Clarity_ID"].astype("Int64").astype(str); u["g"] = u["UnitType"].map(gnnb).fillna("Other")
ud = u.dropna(subset=["Clarity_ID"]).drop_duplicates("cid", keep="first").set_index("cid")
a3 = raw.enc_unit_edges.copy()
a3["UnitType"] = (a3["DepartmentID"].astype(str).str.replace(r"\.0+$", "", regex=True).map(ud["g"]).fillna(a3["UnitType"]))
a3["InTime"] = pd.to_datetime(a3["InTime"]); a3["LogID"] = a3["LogID"].astype(str)
seqs = a3.sort_values(["LogID", "InTime"]).groupby("LogID", sort=False)["UnitType"].apply(list)


def collapse(s):
    o = []
    for x in s:
        if not o or o[-1] != x:
            o.append(x)
    return o


rows = []
for lid, rs in seqs.items():
    c = collapse(rs); f = {"LogID": lid}
    for un in UNITS:
        f[f"cnt_{un}"] = rs.count(un)
    f["n_stops"] = len(c); f["has_ICU"] = int("Intensive" in c)
    f["ICU_after_OR"] = int("Intensive" in c and "OR" in c and c.index("Intensive") > c.index("OR"))
    f[f"end_{c[-1]}"] = 1
    for a, b in zip(c, c[1:]):
        f[f"bg_{a}>{b}"] = f.get(f"bg_{a}>{b}", 0) + 1
    rows.append(f)
Fseq = merged[["LogID"]].merge(pd.DataFrame(rows).fillna(0).astype({"LogID": str}), on="LogID", how="left").fillna(0)
seq_all = [c for c in Fseq.columns if c != "LogID"]

# --- node graph (encounters 0..N, providers, units) ------------------
P_ids = sorted({p for p in _norm(raw.enc_prov_edges["ProvID"]) if p and p != "nan"})
U_ids = sorted({d for d in _norm(raw.enc_unit_edges["DepartmentID"]) if d and d != "nan"})
pidx = {p: N + i for i, p in enumerate(P_ids)}
uidx = {d: N + len(P_ids) + i for i, d in enumerate(U_ids)}
TOT = N + len(P_ids) + len(U_ids)
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
g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=TOT).to(dev)
g = dgl.add_self_loop(g)                                 # also extend etype for self-loops
etype = torch.cat([torch.tensor(et, dtype=torch.long),
                   torch.full((TOT,), 4, dtype=torch.long)]).to(dev)   # 5 rel types
ntype = torch.zeros(TOT, dtype=torch.long, device=dev)
ntype[N:N + len(P_ids)] = 1; ntype[N + len(P_ids):] = 2
NREL = 5
print(f"[zoo] cohort={N:,} | nodes={TOT:,} (P={len(P_ids)},U={len(U_ids)}) edges={g.num_edges():,}", flush=True)


class Net(nn.Module):
    def __init__(self, arch, ind, hid, nl, dp):
        super().__init__()
        self.arch = arch; self.drop = nn.Dropout(dp); self.layers = nn.ModuleList()
        dims = [ind] + [hid] * nl
        for i in range(nl):
            di, do = dims[i], dims[i + 1]
            if arch == "sage":
                self.layers.append(SAGEConv(di, do, "mean"))
            elif arch == "gat":
                self.layers.append(GATConv(di, do, num_heads=4))
            elif arch == "rgcn":
                self.layers.append(RelGraphConv(di, do, NREL, regularizer="basis", num_bases=4))
            elif arch == "hgt":
                self.layers.append(HGTConv(di, do // 4, 4, 3, NREL, dropout=dp))
        self.cls = nn.Linear(hid, 2)

    def forward(self, gg, x):
        h = x
        for conv in self.layers:
            if self.arch == "gat":
                h = conv(gg, h).mean(1)
            elif self.arch == "rgcn":
                h = conv(gg, h, etype)
            elif self.arch == "hgt":
                h = conv(gg, h, ntype, etype)
            else:
                h = conv(gg, h)
            h = self.drop(Fn.relu(h))
        return self.cls(h)


def train_eval(arch, Xnodes, yt, trm, vam, hp, return_test=None):
    set_seed(SEED)
    net = Net(arch, Xnodes.shape[1], hp["hid"], hp["nl"], hp["dp"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=1e-4)
    npos = float((yt[trm] == 1).sum()); nneg = float((yt[trm] == 0).sum())
    w = torch.tensor([(npos+nneg)/(2*max(nneg, 1)), (npos+nneg)/(2*max(npos, 1))], device=dev)
    lf = nn.CrossEntropyLoss(weight=w)
    best, state, pat = -1.0, None, PATIENCE
    yv = yt[vam].cpu().numpy(); vm = vam.cpu().numpy()
    for ep in range(MAX_EPOCHS):
        net.train()
        logit = net(g, Xnodes)[:N]
        loss = lf(logit[trm], yt[trm]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.softmax(net(g, Xnodes)[:N], -1)[:, 1].cpu().numpy()
        try: va = roc_auc_score(yv, p[vm])
        except ValueError: va = 0.5
        if va > best:
            best, state, pat = va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, PATIENCE
        else:
            pat -= 1
            if pat <= 0: break
    if return_test is not None:
        net.load_state_dict(state); net.eval()
        with torch.no_grad():
            p = torch.softmax(net(g, Xnodes)[:N], -1)[:, 1].cpu().numpy()
        te = return_test
        return roc_auc_score(y[te], p[te]), average_precision_score(y[te], p[te])
    return best


def build_nodefeat(tr):
    Xtab_tr, st = fit_preprocess(merged[feat_cols].loc[np.isin(np.arange(N), tr)].reset_index(drop=True), id_cols=[])
    Xtab = apply_preprocess(merged[feat_cols], st)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cpt_arr[tr])
    Xcpt = ohe.transform(cpt_arr)
    keep = [c for c in seq_all if (Fseq.iloc[tr][c] > 0).sum() >= MIN_SUP]
    Xseq = Fseq[keep].values.astype(float)
    Xall = np.hstack([Xtab, Xcpt, Xseq])
    sel = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                                 class_weight="balanced", random_state=42, n_jobs=-1).fit(Xall[tr], y[tr])
    top = np.argsort(sel.feature_importances_)[::-1][:TOPK]
    Xsel = StandardScaler().fit(Xall[tr][:, top]).transform(Xall[:, top])
    Xnodes = np.zeros((TOT, TOPK), dtype=np.float32); Xnodes[:N] = Xsel
    return torch.tensor(Xnodes, device=dev), Xall, top


skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
yt = torch.tensor(y, dtype=torch.long, device=dev)
results = {a: {"au": [], "ap": []} for a in ARCHS}
tree = {"au": [], "ap": []}
for fi, (tr_all, te) in enumerate(skf.split(np.zeros(N), y)):
    rng = np.random.default_rng(SEED + fi); tr_all = tr_all.copy(); rng.shuffle(tr_all)
    nv = int(round(VAL_FRAC * N)); va, tr = tr_all[:nv], tr_all[nv:]
    trm = torch.zeros(N, dtype=torch.bool, device=dev); trm[tr] = True
    vam = torch.zeros(N, dtype=torch.bool, device=dev); vam[va] = True
    Xnodes, Xall, _ = build_nodefeat(tr)
    hg = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                                        random_state=42).fit(Xall[tr], y[tr])
    ph = hg.predict_proba(Xall[te])[:, 1]
    tree["au"].append(roc_auc_score(y[te], ph)); tree["ap"].append(average_precision_score(y[te], ph))
    print(f"[zoo] fold {fi+1}/{K}  tree(enriched) AUROC={tree['au'][-1]:.3f}", flush=True)

    for arch in ARCHS:
        try:
            def obj(trial):
                hp = dict(hid=trial.suggest_categorical("hid", [32, 64]),
                          nl=trial.suggest_int("nl", 1, 2),
                          lr=trial.suggest_float("lr", 1e-3, 3e-2, log=True),
                          dp=trial.suggest_float("dp", 0.1, 0.5))
                return train_eval(arch, Xnodes, yt, trm, vam, hp)
            study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
            study.optimize(obj, n_trials=N_TRIALS)
            bp = study.best_params; hp = dict(hid=bp["hid"], nl=bp["nl"], lr=bp["lr"], dp=bp["dp"])
            au, ap = train_eval(arch, Xnodes, yt, trm, vam, hp, return_test=te)
            results[arch]["au"].append(au); results[arch]["ap"].append(ap)
            print(f"[zoo] fold {fi+1} {arch:5s} AUROC={au:.3f} AUPRC={ap:.3f} "
                  f"(val {study.best_value:.3f}, hp={bp})", flush=True)
        except Exception as e:
            print(f"[zoo] fold {fi+1} {arch:5s} FAILED: {type(e).__name__}: {e}", flush=True)

print(f"\n=== Nested-CV results ({K}-fold, TPE {N_TRIALS}/fold) | base {y.mean()*100:.2f}% ===")
bt = np.array(tree["au"])
print(f"  {'tree(enriched)':16s} AUROC {bt.mean():.3f} +/- {bt.std():.3f}  AUPRC {np.mean(tree['ap']):.3f}")
for a in ARCHS:
    if not results[a]["au"]:
        print(f"  {a:16s} (no successful folds)"); continue
    au = np.array(results[a]["au"]); d = au - bt[:len(au)]
    print(f"  {a:16s} AUROC {au.mean():.3f} +/- {au.std():.3f}  AUPRC {np.mean(results[a]['ap']):.3f}"
          f"   vs tree dAUROC {d.mean():+.4f} (folds>0 {int((d>0).sum())}/{len(au)})")
