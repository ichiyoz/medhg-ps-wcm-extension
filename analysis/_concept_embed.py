"""Helper for V2: embed encounters with a GNN over the 5-node clinical-concept
heterograph (encounter + provider + unit + DIAGNOSIS + PROCEDURE nodes), so
patients connect through shared conditions and operations. Returns encounter
embeddings (last hidden layer) for outcome-driven clustering. Trained once on
all rows (descriptive use). Replicates cv_gnn_clinical's graph + a SAGE encoder.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, dgl
from dgl.nn import SAGEConv
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import medhg_ps.config as C
from medhg_ps.data import fit_preprocess, apply_preprocess
from medhg_ps.train import set_seed, _resolve_device
from medhg_ps.deploy import assemble_training_frame, _load_cpt_map

dev = _resolve_device(C.DEFAULTS_TRAIN.device)
COMORB = ["Diabetes Mellitus", "Hypertension requiring medication", "Heart Failure",
          "History of Severe COPD", "Ascites", "Disseminated Cancer", "Bleeding Disorder",
          "Preop Acute Kidney Injury", "Preop Dialysis", "Ventilator Dependent",
          "Immunosuppressive Therapy", "Current Smoker within 1 year", "Preop RBC Transfusions (72h)"]
_norm = lambda s: s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def embed(hid: int = 64, epochs: int = 60):
    merged, feat_cols, cpt_arr, Fseq, seq_all, y = assemble_training_frame()
    y = np.asarray(y).astype(int); N = len(merged)
    merged = merged.reset_index(drop=True)
    row_of = {l: i for i, l in enumerate(_norm(merged["LogID"]))}
    raw = __import__("medhg_ps.data", fromlist=["load_raw"]).load_raw()

    P_ids = sorted({p for p in _norm(raw.enc_prov_edges["ProvID"]) if p and p != "nan"})
    U_ids = sorted({d for d in _norm(raw.enc_unit_edges["DepartmentID"]) if d and d != "nan"})
    DX_ids = [c for c in COMORB if c in merged.columns]
    CPT_ids = sorted({v for v in cpt_arr.ravel() if v and v != "UNK"})
    pidx = {p: N + i for i, p in enumerate(P_ids)}; o = N + len(P_ids)
    uidx = {d: o + i for i, d in enumerate(U_ids)}; o += len(U_ids)
    dxidx = {d: o + i for i, d in enumerate(DX_ids)}; o += len(DX_ids)
    cidx = {c: o + i for i, c in enumerate(CPT_ids)}; TOT = o + len(CPT_ids)

    src, dst = [], []
    ap = raw.enc_prov_edges[["LogID", "ProvID"]].copy()
    for lid, pv in zip(_norm(ap["LogID"]), _norm(ap["ProvID"])):
        r, pp = row_of.get(lid), pidx.get(pv)
        if r is not None and pp is not None: src += [r, pp]; dst += [pp, r]
    au = raw.enc_unit_edges[["LogID", "DepartmentID"]].copy()
    for lid, dp in zip(_norm(au["LogID"]), _norm(au["DepartmentID"])):
        r, uu = row_of.get(lid), uidx.get(dp)
        if r is not None and uu is not None: src += [r, uu]; dst += [uu, r]
    for dx in DX_ids:
        present = ~merged[dx].astype(str).str.strip().str.lower().isin(["no", "0", "nan", "none", ""])
        dn = dxidx[dx]
        for r in np.nonzero(present.values)[0]: src += [int(r), dn]; dst += [dn, int(r)]
    for r in range(N):
        cn = cidx.get(cpt_arr[r, 0])
        if cn is not None: src += [r, cn]; dst += [cn, r]
    g = dgl.add_self_loop(dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=TOT)).to(dev)

    # node features: encounter clinical features; concept nodes zero-padded
    _, st = fit_preprocess(merged[feat_cols], id_cols=[])
    Xenc = StandardScaler().fit_transform(apply_preprocess(merged[feat_cols], st)).astype(np.float32)
    Xn = np.zeros((TOT, Xenc.shape[1]), dtype=np.float32); Xn[:N] = Xenc
    Xn = torch.tensor(Xn, device=dev); yt = torch.tensor(y, device=dev)

    class Net(nn.Module):
        def __init__(self, ind):
            super().__init__()
            self.c1 = SAGEConv(ind, hid, "mean"); self.c2 = SAGEConv(hid, hid, "mean")
            self.cls = nn.Linear(hid, 2); self.drop = nn.Dropout(0.2)
        def emb(self, gg, x):
            h = self.drop(Fn.relu(self.c1(gg, x)))
            return self.drop(Fn.relu(self.c2(gg, h)))
        def forward(self, gg, x): return self.cls(self.emb(gg, x))

    set_seed(42)
    net = Net(Xn.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    w = torch.tensor([(y == 1).sum() + (y == 0).sum()], device=dev)
    cw = torch.tensor([1.0, float((y == 0).sum()) / max(float((y == 1).sum()), 1)], device=dev)
    lf = nn.CrossEntropyLoss(weight=cw)
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = lf(net(g, Xn)[:N], yt); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        E = net.emb(g, Xn)[:N].cpu().numpy()
    print(f"[concept-embed] 5-node graph nodes={TOT:,} (enc={N} P={len(P_ids)} U={len(U_ids)} "
          f"DX={len(DX_ids)} CPT={len(CPT_ids)}); emb dim={E.shape[1]}", flush=True)
    return E
