"""Tabular Deep Significance Clustering (DICE).

Python port of DICE (Huang, Axsom, Lee, Subramanian, Zhang; arXiv:2101.02344 /
PMC8500061) adapted for TABULAR features (a dense autoencoder replaces the
original LSTM, since the surgical cohort has fixed-width features, not
sequences). This is the clustering stage of the "DICE first, ML later"
pipeline (PLOS pdig.0000606).

Objective (Eq. 6 + effect-size constraint):

    min  lam_ae*L_AE + L_clustering + lam_l1*L1 + lam_l2*L2
         + lam_sig*relu(aG - G_min)               # statistical significance
         + lam_smd*relu(SMD_target - SMD_min)     # clinical effect size (this study)
         + lam_bal*KL(mean(chat) || Uniform)

  L_AE         reconstruction MSE of the autoencoder
  L_clustering k-means loss ||z - M_{c}||^2 on the latent z
  L1           cross-entropy of the soft cluster head g1(z) vs k-means pseudo-labels
  L2           (class-weighted) BCE of the outcome head g2([chat, v]) vs y
  G_min        smallest pairwise likelihood-ratio statistic across clusters (p<0.05
               when G > 3.841)
  SMD_min      smallest pairwise standardized-mean-difference of the per-cluster
               readmission rate across clusters (SMD >= 0.3 = small-to-moderate
               clinical separation by Cohen's convention; >= 0.5 = moderate)
  KL(...)      anti-collapse balance term (see below)

Defaults lam_ae=1, lam_l1=10, lam_l2=1, lam_sig=1, lam_smd=1, lam_bal=0.1;
aG=3.841 (chi-square crit, df=1, alpha=0.05); SMD_target=0.3.

Rationale for the SMD constraint. The likelihood-ratio significance test can
pass with tiny but statistically distinguishable effect sizes (large N inflates
significance without clinical meaning). Requiring SMD >= 0.3 on the per-cluster
readmission rate additionally enforces a clinically meaningful separation
between every pair of clusters, so the tier structure is both significantly
and substantively differentiated. Formula:
    SMD_{k1,k2} = |r_k1 - r_k2| / sqrt( (r_k1*(1-r_k1) + r_k2*(1-r_k2)) / 2 )
where r_k is the soft-membership-weighted readmission rate of cluster k
(matches Table 1 SMD convention). Fully differentiable; backprops into the
soft memberships chat.

Significance constraint (EXACT, default). The paper enforces a likelihood-ratio
test G_{k1,k2}=2*(LL_full - LL_reduced) between every cluster pair, where the
reduced model drops the two clusters' membership columns. We FIT both nested
logistic models by unrolled IRLS/Newton in torch (so gradients flow through the
solve into the soft memberships chat), then take G from the converged
log-likelihoods. relu(aG - G_min) pushes the weakest-separated pair above the
0.05 threshold during joint optimization. The IRLS solve uses a small ridge
(1e-4) for stability; reduced designs mask the pair's columns (ridge keeps the
normal equations invertible). To bound cost, the term is evaluated once per
OUTER iteration on a capped subsample, not per minibatch. The cheaper
masked-forward-pass surrogate is retained behind EXACT_SIG=False.

Anti-collapse. Deep clustering degenerates to empty clusters. Three guards:
(i) k-means++ init (sklearn default); (ii) empty/near-empty k-means centroids
are reseeded to the farthest assigned point each outer iteration; (iii) a
balance term lam_bal*KL(batch-mean soft assignment || Uniform(K)) pushes the
cluster head to use every cluster.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import medhg_ps.config as C
from medhg_ps.train import set_seed, _resolve_device

SMOKE = os.environ.get("MEDHG_SMOKE") == "1"
EXACT_SIG = os.environ.get("DICE_SURROGATE_SIG") != "1"   # exact IRLS LR test by default
SEED = 42
ALPHA_G = 3.841                       # chi-square critical value, df=1, alpha=0.05
SMD_TARGET = 0.30                     # min pairwise SMD on per-cluster readmit rates
                                      # Cohen: 0.2 small, 0.5 medium, 0.8 large
LAM = dict(ae=1.0, clust=1.0, l1=10.0, l2=1.0, sig=1.0, smd=1.0, bal=0.1)   # Eq.6 + effect-size + balance
DEV = _resolve_device(C.DEFAULTS_TRAIN.device)

# training budget (Algorithm 1)
PRETRAIN_EP = 2 if SMOKE else 10
N_ITER      = 4 if SMOKE else 40
NAS_NITER   = 3 if SMOKE else 12      # cheaper fits during the K/d search
INNER_EP    = 1
BATCH       = 512

# exact significance (IRLS) settings
IRLS_STEPS  = 15                      # unrolled Newton steps (fixed-budget convergence)
IRLS_RIDGE  = 1e-2                    # ridge on the normal equations; also resolves the
                                      # intercept vs sum-to-1 membership collinearity
SIG_CAP     = 4096                    # rows sampled for the per-outer-iter significance step


class DICEModel(nn.Module):
    """Dense autoencoder + soft cluster head (g1) + logistic outcome head (g2)."""

    def __init__(self, p: int, d: int, K: int, v_dim: int = 0, hidden: int = 128):
        super().__init__()
        self.K, self.v_dim = K, v_dim
        self.encoder = nn.Sequential(nn.Linear(p, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.decoder = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, p))
        self.cluster_head = nn.Linear(d, K)            # g1: z -> cluster logits
        self.outcome_head = nn.Linear(K + v_dim, 1)    # g2: [chat, v] -> outcome logit

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def cluster_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.cluster_head(z)

    def soft_membership(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.cluster_head(z), dim=-1)

    def outcome_logit(self, chat: torch.Tensor, v: Optional[torch.Tensor] = None) -> torch.Tensor:
        inp = chat if (v is None or self.v_dim == 0) else torch.cat([chat, v], dim=-1)
        return self.outcome_head(inp).squeeze(-1)

    # -- significance: surrogate (fast) -------------------------------------
    def significance_surrogate(self, chat: torch.Tensor, y: torch.Tensor,
                               v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """relu(aG - min_pair G) using the CURRENT outcome head on masked inputs."""
        logit_full = self.outcome_logit(chat, v)
        ll_full = -F.binary_cross_entropy_with_logits(logit_full, y, reduction="sum")
        gmin = None
        for k1 in range(self.K):
            for k2 in range(k1 + 1, self.K):
                mask = torch.ones(self.K, device=chat.device)
                mask[k1] = 0.0; mask[k2] = 0.0
                logit_red = self.outcome_logit(chat * mask, v)
                ll_red = -F.binary_cross_entropy_with_logits(logit_red, y, reduction="sum")
                g = 2.0 * (ll_full - ll_red)
                gmin = g if gmin is None else torch.minimum(gmin, g)
        if gmin is None:
            return torch.zeros((), device=chat.device)
        return torch.relu(ALPHA_G - gmin)

    # -- significance: exact nested LR test via unrolled IRLS ---------------
    def significance_exact(self, chat: torch.Tensor, y: torch.Tensor,
                           v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """relu(aG - min_pair G) where each G is a true likelihood-ratio test.

        The full model regresses y on [intercept, chat, v]; the reduced model
        for pair (k1,k2) masks those two membership columns (merging them into
        the reference). Both are fit by IRLS (Newton) unrolled for IRLS_STEPS
        with ridge IRLS_RIDGE, so the converged log-likelihoods -- and hence
        G=2*(LL_full-LL_red) -- are differentiable w.r.t. chat.
        """
        n = chat.shape[0]
        ones = torch.ones(n, 1, device=chat.device)
        base = chat if (v is None or self.v_dim == 0) else torch.cat([chat, v], dim=-1)
        Xd_full = torch.cat([ones, base], dim=1)
        ll_full = _irls_loglik(Xd_full, y)
        gmin = None
        for k1 in range(self.K):
            for k2 in range(k1 + 1, self.K):
                colmask = torch.ones(base.shape[1], device=chat.device)
                colmask[k1] = 0.0; colmask[k2] = 0.0       # drop the two cluster columns
                Xd_red = torch.cat([ones, base * colmask], dim=1)
                ll_red = _irls_loglik(Xd_red, y)
                g = 2.0 * (ll_full - ll_red)
                gmin = g if gmin is None else torch.minimum(gmin, g)
        if gmin is None:
            return torch.zeros((), device=chat.device)
        return torch.relu(ALPHA_G - gmin)

    def significance_penalty(self, chat, y, v=None):
        return (self.significance_exact(chat, y, v) if EXACT_SIG
                else self.significance_surrogate(chat, y, v))

    # -- effect size: min pairwise SMD on per-cluster readmission rate --------
    def smd_penalty(self, chat: torch.Tensor, y: torch.Tensor,
                    smd_target: float = SMD_TARGET,
                    min_soft_n: float = 5.0) -> torch.Tensor:
        """relu(smd_target - min_pair_SMD) on per-cluster readmit rates.

        Soft per-cluster readmit rate: r_k = sum_i(chat[i,k] * y[i]) / n_k
        with n_k = sum_i(chat[i,k]) the soft cluster size. Cohen-style SMD
        for binary outcomes:
            SMD_{k1,k2} = |r_k1 - r_k2| / sqrt((r_k1*(1-r_k1) + r_k2*(1-r_k2))/2)
        (matches the Table-1 SMD convention.) Clusters with soft size below
        min_soft_n are ignored (would inflate SMD via variance underestimate).
        Fully differentiable in chat.
        """
        eps = 1e-6
        n_k = chat.sum(0)                                      # [K]
        r_k = (chat * y.unsqueeze(1)).sum(0) / n_k.clamp_min(eps)   # [K]
        r_k = r_k.clamp(eps, 1 - eps)
        valid = n_k > min_soft_n
        if valid.sum() < 2:
            return torch.zeros((), device=chat.device)
        smd_min = None
        for k1 in range(self.K):
            for k2 in range(k1 + 1, self.K):
                if not (valid[k1] and valid[k2]):              # skip tiny clusters
                    continue
                num = torch.abs(r_k[k1] - r_k[k2])
                den = torch.sqrt((r_k[k1] * (1 - r_k[k1])
                                  + r_k[k2] * (1 - r_k[k2])) / 2 + eps)
                smd = num / den
                smd_min = smd if smd_min is None else torch.minimum(smd_min, smd)
        if smd_min is None:
            return torch.zeros((), device=chat.device)
        return torch.relu(smd_target - smd_min)


def _irls_loglik(Xd: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fit logistic regression coefficients by unrolled IRLS and return the
    Bernoulli log-likelihood at the converged fit. Differentiable in Xd."""
    n, m = Xd.shape
    eye = torch.eye(m, device=Xd.device)
    beta = torch.zeros(m, device=Xd.device)
    for _ in range(IRLS_STEPS):
        eta = Xd @ beta
        mu = torch.sigmoid(eta).clamp(1e-6, 1 - 1e-6)
        w = (mu * (1 - mu)).clamp_min(1e-6)
        zwork = eta + (y - mu) / w                          # IRLS working response
        A = Xd.t() @ (w.unsqueeze(1) * Xd) + IRLS_RIDGE * eye
        b = Xd.t() @ (w * zwork)
        try:
            beta = torch.linalg.solve(A, b)
        except torch._C._LinAlgError:                       # singular: pinv fallback
            beta = torch.linalg.pinv(A) @ b
    mu = torch.sigmoid(Xd @ beta).clamp(1e-6, 1 - 1e-6)
    return (y * torch.log(mu) + (1 - y) * torch.log(1 - mu)).sum()


def _kmeans_with_reseed(Z: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """k-means++ on Z; reseed any empty/near-empty cluster to the point farthest
    from its assigned centroid. Returns (centers [K,d], hard labels [N])."""
    km = KMeans(n_clusters=K, n_init=(2 if SMOKE else 10),
                init="k-means++", random_state=SEED).fit(Z)
    centers, labels = km.cluster_centers_.copy(), km.labels_.copy()
    counts = np.bincount(labels, minlength=K)
    dist = ((Z - centers[labels]) ** 2).sum(1)
    for k in np.where(counts == 0)[0]:                      # reseed dead centroids
        far = int(np.argmax(dist))
        centers[k] = Z[far]
        labels[far] = k
        dist[far] = -1.0
    return centers, labels


def _to_dev(X: np.ndarray, y: Optional[np.ndarray] = None):
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=DEV)
    yt = None if y is None else torch.tensor(np.asarray(y), dtype=torch.float32, device=DEV)
    return Xt, yt


def fit(X: np.ndarray, y: np.ndarray, K: int, d: int,
        v: Optional[np.ndarray] = None, n_iter: int = N_ITER,
        lr: float = 1e-3, verbose: bool = False,
        spread: float = 0.0, lam_bal: Optional[float] = None,
        smd_target: float = SMD_TARGET,
        seed: int = SEED) -> DICEModel:
    """Train DICE (Algorithm 1) on standardized features X and labels y.

    spread > 0 adds a tier-separation term that MAXIMIZES the variance of the
    soft-membership-weighted per-cluster outcome rate, pulling the clusters
    into distinct low/medium/high risk levels. lam_bal overrides the balance
    weight (raise it alongside spread so every tier stays populated).

    smd_target is the minimum required pairwise SMD on per-cluster readmission
    rates. The default 0.30 enforces small-to-moderate Cohen effect size
    between every pair of clusters, in addition to the LR-test significance
    (p<0.05) constraint. Set smd_target=0.0 to disable the effect-size
    constraint (original DICE behavior)."""
    set_seed(seed)
    Xt, yt = _to_dev(X, y)
    vt = None if v is None else _to_dev(v)[0]
    p = Xt.shape[1]
    model = DICEModel(p, d, K, v_dim=(0 if v is None else v.shape[1])).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    npos = max(float((y == 1).sum()), 1.0)
    pos_weight = torch.tensor(float((y == 0).sum()) / npos, device=DEV)
    bal_w = LAM["bal"] if lam_bal is None else lam_bal
    rng = np.random.default_rng(seed)
    n = Xt.shape[0]
    uniform = torch.full((K,), 1.0 / K, device=DEV)

    # --- pretrain the autoencoder ---
    for _ in range(PRETRAIN_EP):
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            z = model.encode(Xt[b])
            F.mse_loss(model.decoder(z), Xt[b]).backward()
            opt.step()

    # --- main joint loop ---
    for it in range(n_iter):
        with torch.no_grad():
            Z = model.encode(Xt).cpu().numpy()
        centers_np, labels_np = _kmeans_with_reseed(Z, K)
        centers = torch.tensor(centers_np, dtype=torch.float32, device=DEV)
        hard = torch.tensor(labels_np, dtype=torch.long, device=DEV)
        for _ in range(INNER_EP):
            perm = rng.permutation(n)
            for s in range(0, n, BATCH):
                b = perm[s:s + BATCH]
                opt.zero_grad()
                xb = Xt[b]
                z = model.encode(xb)
                chat = model.soft_membership(z)
                vb = None if vt is None else vt[b]
                l_ae = F.mse_loss(model.decoder(z), xb)
                l_clust = ((z - centers[hard[b]]) ** 2).sum(1).mean()
                l1 = F.cross_entropy(model.cluster_logits(z), hard[b])
                l2 = F.binary_cross_entropy_with_logits(
                    model.outcome_logit(chat, vb), yt[b], pos_weight=pos_weight)
                qbar = chat.mean(0).clamp_min(1e-8)
                l_bal = (qbar * (qbar.log() - uniform.log())).sum()     # KL(qbar||U)
                loss = (LAM["ae"] * l_ae + LAM["clust"] * l_clust + LAM["l1"] * l1
                        + LAM["l2"] * l2 + bal_w * l_bal)
                if spread > 0.0:                                        # tier separation
                    nk = chat.sum(0).clamp_min(1e-6)
                    rk = (chat * yt[b].unsqueeze(1)).sum(0) / nk        # per-cluster risk
                    loss = loss - spread * ((rk - rk.mean()) ** 2).mean()
                loss.backward(); opt.step()

        # significance + effect size: once per OUTER iteration on a capped subsample
        idx = rng.permutation(n)[:min(SIG_CAP, n)]
        opt.zero_grad()
        zc = model.encode(Xt[idx])
        chatc = model.soft_membership(zc)
        vc = None if vt is None else vt[idx]
        l_sig = model.significance_penalty(chatc, yt[idx], vc)
        l_smd = model.smd_penalty(chatc, yt[idx], smd_target=smd_target)
        (LAM["sig"] * l_sig + LAM["smd"] * l_smd).backward()
        opt.step()

    if verbose:
        with torch.no_grad():
            Zf = model.encode(Xt); chatf = model.soft_membership(Zf)
            cnt = np.bincount(chatf.argmax(1).cpu().numpy(), minlength=K)
            # audit final significance + SMD
            idx_full = rng.permutation(n)[:min(SIG_CAP, n)]
            chatc = model.soft_membership(model.encode(Xt[idx_full]))
            n_k = chatc.sum(0).cpu().numpy()
            r_k = (chatc * yt[idx_full].unsqueeze(1)).sum(0).cpu().numpy() / np.clip(n_k, 1, None)
            smd_pairs = []
            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    d1 = r_k[k1] * (1 - r_k[k1]); d2 = r_k[k2] * (1 - r_k[k2])
                    if n_k[k1] > 5 and n_k[k2] > 5:
                        smd_pairs.append(abs(r_k[k1] - r_k[k2]) / (np.sqrt((d1 + d2) / 2) + 1e-6))
            smd_min = min(smd_pairs) if smd_pairs else float('nan')
        print(f"[dice] fit K={K} d={d}: cluster argmax counts {cnt.tolist()} "
              f"(empty={(cnt == 0).sum()})  min_SMD={smd_min:.3f}  target={smd_target:.2f}",
              flush=True)
    return model


def cluster_proba(model: DICEModel, X: np.ndarray) -> np.ndarray:
    """Soft cluster membership chat in [N, K] for new rows."""
    model.eval()
    with torch.no_grad():
        Xt, _ = _to_dev(X)
        return model.soft_membership(model.encode(Xt)).cpu().numpy()


def embed(model: DICEModel, X: np.ndarray) -> np.ndarray:
    """Latent autoencoder embedding z in [N, d] for new rows (richer than chat)."""
    model.eval()
    with torch.no_grad():
        Xt, _ = _to_dev(X)
        return model.encode(Xt).cpu().numpy()


def _val_auc(model: DICEModel, Xv: np.ndarray, yv: np.ndarray,
             vv: Optional[np.ndarray]) -> float:
    model.eval()
    with torch.no_grad():
        Xt, _ = _to_dev(Xv)
        chat = model.soft_membership(model.encode(Xt))
        vt = None if vv is None else _to_dev(vv)[0]
        p = torch.sigmoid(model.outcome_logit(chat, vt)).cpu().numpy()
    try:
        return roc_auc_score(yv, p)
    except ValueError:
        return 0.5


def search_fit(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
               vtr: Optional[np.ndarray] = None, vva: Optional[np.ndarray] = None,
               Ks: Tuple[int, ...] = (2, 3, 4, 5),
               ds: Tuple[int, ...] = (16, 32)) -> Tuple[int, int, float]:
    """NAS over (K, d): pick argmax validation AUROC of the DICE outcome head.

    Returns (K*, d*, best_val_auc). Cheap fits (NAS_NITER) are used for the
    search; the caller refits the winner on the full train split.
    """
    if SMOKE:
        return 3, 16, _val_auc(fit(Xtr, ytr, 3, 16, vtr, n_iter=NAS_NITER), Xva, yva, vva)
    best = (Ks[0], ds[0], -1.0)
    for K in Ks:
        for d in ds:
            m = fit(Xtr, ytr, K, d, vtr, n_iter=NAS_NITER)
            a = _val_auc(m, Xva, yva, vva)
            if a > best[2]:
                best = (K, d, a)
    return best
