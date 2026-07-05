"""
ie-HGCN (interpretable efficient Heterogeneous Graph Convolutional Network).

Faithful re-implementation of the equations in Chen et al. (npj Health
Systems 2025) §"The proposed framework":

    Eq 1 -- per-type aggregation with attention:
        H^Omega_{l+1} = ELU( a^Omega_l . Z^Omega_l
                            + sum_{Gamma in N^Omega} a^Gamma_l . Z^Gamma_l )

    Eq 2 -- attention coefficients (softmax over the union of self and
            neighbor types):
        a^Omega = softmax( ELU( [ Z^Omega W_k || (Z^Omega W_q) ] . w_a^Omega ) )
        a^Gamma = softmax( ELU( [ Z^Gamma W_k || (Z^Omega W_q) ] . w_a^Omega ) )

    Eq 3 -- neighbor-side projection through the row-normalised
            adjacency  Â^{Omega-Gamma}:
        Z^Gamma = Â^{Omega-Gamma} . H^Gamma_{l-1} . W^{Gamma->Omega}

    Eq 4 -- self-side projection (Â^{Omega-Omega} = I):
        Z^Omega = H^Omega_{l-1} . W^{Omega->Omega}

After three stacked layers the encounter-node embeddings are fed into a
softmax classifier to produce y_hat (paper Eq below Eq 4):

        y_hat = softmax(H^ENC_3)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


# =====================================================================
# Single ie-HGCN layer
# =====================================================================
class IeHGCNLayer(nn.Module):
    """One layer of ie-HGCN.

    For a target node type Omega with neighbor types {Gamma_1, ..., Gamma_K}:

      Step 1: self-projection   Z^Omega = H^Omega . W^{Omega->Omega}
      Step 2: each neighbor     Z^Gamma_k = Â^{Omega-Gamma_k} . H^Gamma_k . W^{Gamma_k->Omega}
                                where Â is row-normalised by message passing
      Step 3: per-node attention over the (K+1) type channels with a
              concatenated [key || query] scoring function (Eq 2),
              softmax over types
      Step 4: weighted sum (Eq 1), ELU activation

    The mean normalised attention weight per type channel is cached on
    `self.last_attn` so explain.py can build meta-path importances by
    multiplying them along an edge sequence (paper §Explainable AI).
    """

    def __init__(
        self,
        node_types: Tuple[str, ...],
        canonical_edges: Tuple[Tuple[str, str, str], ...],
        in_dims: Dict[str, int],
        out_dim: int,
        attn_dim: int,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.node_types       = node_types
        self.canonical_edges  = canonical_edges
        self.out_dim          = out_dim
        self.attn_dim         = attn_dim

        # Per (source_type -> target_type) transformation matrix W^{Gamma->Omega}.
        # The "self" channel uses the (Omega -> Omega) entry.
        self.W = nn.ModuleDict()
        for src in node_types:
            for dst in node_types:
                self.W[f"{src}__{dst}"] = nn.Linear(in_dims[src], out_dim, bias=False)

        # Per target-type attention parameters W_k, W_q (linear maps to
        # attention dim) and w_a (the attention parameter vector that
        # produces a scalar score per node per channel).
        self.W_k = nn.ModuleDict({nt: nn.Linear(out_dim, attn_dim, bias=False)
                                  for nt in node_types})
        self.W_q = nn.ModuleDict({nt: nn.Linear(out_dim, attn_dim, bias=False)
                                  for nt in node_types})
        # The score function takes [k || q] (2 * attn_dim) -> scalar.
        self.w_a = nn.ModuleDict({nt: nn.Linear(2 * attn_dim, 1, bias=False)
                                  for nt in node_types})

        # Regularisation - paper Table 2 has both knobs.
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.ModuleDict({nt: nn.BatchNorm1d(out_dim) if use_batch_norm
                                 else nn.Identity()
                                 for nt in node_types})

        # Cache for last forward pass - keyed by (target type) -> tensor
        # of shape (n_target_nodes, K+1) holding per-node, per-channel
        # attention weights, in channel order [self, *neighbors_sorted].
        self.last_attn: Dict[str, torch.Tensor] = {}
        # Channel labels matching last_attn columns.
        self.last_channels: Dict[str, List[str]] = {}

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    def _neighbor_types(self, target: str) -> List[Tuple[str, str, str]]:
        """All canonical edges with dst == target. Sorted so the
        attention-channel order is deterministic across forward calls."""
        nbrs = [e for e in self.canonical_edges if e[2] == target]
        return sorted(nbrs)

    # -----------------------------------------------------------------
    def _row_norm_message(
        self, g: dgl.DGLHeteroGraph, etype: Tuple[str, str, str],
        src_h: torch.Tensor
    ) -> torch.Tensor:
        """One step of message passing that computes  Â . src_h  for the
        given edge type, where Â is the row-normalised adjacency (each
        destination row sums to 1; isolated rows stay at zero). This is
        Eq 3's  Â^{Omega-Gamma} . H^Gamma  factor, before the W projection.
        """
        with g.local_scope():
            g.nodes[etype[0]].data["m"] = src_h
            # Sum-aggregate, then divide by destination in-degree.
            g.update_all(fn.copy_u("m", "msg"), fn.sum("msg", "agg"), etype=etype)
            agg = g.nodes[etype[2]].data["agg"]
            deg = g.in_degrees(etype=etype).clamp(min=1).unsqueeze(-1).float()
            return agg / deg

    # -----------------------------------------------------------------
    def forward(
        self, g: dgl.DGLHeteroGraph, h: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        self.last_attn   = {}
        self.last_channels = {}

        for target in self.node_types:
            # ---- Step 1+2: compute Z^Omega (self) and Z^Gamma (neighbors) ----
            # Eq 4: Z^Omega = H^Omega . W^{Omega->Omega}
            z_self = self.W[f"{target}__{target}"](h[target])

            z_nbrs: List[torch.Tensor] = []
            channels: List[str] = [target]  # self channel first
            for etype in self._neighbor_types(target):
                src_type = etype[0]
                # Eq 3: Â . H^Gamma  then  . W^{Gamma->Omega}
                propagated = self._row_norm_message(g, etype, h[src_type])
                z_nbrs.append(self.W[f"{src_type}__{target}"](propagated))
                channels.append(src_type + "::" + etype[1])
            # Stack channels: shape (n_target, K+1, out_dim)
            z_stack = torch.stack([z_self] + z_nbrs, dim=1)

            # ---- Step 3: attention scores (Eq 2) ----
            # Eq 2 applies ELU exactly once, AFTER the dot product with
            # the W_a parameter vector. The W_k / W_q projections are
            # raw linears with no activation, then concatenated, then
            # scored by W_a, then ELU, then softmax over channels.
            q = self.W_q[target](z_self)                              # (n_target, attn_dim)
            scores: List[torch.Tensor] = []
            for c in range(z_stack.shape[1]):
                k = self.W_k[target](z_stack[:, c, :])                # (n_target, attn_dim)
                concat = torch.cat([k, q], dim=-1)                    # (n_target, 2 * attn_dim)
                scores.append(F.elu(self.w_a[target](concat)))        # (n_target, 1)
            scores = torch.cat(scores, dim=-1)                         # (n_target, K+1)
            attn   = F.softmax(scores, dim=-1)                         # softmax over types

            # ---- Step 4: weighted sum (Eq 1), activation, BN, dropout ----
            # attn -> (n_target, K+1, 1); z_stack -> (n_target, K+1, out_dim)
            h_next = (attn.unsqueeze(-1) * z_stack).sum(dim=1)
            h_next = F.elu(h_next)
            h_next = self.bn[target](h_next)
            h_next = self.dropout(h_next)
            out[target] = h_next

            # Detach + cache attention for meta-path importance.
            self.last_attn[target]     = attn.detach()
            self.last_channels[target] = channels

        return out


# =====================================================================
# Three-layer ie-HGCN + readmission classifier
# =====================================================================
class MedHGPS(nn.Module):
    """Three stacked ie-HGCN layers (paper: l in [1, 3]) followed by a
    linear classifier on encounter-node embeddings.

    forward() returns RAW LOGITS over {no readmission, readmission
    within 30 days}. Use nn.CrossEntropyLoss directly on these logits,
    or apply torch.softmax(logits, dim=-1) to get probabilities.
    """

    def __init__(
        self,
        in_dims: Dict[str, int],
        cfg: C.TrainConfig = C.DEFAULTS_TRAIN,
        n_classes: int = 2,
    ):
        super().__init__()
        nts   = C.NODE_TYPES
        edges = C.EDGE_TYPES

        # Three hidden widths (paper varies first-layer dim; we taper).
        dims_per_layer = [cfg.hidden_dim_1, cfg.hidden_dim_2, cfg.hidden_dim_3]
        assert len(dims_per_layer) == cfg.n_layers

        self.layers = nn.ModuleList()
        cur_in = dict(in_dims)
        for d in dims_per_layer:
            self.layers.append(IeHGCNLayer(
                node_types=nts, canonical_edges=edges,
                in_dims=cur_in, out_dim=d,
                attn_dim=cfg.attn_dim,
                dropout=cfg.dropout, use_batch_norm=cfg.batch_norm,
            ))
            cur_in = {nt: d for nt in nts}

        self.classifier = nn.Linear(dims_per_layer[-1], n_classes)

    def forward(
        self, g: dgl.DGLHeteroGraph, h: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        for layer in self.layers:
            h = layer(g, h)
        enc_emb = h[C.ENC_NTYPE]
        logits  = self.classifier(enc_emb)
        return logits, h

    # -----------------------------------------------------------------
    # Convenience for embedding extraction (paper Fig 6b shows softmax
    # on top of H^ENC_3; we also expose H^PROV_3 and H^UNIT_3 for
    # downstream RF integration via Option B).
    # -----------------------------------------------------------------
    @torch.no_grad()
    def get_final_embeddings(
        self, g: dgl.DGLHeteroGraph, h: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        for layer in self.layers:
            h = layer(g, h)
        return h

    # -----------------------------------------------------------------
    # Attention coefficient cache - explain.py multiplies these along
    # meta-paths to compute path importance (paper §Explainable AI).
    # -----------------------------------------------------------------
    def collect_attention(self) -> List[Dict[str, Dict[str, torch.Tensor]]]:
        """Returns one dict per layer:
            { target_type: { channel_label: mean_normalised_attention } }
        """
        out = []
        for layer in self.layers:
            layer_out = {}
            for target, attn in layer.last_attn.items():
                channels = layer.last_channels[target]
                mean_attn = attn.mean(dim=0)
                layer_out[target] = {
                    ch: mean_attn[i].item() for i, ch in enumerate(channels)
                }
            out.append(layer_out)
        return out


# =====================================================================
# Parameter count (paper reports 3.2M for MedHG-PS)
# =====================================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
