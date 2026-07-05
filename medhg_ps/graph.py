"""
Heterogeneous graph construction.

Builds the DGL hetero graph described in Chen et al. (npj Health Systems
2025) Methods § The proposed framework:

    "A heterogeneous graph G = (V, E), as shown in Fig 6a, was first
     constructed as the starting point using the input variables. The
     graph consists of three types of nodes V in {ENC, P, C}, representing
     encounter, provider, and care unit. All nodes are connected by
     undirected edges E, which represent their interaction."

Undirected edges are represented in DGL by adding both directions, so
the four canonical edge types are:

    (encounter, has_provider,  provider)
    (provider,  treated,       encounter)
    (encounter, visited_unit,  unit)
    (unit,      hosted,        encounter)

Node-feature matrices H^ENC, H^PROV, H^UNIT are attached as ndata['h'].
The encounter readmission label and train/val/test masks are also
stored on encounter nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import dgl
import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import RawExtract


@dataclass
class GraphArtifacts:
    """Bundle that downstream modules consume."""
    g: dgl.DGLHeteroGraph
    enc_id_to_idx:  Dict[str, int]
    prov_id_to_idx: Dict[str, int]
    unit_id_to_idx: Dict[str, int]
    enc_idx_to_id:  Dict[int, str]
    prov_idx_to_id: Dict[int, str]
    unit_idx_to_id: Dict[int, str]
    n_enc_features:  int
    n_prov_features: int
    n_unit_features: int


def _id_map(values) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build forward (str -> int) and reverse (int -> str) maps. Dedupes
    while preserving first-seen order so duplicate source rows collapse
    to a single node id - otherwise the forward map's "last write wins"
    would leave the reverse map oversized and DGL edges referencing
    invalid indices."""
    unique_values = list(dict.fromkeys(values))
    forward = {v: i for i, v in enumerate(unique_values)}
    reverse = {i: v for v, i in forward.items()}
    return forward, reverse


def build_graph(
    raw: RawExtract,
    encounters_merged: pd.DataFrame,
    enc_features: np.ndarray,
    prov_ids: pd.DataFrame,
    prov_features: np.ndarray,
    unit_ids: pd.DataFrame,
    unit_features: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> GraphArtifacts:

    # Integer ID maps - DGL requires contiguous int64 node ids per type.
    enc_fwd,  enc_rev  = _id_map(encounters_merged["LogID"].astype(str).tolist())
    prov_fwd, prov_rev = _id_map(prov_ids["ProvID"].astype(str).tolist())
    unit_fwd, unit_rev = _id_map(unit_ids["DepartmentID"].astype(str).tolist())

    # ENC <-> PROV edges. We filter to provider IDs that appear in
    # prov_attrs (A4) so every endpoint has a feature row.
    ep = raw.enc_prov_edges.copy()
    ep["src"] = ep["LogID"].astype(str).map(enc_fwd)
    ep["dst"] = ep["ProvID"].astype(str).map(prov_fwd)
    ep = ep.dropna(subset=["src", "dst"])
    ep_src = torch.tensor(ep["src"].astype(int).values, dtype=torch.int64)
    ep_dst = torch.tensor(ep["dst"].astype(int).values, dtype=torch.int64)

    # ENC <-> UNIT edges, weighted by hours (used downstream for
    # hours-weighted embedding aggregation; the GNN itself treats edges
    # as unweighted per the paper).
    eu = raw.enc_unit_edges.copy()
    eu["src"] = eu["LogID"].astype(str).map(enc_fwd)
    eu["dst"] = eu["DepartmentID"].astype(str).map(unit_fwd)
    eu = eu.dropna(subset=["src", "dst"])
    eu_src = torch.tensor(eu["src"].astype(int).values, dtype=torch.int64)
    eu_dst = torch.tensor(eu["dst"].astype(int).values, dtype=torch.int64)
    eu_w   = torch.tensor(eu["Hours"].fillna(0.0).astype(float).values,
                          dtype=torch.float32)

    data_dict = {
        C.ETYPE_ENC_PROV: (ep_src, ep_dst),
        C.ETYPE_PROV_ENC: (ep_dst, ep_src),
        C.ETYPE_ENC_UNIT: (eu_src, eu_dst),
        C.ETYPE_UNIT_ENC: (eu_dst, eu_src),
    }
    num_nodes_dict = {
        C.ENC_NTYPE:  len(enc_fwd),
        C.PROV_NTYPE: len(prov_fwd),
        C.UNIT_NTYPE: len(unit_fwd),
    }
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    g.nodes[C.ENC_NTYPE].data["h"]  = torch.tensor(enc_features,  dtype=torch.float32)
    g.nodes[C.PROV_NTYPE].data["h"] = torch.tensor(prov_features, dtype=torch.float32)
    g.nodes[C.UNIT_NTYPE].data["h"] = torch.tensor(unit_features, dtype=torch.float32)

    y = encounters_merged["ReadmittedWithin30Days"].astype(int).values
    g.nodes[C.ENC_NTYPE].data["y"]          = torch.tensor(y, dtype=torch.long)
    g.nodes[C.ENC_NTYPE].data["train_mask"] = torch.tensor(train_mask, dtype=torch.bool)
    g.nodes[C.ENC_NTYPE].data["val_mask"]   = torch.tensor(val_mask,   dtype=torch.bool)
    g.nodes[C.ENC_NTYPE].data["test_mask"]  = torch.tensor(test_mask,  dtype=torch.bool)
    g.edges[C.ETYPE_ENC_UNIT].data["hours"] = eu_w
    g.edges[C.ETYPE_UNIT_ENC].data["hours"] = eu_w

    return GraphArtifacts(
        g=g,
        enc_id_to_idx=enc_fwd, prov_id_to_idx=prov_fwd, unit_id_to_idx=unit_fwd,
        enc_idx_to_id=enc_rev, prov_idx_to_id=prov_rev, unit_idx_to_id=unit_rev,
        n_enc_features=enc_features.shape[1],
        n_prov_features=prov_features.shape[1],
        n_unit_features=unit_features.shape[1],
    )


def save_graph(artifacts: GraphArtifacts, path) -> None:
    """Persist the DGL graph + id maps. The id maps live in a sidecar
    pickle since dgl.save_graphs only stores tensors."""
    import pickle, dgl
    path.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(path), [artifacts.g])
    with open(str(path) + ".idmaps.pkl", "wb") as f:
        pickle.dump({
            "enc_id_to_idx":  artifacts.enc_id_to_idx,
            "prov_id_to_idx": artifacts.prov_id_to_idx,
            "unit_id_to_idx": artifacts.unit_id_to_idx,
            "enc_idx_to_id":  artifacts.enc_idx_to_id,
            "prov_idx_to_id": artifacts.prov_idx_to_id,
            "unit_idx_to_id": artifacts.unit_idx_to_id,
            "n_enc_features":  artifacts.n_enc_features,
            "n_prov_features": artifacts.n_prov_features,
            "n_unit_features": artifacts.n_unit_features,
        }, f)


def load_graph(path) -> GraphArtifacts:
    import pickle, dgl
    (g,), _ = dgl.load_graphs(str(path))
    with open(str(path) + ".idmaps.pkl", "rb") as f:
        meta = pickle.load(f)
    return GraphArtifacts(g=g, **meta)
