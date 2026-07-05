"""Deployable GRU care-path model (Table-2 candidate).

The Table-2 "Gradient-boosted tree + GRU care-path sequence encoder" model:
a GRU reads each encounter's time-ordered care-unit visits and its final
hidden state is concatenated to the tabular + CPT features, fed to an
isotonic-calibrated histogram gradient-boosted tree.

Unlike medhg_ps.deploy.ReadmissionModel (torch-free), this bundle carries the
trained GRU, so loading it needs torch. The fitted `ReadmissionGRUModel` is
what gets pickled; call `.predict_proba(records)` with raw per-encounter
records (tabular fields + PrimaryCPT + a `postop_visits` list).

Build/export with:   PYTHONPATH=. python analysis/export_model_gru.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from medhg_ps.data import PreprocessState, apply_preprocess

# --- sequence featurisation (must match cv_seq_gru / export_model_gru) -------
UNITS = ("ED", "Acute", "OR", "Intensive", "Intermediate", "Other")
U2I = {u: i for i, u in enumerate(UNITS)}
PAD = len(UNITS)                 # padding token index (embedding padding_idx)
MAXLEN = 40                      # max trajectory length (observed max 46; tail clipped)
EMB_DIM, HID, NUMF = 16, 32, 4   # unit-embed dim, GRU hidden, per-step numeric features


def visit_step_features(hours: float, arrival_hour: float, position: int) -> List[float]:
    """The 4 numeric per-visit features used at train and inference time."""
    return [float(np.log1p(max(float(hours or 0.0), 0.0))),
            float(np.sin(2 * np.pi * (arrival_hour or 0.0) / 24.0)),
            float(np.cos(2 * np.pi * (arrival_hour or 0.0) / 24.0)),
            (position + 1) / MAXLEN]


def build_seq_arrays(visits: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build (idx[MAXLEN], num[MAXLEN, NUMF], length) for one encounter.

    `visits` is the time-ordered list of care-unit visits, each a dict with
    keys `unit` (coarse bucket string), `hours` (length of stay), and
    `arrival_hour` (hour-of-day of arrival, 0-24). An empty list yields a
    single PAD step (length 1), the "no trajectory" encoding.
    """
    idx = np.full(MAXLEN, PAD, dtype=np.int64)
    num = np.zeros((MAXLEN, NUMF), dtype=np.float32)
    steps = list(visits)[:MAXLEN]
    for j, v in enumerate(steps):
        idx[j] = U2I.get(str(v.get("unit", "Other")), U2I["Other"])
        num[j] = visit_step_features(v.get("hours", 0.0), v.get("arrival_hour", 0.0), j)
    return idx, num, max(len(steps), 1)


class SeqGRU(nn.Module):
    """GRU over the unit trajectory; `encode` returns the final hidden state."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(PAD + 1, EMB_DIM, padding_idx=PAD)
        self.gru = nn.GRU(EMB_DIM + NUMF, HID, batch_first=True)
        self.head = nn.Linear(HID, 2)

    def encode(self, idx: torch.Tensor, num: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.emb(idx), num], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]

    def forward(self, idx: torch.Tensor, num: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(idx, num, lens))


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


@dataclass
class ReadmissionGRUModel:
    """Self-contained GRU-care-path readmission predictor (needs torch to load)."""
    gru: SeqGRU                            # trained encoder (eval mode, cpu)
    preprocess_state: PreprocessState      # tabular NSQIP transform
    tab_feat_cols: List[str]               # tabular columns fed to apply_preprocess
    cpt_encoder: OneHotEncoder             # PrimaryCPT one-hot
    scaler: StandardScaler                 # over [tab | cpt | gru_emb]
    clf: Any                               # fitted isotonic-calibrated classifier
    n_features: int
    base_rate: float = 0.0
    cv_metrics: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.10                # default operating point (DCA-informed)
    version: str = "1.0"

    def _seq_embed(self, seqs: List[Sequence[Dict[str, Any]]]) -> np.ndarray:
        idx, num, lens = [], [], []
        for s in seqs:
            i, n, L = build_seq_arrays(s if isinstance(s, (list, tuple)) else [])
            idx.append(i); num.append(n); lens.append(L)
        self.gru.eval()
        with torch.no_grad():
            emb = self.gru.encode(torch.tensor(np.stack(idx)),
                                  torch.tensor(np.stack(num)),
                                  torch.tensor(np.asarray(lens)))
        return emb.cpu().numpy()

    def predict_proba(self, records: List[Dict[str, Any]]) -> np.ndarray:
        if isinstance(records, dict):
            records = [records]
        df = pd.DataFrame(records)
        Xtab = apply_preprocess(df.reindex(columns=self.tab_feat_cols), self.preprocess_state)
        cpt = (_norm(df["PrimaryCPT"]).fillna("UNK") if "PrimaryCPT" in df.columns
               else pd.Series(["UNK"] * len(df)))
        Xcpt = self.cpt_encoder.transform(np.asarray(cpt.astype(str), dtype=object).reshape(-1, 1))
        seqs = df["postop_visits"] if "postop_visits" in df.columns else [[]] * len(df)
        Xemb = self._seq_embed(list(seqs))
        X = self.scaler.transform(np.hstack([Xtab, Xcpt, Xemb]))
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, records: List[Dict[str, Any]],
                threshold: Optional[float] = None) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(records) >= t).astype(int)
