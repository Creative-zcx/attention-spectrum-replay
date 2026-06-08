from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def compute_ap_af(score_matrix: np.ndarray) -> Dict[str, float]:
    """Compute AP and AF from a stage x task matrix m[a,b].

    Entries for unseen tasks may be 0 or nan; AF uses non-final tasks.
    """
    m = np.asarray(score_matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("score_matrix must be square [M,M]")
    M = m.shape[0]
    ap = float(np.nanmean(m[M - 1, :]))
    if M <= 1:
        af = 0.0
    else:
        drops = []
        for t in range(M - 1):
            hist = m[t:, t]
            drops.append(float(np.nanmax(hist) - m[M - 1, t]))
        af = float(np.mean(drops))
    return {"AP": ap, "AF": af}


def compute_last_avg(score_matrix: np.ndarray) -> Dict[str, float]:
    m = np.asarray(score_matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("score_matrix must be square [M,M]")
    M = m.shape[0]
    last = float(np.nanmean(m[M - 1, :]))
    seen_avgs = []
    for a in range(M):
        seen_avgs.append(float(np.nanmean(m[a, : a + 1])))
    avg = float(np.mean(seen_avgs))
    return {"Last": last, "Avg": avg}
