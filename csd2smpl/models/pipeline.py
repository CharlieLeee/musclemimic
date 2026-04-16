"""Full pipeline: markers → SMPL params (encoder + head)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from csd2smpl.models.encoder import MarkerTransformerEncoder
from csd2smpl.models.smpl_head import SMPLHead


class Markers2SMPL(nn.Module):
    """Swappable-encoder + fixed SMPL head pipeline.

    Config keys
    -----------
    n_markers : int     — input marker count
    n_smpl_joints : int — SMPL output joint count (always 24)
    n_betas : int       — shape coefficients
    d_model, n_heads, n_layers, dropout : Transformer hyperparams
    max_seq_len : int (optional) — positional-embedding capacity
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = MarkerTransformerEncoder(
            n_markers=cfg["n_markers"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            max_seq_len=cfg.get("max_seq_len", 4096),
        )
        self.head = SMPLHead(
            d_model=cfg["d_model"],
            n_joints=cfg["n_smpl_joints"],
            n_betas=cfg["n_betas"],
        )

    def forward(
        self,
        markers: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(poses, betas, trans)``. See :class:`SMPLHead` for shapes."""
        feat = self.encoder(markers, mask)
        return self.head(feat)


# Back-compat alias for the old class name.
CSD2SMPL = Markers2SMPL
