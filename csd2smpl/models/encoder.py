"""Swappable marker-sequence encoder (markers → per-frame features)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MarkerTransformerEncoder(nn.Module):
    """Temporal Transformer over root-centred marker sequences.

    Parameters
    ----------
    n_markers : int
        Number of input markers per frame (e.g. 41 for CMU 41).
    d_model : int
        Transformer hidden size.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    dropout : float
        Dropout probability inside the Transformer block.
    max_seq_len : int
        Upper bound on sequence length for the positional embedding table.
    """

    def __init__(
        self,
        n_markers: int = 41,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.n_markers = n_markers
        self.d_model = d_model

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 3))

        self.input_proj = nn.Sequential(
            nn.Linear(n_markers * 3, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        markers: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a marker sequence into per-frame features.

        Parameters
        ----------
        markers : torch.Tensor
            Shape ``(B, T, M, 3)``.
        mask : torch.Tensor or None
            Shape ``(B, T, M)``, dtype bool. ``True`` = marker valid.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, d_model)``.
        """
        b, t, m, _ = markers.shape

        if mask is not None:
            occluded = ~mask.unsqueeze(-1)  # (B, T, M, 1)
            markers = markers.masked_fill(occluded, 0.0) + \
                self.mask_token * occluded.float()

        x = markers.reshape(b, t, m * 3)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :t, :]
        x = self.transformer(x)
        return self.norm(x)


# Back-compat alias for the old class name.
JointTransformerEncoder = MarkerTransformerEncoder
