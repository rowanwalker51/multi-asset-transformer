from typing import Tuple, Optional

import torch
import torch.nn as nn

from src.common.config import CommonConfig, load_yaml, get_config_path
from src.models.config import ModelConfig


# Load YAML files
common_cfg = CommonConfig(**load_yaml(get_config_path("common.yaml"))["common"])
model_cfg = ModelConfig(**load_yaml(get_config_path("model.yaml"))["model"])


def apply_rotary(sin: torch.Tensor, 
                 cos: torch.Tensor, 
                 x: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to the input tensor.

    Parameters
    ----------
    sin : torch.Tensor
        Sine embeddings of shape (seq_len, half_dim).
    cos : torch.Tensor
        Cosine embeddings of shape (seq_len, half_dim).
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, d_model), where
        d_model is even and split into two halves for rotation.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as `x` with rotary positional embeddings applied.
    """

    B, T, D = x.shape
    half = D // 2  # Split last dimension into two halves

    # Split tensor along the last dimension
    x1 = x[..., :half]
    x2 = x[..., half:]

    # Broadcast sine and cosine embeddings across batch dimension
    sin = sin[:T].unsqueeze(0)  # (1, T, half)
    cos = cos[:T].unsqueeze(0)  # (1, T, half)

    # Apply rotation and concatenate halves
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def build_rope(d_model: int, 
               seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct rotary positional embeddings (RoPE) for a transformer.

    Parameters
    ----------
    d_model : int
        Dimensionality of the model. Must be even.
    seq_len : int
        Length of the input sequence.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        sin, cos tensors of shape (seq_len, d_model//2) for rotary embeddings.
    """

    # Ensure model dimension is even for splitting
    assert d_model % 2 == 0, "d_model must be even for RoPE"

    half_dim = d_model // 2

    # Compute base frequencies for each pair of dimensions
    theta = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
    
    # Sequence positions
    t = torch.arange(seq_len).float().unsqueeze(1)  # (seq_len, 1)

    # Compute angles for sin/cos
    angles = t * theta  # (seq_len, half_dim/2)

    # Interleave sin/cos values into full half-dimension
    sin = torch.zeros(seq_len, half_dim)
    cos = torch.zeros(seq_len, half_dim)
    sin[:, ::2] = torch.sin(angles)
    cos[:, ::2] = torch.cos(angles)

    return sin, cos


class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for multivariate time series with stock and regime embeddings.
    Incorporates rotary positional embeddings (RoPE) and a CLS token for classification.
    """

    def __init__(self,
                 feature_dim: int,
                 d_model: int = model_cfg.d_model,
                 nhead: int = model_cfg.n_head,
                 num_layers: int = model_cfg.n_layers,
                 num_classes: int = model_cfg.n_classes,
                 seq_len: int = common_cfg.seq_len,
                 num_stocks: int = common_cfg.num_stocks,
                 n_regimes: int = common_cfg.n_regimes,
                 dropout: float = model_cfg.dropout):
        super().__init__()

        self.seq_len = seq_len

        # Input projection to model dimension
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Stock embedding
        self.stock_emb = nn.Embedding(num_stocks, d_model)
        self.stock_proj = nn.Linear(d_model, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # LayerNorm before transformer
        self.pre_ln = nn.LayerNorm(d_model)

        # Rotary positional embeddings
        sin, cos = build_rope(d_model, seq_len + 1)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

        # Regime embedding projection
        self.regime_proj = nn.Linear(n_regimes, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=d_model * 4,
                                                   dropout=dropout,
                                                   batch_first=True,
                                                   norm_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Final classification layer
        self.fc_head = nn.Sequential(nn.LayerNorm(d_model),
                                     nn.Linear(d_model, d_model // 2),
                                     nn.GELU(),
                                     nn.Linear(d_model // 2, num_classes))

    
    def forward(self,
                x: torch.Tensor,
                stock_id: torch.Tensor,
                regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, feature_dim)
        stock_id : torch.Tensor
            Stock indices of shape (batch_size,)
        regime_probs : torch.Tensor
            Regime probability sequences of shape (batch_size, seq_len, n_regimes)

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes)
        """

        B, T, _ = x.size()

        # Project input features
        x = self.input_proj(x)

        # Stock embedding
        stock_vec = self.stock_emb(stock_id)          # (B, d_model)
        stock_vec = self.stock_proj(stock_vec)        # (B, d_model)
        stock_vec = stock_vec.unsqueeze(1).repeat(1, T, 1)  # (B, T, d_model)
        x = x + stock_vec

        # Regime embedding
        regime_emb = self.regime_proj(regime_probs)  # (B, T, d_model)

        # CLS token regime placeholder for shape consistency
        cls_regime = torch.zeros(B, 1, regime_emb.size(-1), device=x.device)

        # Pre-Layer normalisation before adding CLS
        x = self.pre_ln(x)

        # Add CLS token
        cls = self.cls_token.repeat(B, 1, 1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)       # (B, T+1, d_model)

        # Append regime embedding for CLS token
        regime_emb = torch.cat([cls_regime, regime_emb], dim=1)  # (B, T+1, d_model)
        x = x + regime_emb

        # Apply rotary positional embeddings excluding CLS
        cls, seq = x[:, :1], x[:, 1:]

        sin = self.rope_sin[:T]
        cos = self.rope_cos[:T]

        seq = apply_rotary(sin, cos, seq)
        x = torch.cat([cls, seq], dim=1)

        # Dropout
        x = self.dropout(x)

        # Setting up causal mask
        mask = torch.triu(
            torch.ones(T+1, T+1, device=x.device),
            diagonal=1
        ).bool()
        
        # CLS attends to everything
        mask[0, :] = False

        # Transformer
        x = self.transformer(x, mask=mask)

        # Use CLS token output for classification
        cls_out = x[:, 0, :]

        return self.fc_head(cls_out)