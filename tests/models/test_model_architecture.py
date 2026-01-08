import pytest
import torch
from src.models.model import TimeSeriesTransformer, apply_rotary, build_rope

@pytest.fixture
def dummy_inputs_transformer():
    batch_size = 4
    seq_len = 10
    feature_dim = 6
    n_stocks = 5
    n_regimes = 3
    x = torch.randn(batch_size, seq_len, feature_dim)
    stock_id = torch.randint(0, n_stocks, (batch_size,))
    regime_probs = torch.rand(batch_size, seq_len, n_regimes)
    return x, stock_id, regime_probs, feature_dim, seq_len, n_stocks, n_regimes

    
def test_forward_pass_shape(dummy_inputs_transformer):
    x, stock_id, regime_probs, feature_dim, seq_len, n_stocks, n_regimes = dummy_inputs_transformer
    model = TimeSeriesTransformer(
        feature_dim=feature_dim,
        d_model=8,
        nhead=2,
        num_layers=1,
        num_classes=2,
        seq_len=seq_len,
        num_stocks=n_stocks,
        n_regimes=n_regimes,
        dropout=0.0
    )
    out = model(x, stock_id, regime_probs)
    assert out.shape == (x.size(0), 2)


def test_cls_token_added(dummy_inputs_transformer):
    x, stock_id, regime_probs, feature_dim, seq_len, n_stocks, n_regimes = dummy_inputs_transformer
    model = TimeSeriesTransformer(feature_dim=feature_dim,
                                  d_model=8,
                                  nhead=2,
                                  num_layers=1,
                                  num_classes=2,
                                  seq_len=seq_len,
                                  num_stocks=n_stocks,
                                  n_regimes=n_regimes)
    with torch.no_grad():
        out = model(x, stock_id, regime_probs)
    assert torch.isfinite(out).all()


def test_apply_rotary_preserves_shape():
    seq_len = 10
    d_model = 8
    half_dim = d_model // 2
    x = torch.randn(2, seq_len, d_model)
    sin, cos = build_rope(d_model, seq_len)
    y = apply_rotary(sin, cos, x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_embeddings_shapes(dummy_inputs_transformer):
    x, stock_id, regime_probs, feature_dim, seq_len, n_stocks, n_regimes = dummy_inputs_transformer
    model = TimeSeriesTransformer(feature_dim=feature_dim,
                                  d_model=8,
                                  nhead=2,
                                  num_layers=1,
                                  num_classes=2,
                                  seq_len=seq_len,
                                  num_stocks=n_stocks,
                                  n_regimes=n_regimes)
    stock_emb = model.stock_emb(stock_id)           # (B, d_model)
    regime_emb = model.regime_proj(regime_probs)    # (B, T, d_model)
    assert stock_emb.shape[0] == x.size(0)
    assert regime_emb.shape == (x.size(0), seq_len, 8)