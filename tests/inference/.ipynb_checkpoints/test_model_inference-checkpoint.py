import pytest
import numpy as np
import pandas as pd
import torch

from src.inference.inference import model_inference
from src.models.model import TimeSeriesTransformer
from src.common.config import CommonConfig, load_yaml, get_config_path

# Load configs
common_cfg = CommonConfig(**load_yaml(get_config_path("common.yaml"))["common"])


@pytest.fixture
def dummy_full_df():
    """Generate a small MultiIndex DataFrame with features and regimes."""
    seq_len = 5
    dates = pd.date_range("2025-01-01", periods=seq_len + 2)  # > seq_len
    tickers = ["AAPL", "GOOG"]  # matches num_stocks
    index = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])

    feature_cols = [f"feat_{i}" for i in range(8)]
    df = pd.DataFrame(np.random.rand(len(index), len(feature_cols)), index=index, columns=feature_cols)

    # Add dummy regime probabilities
    for i in range(3):  # n_regimes
        df[f"Regime_{i}_prob"] = np.random.rand(len(index))

    # Optional label column
    df["Label"] = np.random.randint(0, 2, size=len(index))

    return df, feature_cols


@pytest.fixture
def dummy_tickers():
    return ["AAPL", "GOOG"]


@pytest.fixture
def dummy_hold_days():
    return [1, 2]


def test_model_inference_runs(monkeypatch, dummy_full_df, dummy_tickers, dummy_hold_days):
    """
    Run the inference loop with dummy data without saving/loading model checkpoints.
    """

    df, feature_cols = dummy_full_df

    # Mock load_model to return a fresh TimeSeriesTransformer
    def fake_load_model(feature_dim, hold_days, model_load_path=None):
        return TimeSeriesTransformer(
            feature_dim=feature_dim,
            num_stocks=len(dummy_tickers),
            n_regimes=common_cfg.n_regimes,
            seq_len=5  # matches dummy DF rolling window
        ).eval()

    monkeypatch.setattr("src.inference.inference.load_model", fake_load_model)

    # Run inference loop
    try:
        model_inference(
            tickers=dummy_tickers,
            full_df=df,
            feature_dim=len(feature_cols),
            hold_days=dummy_hold_days,
            n_regimes=common_cfg.n_regimes,
            seq_len=5,
            batch_size=2,
            verbose=False
        )
    except Exception as e:
        pytest.fail(f"Inference loop failed: {e}")