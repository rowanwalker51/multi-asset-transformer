import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import load_raw_data, create_features, generate_valid_tickers, hmm_features, hmm_model, generate_model_inputs


@pytest.fixture(scope="session")
def macro_data():
    """Load raw macro and market data once."""
    return load_raw_data()

@pytest.fixture(scope="session")
def tickers():
    """Return a few valid tickers for testing purposes."""
    return generate_valid_tickers("2020-01-01", "2021-12-31", num_stocks=3)

@pytest.fixture
def small_features(tickers):
    """Return features for one ticker for hold_days=5."""
    return create_features(tickers[0], hold_days=5)

@pytest.fixture
def small_hmm_features():
    """Return a tiny HMM feature set."""
    df = hmm_features()
    return df.head(10)

@pytest.fixture
def small_model_input(tickers):
    """Return minimal model-ready sequences for transformer tests."""
    X, y, stock_ids, regime_X, full_df = generate_model_inputs(
        tickers=tickers,
        train_start="2020-01-01",
        train_end="2020-12-31",
        hold_days=5,
        n_regimes=2,
        seq_len=5,
        verbose=False
    )
    return X, y, stock_ids, regime_X, full_df



