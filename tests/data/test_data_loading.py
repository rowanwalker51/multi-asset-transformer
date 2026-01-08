import pandas as pd

from src.data.preprocess import hmm_model


# Test macro data loader
def test_macro_data_has_datetime_index(macro_data):
    for name, df in macro_data.items():
        assert isinstance(df.index, pd.DatetimeIndex), f"{name} index is not DatetimeIndex"


def test_macro_data_no_duplicates(macro_data):
    for name, df in macro_data.items():
        assert df.index.is_unique, f"{name} index has duplicates"

# Test valid ticker creation
def test_valid_tickers_length(tickers):
    # Should return at least one ticker
    assert len(tickers) > 0, "No valid tickers found"


def test_create_features_columns(small_features):
    df = small_features
    # Must include Label
    assert "Label" in df.columns, "Feature DataFrame missing 'Label'"
    # Should have >5 features
    assert df.shape[1] > 5, "Too few features generated"
    # No NaNs
    assert not df.isna().any().any(), "Feature DataFrame contains NaNs"


# HMM feature testing
def test_hmm_features_output(small_hmm_features):
    df = small_hmm_features
    # Should have >0 columns
    assert df.shape[1] > 0
    # Index must be datetime
    assert isinstance(df.index, pd.DatetimeIndex)


def test_hmm_model_probabilities(small_hmm_features):
    df = small_hmm_features
    n_regimes = 2
    regime_df = hmm_model(df, n_regimes)
    # Columns match number of regimes
    assert regime_df.shape[1] == n_regimes
    # Probabilities are between 0 and 1
    assert (regime_df.values >= 0).all() and (regime_df.values <= 1).all()


# Test model inputs
def test_generate_model_inputs_shapes(small_model_input):
    X, y, stock_ids, regime_X, full_df = small_model_input
    # Consistency
    assert X.shape[0] == y.shape[0] == stock_ids.shape[0] == regime_X.shape[0]
    # Seq length
    assert X.shape[1] == 5  # seq_len
    # Regime columns
    assert regime_X.shape[2] == 2

def test_full_df_indexing(small_model_input):
    _, _, _, _, full_df = small_model_input
    # MultiIndex contains Ticker
    assert "Ticker" in full_df.index.names
    # Index is sorted
    assert full_df.index.is_monotonic_increasing