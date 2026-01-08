import pytest
import torch
import numpy as np
import pandas as pd

from src.training.train import train_model, PROJECT_ROOT

@pytest.fixture
def dummy_data_training():
    """Generate small synthetic dataset for training tests."""
    num_samples = 20
    seq_len = 5
    feature_dim = 8
    n_stocks = 3
    n_regimes = 3

    X = np.random.rand(num_samples, seq_len, feature_dim).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples,))
    stock_ids = np.random.randint(0, n_stocks, size=(num_samples,))
    regime_X = np.random.rand(num_samples, seq_len, n_regimes).astype(np.float32)

    return X, y, stock_ids, regime_X

    
def test_train_model_runs(dummy_data_training, tmp_path):
    """
    Test that `train_model` runs end-to-end on small synthetic data
    and saves the model checkpoint to a temporary directory.
    """
    X, y, stock_ids, regime_X = dummy_data_training

    # Use tmp_path for safe testing
    model_save_path = tmp_path  # this will be a Path object
    hold_days = 1

    # Train model
    train_model(
        X, y, stock_ids, regime_X,
        hold_days=hold_days,
        epochs=1,
        batch_size=2,
        verbose=False,
        model_save_path=model_save_path  # <- override to tmp_path
    )

    # Check model file exists
    saved_file = model_save_path / f'model_{hold_days}.pth'
    assert saved_file.exists()