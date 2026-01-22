import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.data.preprocess import create_features


def time_split(df: pd.DataFrame, train_prop: float = 0.8):
    """
    Simple time-based split.
    """
    n = len(df)
    cutoff = int(n * train_prop)
    train = df.iloc[:cutoff].copy()
    test = df.iloc[cutoff:].copy()
    return train, test


def train_simple_threshold_model(train: pd.DataFrame, feature: str = "Log_return"):
    """
    Simple baseline model: Predict Label=1 if feature > 0 else 0
    """
    if feature not in train.columns:
        # Pick the first non-label numeric feature
        candidates = [c for c in train.columns if c != "Label"]
        if not candidates:
            raise ValueError("No feature columns found.")
        feature = candidates[0]

    return feature


def predict_threshold(df: pd.DataFrame, feature: str):
    return (df[feature].values > 0).astype(int)


@pytest.mark.leakage
def test_no_leakage_baseline_not_perfect():
    """
    Features should not allow near-perfect prediction out-of-sample using a trivial rule.
    """
    df = create_features(ticker="BARC.L", hold_days=5)

    train, test = time_split(df)

    feat = train_simple_threshold_model(train)
    y_pred = predict_threshold(test, feat)
    y_true = test["Label"].values

    acc = accuracy_score(y_true, y_pred)

    assert acc < 0.85, f"Suspiciously high accuracy ({acc:.3f}) on trivial baseline"


@pytest.mark.leakage
def test_permutation_test_accuracy_collapses():
    """
    Permutation test:
    Shuffle labels in TRAIN ONLY. If performance remains high then leakage is likely.
    """
    df = create_features(ticker="BARC.L", hold_days=5)

    train, test = time_split(df)

    # Permute train labels
    rng = np.random.default_rng(42)
    train_perm = train.copy()
    train_perm["Label"] = rng.permutation(train_perm["Label"].values)

    feat = train_simple_threshold_model(train_perm)
    y_pred = predict_threshold(test, feat)
    y_true = test["Label"].values

    acc = accuracy_score(y_true, y_pred)

    # ~0.5 for a binary task (allow noise)
    assert acc < 0.65, f"Permutation test too good ({acc:.3f}) -> possible leakage"


@pytest.mark.leakage
def test_shift_features_forward():
    """
    Shift test:
    Shifting features should not materially improve performance.
    """
    df = create_features(ticker="BARC.L", hold_days=5)

    train, test = time_split(df)

    feat = train_simple_threshold_model(train)

    # Normal prediction
    y_pred_normal = predict_threshold(test, feat)
    y_true = test["Label"].values
    acc_normal = accuracy_score(y_true, y_pred_normal)

    # Shift features backwards so test row t uses feature from t+1
    test_leaky = test.copy()
    test_leaky[feat] = test_leaky[feat].shift(-1)
    test_leaky = test_leaky.dropna()

    y_pred_leaky = predict_threshold(test_leaky, feat)
    y_true_leaky = test_leaky["Label"].values
    acc_leaky = accuracy_score(y_true_leaky, y_pred_leaky)

    assert (acc_leaky - acc_normal) < 0.10, (
        f"Shift test suspicious: normal={acc_normal:.3f}, leaky={acc_leaky:.3f}"
    )