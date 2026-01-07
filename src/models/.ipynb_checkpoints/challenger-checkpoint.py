from typing import Tuple, Sequence

import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.common.config import CommonConfig, load_yaml, get_config_path


# Load YAML files
common_cfg = CommonConfig(**load_yaml(get_config_path("common.yaml"))["common"])


def prepare_challenger_data(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice a labelled feature DataFrame into train and test arrays for
    challenger (non-sequence) models.

    Assumes:
        - DateTimeIndex
        - Binary target column named 'Label'
        - All other columns are numeric features

    Returns
    -------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    """
    df = df.copy()

    train_df = df.loc[train_start:train_end]
    test_df = df.loc[test_start:test_end]

    X_train = train_df.drop(columns=["Label"]).values
    y_train = train_df["Label"].values

    X_test = test_df.drop(columns=["Label"]).values
    y_test = test_df["Label"].values

    # Sanity checks
    assert X_train.shape[0] == y_train.shape[0], "Train X/y row mismatch"
    assert X_test.shape[0] == y_test.shape[0], "Test X/y row mismatch"
    assert X_train.shape[1] == X_test.shape[1], "Train/test feature mismatch"

    return X_train, y_train, X_test, y_test


def train_xgb_challenger(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    random_state: int = common_cfg.random_seed
) -> np.ndarray:
    """
    Train an XGBoost challenger model using a time-series split and
    return out-of-sample probabilities.

    Returns
    -------
    np.ndarray
        Predicted probability of Label == 1 for the test period.
    """
    X_train, y_train, X_test, y_test = prepare_challenger_data(
        df=df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist"
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    param_grid = {
        "model__n_estimators": [35, 40, 45],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.05, 0.1, 0.15],
        "model__subsample": [0.15, 0.2, 0.25],
        "model__colsample_bytree": [0.8, 1.0],
        "model__gamma": [6.25, 6.5, 6.75],
        "model__reg_alpha": [0.0, 0.1, 0.5],
        "model__reg_lambda": [1.0, 1.5, 2.0],
    }

    cv = TimeSeriesSplit(n_splits=5)

    start = time.perf_counter()

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(X_train, y_train)

    elapsed = time.perf_counter() - start
    print(f"XGBoost GridSearch complete ({elapsed:.2f}s)")

    # Probability of the positive class
    return grid.best_estimator_.predict_proba(X_test)[:, 1]


def build_challenger_inference_data(
    probabilities: Sequence[float],
    processed_df: pd.DataFrame,
    test_start: str,
    test_end: str,
    hold_days: int,
    save_path
) -> None:
    """
    Attach challenger model probabilities to the processed feature
    DataFrame and save for downstream inference or ensembling.

    Parameters
    ----------
    probabilities : Sequence[float]
        Output probabilities from the challenger model.
    processed_df : pd.DataFrame
        Feature DataFrame used for modelling (must have DateTimeIndex).
    test_start : str
    test_end : str
    hold_days : int
    save_path : Path
        Location to save the inference parquet.
    """
    challenger = pd.DataFrame(
        {f"Prediction_{hold_days}": probabilities},
        index=processed_df.loc[test_start:test_end].index,
    )

    inference_df = (
        processed_df
        .loc[test_start:test_end]
        .drop(columns=["Label"], errors="ignore")
        .join(challenger, how="left")
    )

    inference_df.to_parquet(save_path)