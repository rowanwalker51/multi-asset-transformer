from typing import List, Optional
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.models.model import TimeSeriesTransformer
from src.common.config import CommonConfig, load_yaml, get_config_path
from src.data.config import load_data_config
from src.inference.config import InferenceConfig
from src.training.config import TrainingConfig


# Load YAML files
common_cfg = CommonConfig(
    **load_yaml(get_config_path("common.yaml"))["common"]
)
inference_cfg = InferenceConfig(
    **load_yaml(get_config_path("inference.yaml"))["inference"]
)
train_cfg = TrainingConfig(
    **load_yaml(get_config_path("training.yaml"))["training"]
)
data_cfg = load_data_config(get_config_path("data.yaml"))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_model(
    feature_dim: int,
    hold_days: int,
    model_load_path: str = train_cfg.model_save_path
) -> nn.Module:
    """
    Load a pre-trained TimeSeriesTransformer model.

    Parameters
    ----------
    feature_dim : int
        Number of input features for the model.
    hold_days : int
        Forward horizon used in the model (used to identify the saved model file).
    path : str, default='../results/model/'
        Directory path where the model checkpoint is saved.

    Returns
    -------
    nn.Module
        Loaded TimeSeriesTransformer model ready for inference or further training.
    """
    # Initialize model architecture
    model = TimeSeriesTransformer(feature_dim)

    # Load trained weights
    load_path = PROJECT_ROOT / model_load_path
    file_name = f"model_{hold_days}.pth"
    model.load_state_dict(
        torch.load(load_path / file_name, map_location="cpu"),
        strict=True
    )

    # Set model to evaluation mode
    model.eval()

    return model


def model_inference(
    tickers: List[str],
    full_df: pd.DataFrame,
    feature_dim: int,
    hold_days: List[int],
    n_regimes: int = common_cfg.n_regimes,
    seq_len: int = common_cfg.seq_len,
    batch_size: int = inference_cfg.batch_size,
    verbose: bool = True
) -> None:
    """
    Generate model predictions for multiple tickers and horizons, and save to CSV.

    Parameters
    ----------
    tickers : List[str]
        Asset identifiers corresponding to the Ticker level in full_df.
    full_df : pd.DataFrame
        Feature DataFrame indexed by Date and Ticker.
    feature_dim : int
        Number of model input features.
    hold_days : List[int]
        Forward prediction horizons corresponding to saved model checkpoints.
    n_regimes : int, default=3
        Number of HMM regime probability columns.
    seq_len : int, default=SEQ_LEN
        Length of the rolling input window.
    batch_size : int, default=256,
        Batch size for inference loop.
    verbose : bool, default=True
        Whether to print timing and status information.

    Returns
    -------
    None
        Predictions are written to file.
    """
    start_time = time.perf_counter()

    # Identify feature and regime columns
    regime_cols = [f"Regime_{i}_prob" for i in range(n_regimes)]
    feature_cols = [c for c in full_df.columns if c not in ["Label", "Ticker"] + regime_cols]

    # Container for prediction DataFrames
    dfs = {}

    # Process each forecast horizon independently
    for horizon in hold_days:
        model = load_model(feature_dim, horizon).to(device)
        model.eval()

        prediction_rows = []

        # Iterate over assets to preserve ticker-specific embeddings
        for ticker_id, ticker in enumerate(tickers):
            df = full_df.xs(ticker, level="Ticker")

            values = df[feature_cols].values.astype(np.float32)
            regimes = df[regime_cols].values.astype(np.float32)

            # Build rolling windows for features and regime probabilities
            X = np.stack([values[i - seq_len:i] for i in range(seq_len, len(df))])
            R = np.stack([regimes[i - seq_len:i] for i in range(seq_len, len(df))])

            # Per-window normalisation to match training-time preprocessing
            mean = X.mean(axis=1, keepdims=True)
            std = np.clip(X.std(axis=1, keepdims=True), 1e-5, None)
            X = (X - mean) / std

            dates = df.index[seq_len:]

            # Batched inference
            for i in range(0, len(X), batch_size):
                x_batch = torch.from_numpy(X[i:i + batch_size]).to(device)
                r_batch = torch.from_numpy(R[i:i + batch_size]).to(device)

                s_batch = torch.full(
                    (x_batch.size(0),),
                    ticker_id,
                    dtype=torch.long,
                    device=device
                )

                with torch.inference_mode():
                    logits = model(x_batch, s_batch, r_batch)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                # Write batch outputs back to row format
                for date, prob in zip(dates[i:i + batch_size], probs):
                    prediction_rows.append({
                        "Date": date,
                        "Ticker": ticker,
                        f"Prediction_{horizon}": float(prob)
                    })

        # Assemble horizon-specific prediction DataFrame
        dfs[horizon] = (
            pd.DataFrame(prediction_rows)
            .set_index(["Date", "Ticker"])
            .sort_index()
        )

    # Merge all horizon predictions into a single table
    pred_df_full = dfs[hold_days[0]].copy()
    for horizon in hold_days[1:]:
        pred_df_full = pred_df_full.join(dfs[horizon], how="outer")

    # Merge predictions back into the full feature set
    merged = (
        full_df.merge(pred_df_full, on=["Date", "Ticker"], how="left")
        .dropna()
    )

    output_path = data_cfg["paths"]["inference"]
    output_file = "inference.parquet"
    merged.to_parquet(output_path / output_file)

    if verbose:
        elapsed = time.perf_counter() - start_time
        print(f"Predicted dataframe saved: {output_path}")
        print(f"Time taken: {elapsed:.2f} seconds")