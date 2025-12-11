from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import sys
sys.path.append("../")

from models.model import TimeSeriesTransformer
from utils.params import SEQ_LEN

import time


def load_model(feature_dim: int,
               hold_days: int,
               path: str = '../results/model/') -> nn.Module:
    """
    Load a pre-trained TimeSeriesTransformer model from disk.

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
    checkpoint_path = f'{path}model_{hold_days}.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    model.eval()  # Optional: set model to evaluation mode

    return model


def model_prediction(tickers: List[str],
                     full_df: pd.DataFrame,
                     feature_dim: int,
                     hold_days: List[int],
                     n_regimes: int = 3,
                     seq_len: int = SEQ_LEN,
                     path: str = '../data/processed/',
                     verbose: bool = True) -> None:
    """
    Generate model predictions for multiple tickers and horizons, and save to CSV.

    Parameters
    ----------
    tickers : List[str]
        List of tickers to predict.
    full_df : pd.DataFrame
        Full feature DataFrame indexed by Date and Ticker.
    feature_dim : int
        Number of input features for the model.
    hold_days : List[int]
        List of forward horizons to predict.
    n_regimes : int, default=3
        Number of HMM regimes used in the model.
    seq_len : int, default=SEQ_LEN
        Length of input sequences.
    path : str, default='../data/processed/'
        Path to save the predicted CSV.
    verbose : bool, default=True
        If True, prints progress and status.
    """
    start = time.perf_counter()
    dfs = {}

    # Loop over each prediction horizon
    for t in hold_days:
        model = load_model(feature_dim, t)
        model.eval()  # Ensure evaluation mode

        pred_rows = []

        # Loop over tickers
        for ticker_id, ticker in enumerate(tickers):
            df = full_df.xs(ticker, level='Ticker')
            
            # Identify feature columns
            regime_cols = [f'Regime_{i}_prob' for i in range(n_regimes)]
            features = [c for c in df.columns if c not in ['Label', 'Ticker'] + regime_cols]

            values = df[features].values

            # Generate rolling sequence predictions
            for i in range(seq_len, len(df)):
                seq = values[i-seq_len:i]
                seq = (seq - seq.mean(axis=0)) / np.clip(seq.std(axis=0), 1e-5, None)

                regime_seq = df[regime_cols].iloc[i-seq_len:i].values.astype(np.float32)

                # Convert to torch tensors
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                regime_tensor = torch.tensor(regime_seq, dtype=torch.float32).unsqueeze(0)
                stock_tensor = torch.tensor([ticker_id], dtype=torch.long)

                # Model inference
                with torch.inference_mode():
                    pred = model(seq_tensor, stock_tensor, regime_tensor)
                    prob_up = torch.softmax(pred, dim=1)[0, 1].item()

                # Append prediction row
                pred_rows.append({
                    "Date": df.index[i],
                    "Ticker": ticker,
                    f"Prediction_{t}": prob_up
                })

        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(pred_rows)
        pred_df['Date'] = pd.to_datetime(pred_df['Date'])
        pred_df = pred_df.set_index(['Date', 'Ticker']).sort_index()

        dfs[t] = pred_df

    # Merge all horizons into one DataFrame
    pred_df_full = dfs[hold_days[0]].copy()
    for t in hold_days[1:]:
        pred_df_full = pred_df_full.join(dfs[t], how="outer")

    # Merge with full_df for reference and drop missing values
    merged = full_df.merge(pred_df_full, on=["Date", "Ticker"], how="left").dropna()

    # Save to CSV
    output_path = f'{path}predicted_df.csv'
    merged.to_csv(output_path)

    end = time.perf_counter()

    if verbose:
        print(f'Predicted dataframe saved: {output_path}')
        print(f"Time taken: {end - start:.2f} seconds")