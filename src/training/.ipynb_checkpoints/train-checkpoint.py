from typing import Optional, List, Dict, Any
import math
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import numpy as np

from src.data.preprocess import create_features, generate_valid_tickers, generate_model_inputs
from src.inference.inference import model_inference
from src.models.model import TimeSeriesTransformer
from src.utils.backtest import backtest
from src.common.config import CommonConfig, load_yaml, get_config_path
from src.training.config import TrainingConfig
from src.utils.seed import seed_worker


# Load YAML files
common_cfg = CommonConfig(**load_yaml(get_config_path("common.yaml"))["common"])
train_cfg = TrainingConfig(**load_yaml(get_config_path("training.yaml"))["training"])


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def train_model(X: np.ndarray,
                y: np.ndarray,
                stock_ids: np.ndarray,
                regime_X: np.ndarray,
                hold_days: int,
                batch_size: int = train_cfg.batch_size,
                weight_decay: float = train_cfg.weight_decay,
                lr: float = train_cfg.lr,
                epochs: int = train_cfg.epochs,
                device: Optional[str] = device,
                model_save_path: str = train_cfg.model_save_path,
                verbose: bool = True) -> None:
    """
    Train a TimeSeriesTransformer model on multivariate time series data.

    Parameters
    ----------
    X : np.ndarray
        Input sequences of shape (num_samples, seq_len, feature_dim)
    y : np.ndarray
        Labels of shape (num_samples,)
    stock_ids : np.ndarray
        Stock indices of shape (num_samples,)
    regime_X : np.ndarray
        Regime sequences of shape (num_samples, seq_len, n_regimes)
    hold_days : int
        Forward horizon for the model (used in saving the model)
    batch_size : int, default=BATCH_SIZE
        Batch size for training
    weight_decay : float, default=WEIGHT_DECAY
        Weight decay for AdamW optimizer
    lr : float, default=LR
        Learning rate for optimizer
    epochs : int, default=EPOCHS
        Number of training epochs
    device : str, optional
        Device to train on
    path : str, default="../results/model/"
        Directory to save the trained model
    verbose : bool, default=True
        Whether to print training progress
    """
    # Convert data to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    stock_tensor = torch.tensor(stock_ids, dtype=torch.long)
    regime_tensor = torch.tensor(regime_X, dtype=torch.float32)

    # Create dataset and loader
    dataset = TensorDataset(X_tensor, 
                            stock_tensor, 
                            regime_tensor, 
                            y_tensor)
    
    g = torch.Generator()
    g.manual_seed(common_cfg.random_seed)
    
    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=os.cpu_count()
    )

    # Initialize model, optimizer, and loss function
    model = TimeSeriesTransformer(X.shape[2]).to(device)

    # Optimiser with weight decay
    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr
    )

    # Learning Rate Scheduler
    total_steps = epochs * len(loader)
    warmup_steps = int(0.1 * total_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Loss metric
    loss_fn = nn.CrossEntropyLoss()

    if verbose:
        print(f'Beginning training for {epochs} epochs on device "{device}"')
        print('~' * 60)

    # Training loop
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        total_loss = 0

        for xb, stock_id, regime, yb in loader:
            xb, stock_id, regime, yb = xb.to(device), stock_id.to(device), regime.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb, stock_id, regime)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()

        end = time.perf_counter()
        if verbose:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}. Time taken = {end - start:.2f}s")

    # Save trained model
    save_path = Path(PROJECT_ROOT / model_save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    file_name = f'model_{hold_days}.pth'
    torch.save(model.state_dict(), save_path / file_name)
    if verbose:
        print('~' * 60)
        print(f'Training completed. Model saved: {save_path / file_name}')


def generate_walk_forward_dates(start: str,
                                end: str,
                                train_length: int,
                                test_length: int) -> List[List[pd.Timestamp]]:
    """
    Generate walk-forward training and testing date ranges.

    Parameters
    ----------
    start : str
        Start date for the first training period (e.g., '2000-01-01').
    end : str
        End date for the last testing period (e.g., '2025-12-31').
    train_length : int
        Length of training period in years.
    test_length : int
        Length of testing period in years.

    Returns
    -------
    List[List[pd.Timestamp]]
        List of [train_start, train_end, test_end] date ranges for each walk-forward split.
    """

    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    dates = []

    # Generate rolling walk-forward splits with 1-year increments
    while start_date <= end_date - pd.DateOffset(years=train_length + test_length):
        train_start = start_date
        train_end = start_date + pd.DateOffset(years=train_length)
        test_end = start_date + pd.DateOffset(years=train_length + test_length)

        dates.append([train_start, train_end, test_end])

        # Move start date forward by 1 year for next split
        start_date += pd.DateOffset(years=1)

    return dates


def walk_forward_validation(param_grid: Dict[str, Any],
                            hold_days: int,
                            start: str = '2000-11-20',
                            end: str = '2025-11-20',
                            train_length: int = 5,
                            test_length: int = 1,
                            num_stocks: int = common_cfg.num_stocks,
                            seq_len: int = common_cfg.seq_len) -> pd.DataFrame:
    """
    Perform walk-forward validation for a trading model using sequential train/test splits.

    Parameters
    ----------
    param_grid : Dict[str, Any]
        Parameters for backtesting and optimization.
    hold_days : int
        Forward horizon for model predictions.
    start : str, default='2000-11-20'
        Start date for the first training period.
    end : str, default='2025-11-20'
        End date for the last testing period.
    train_length : int, default=5
        Training window length in years.
    test_length : int, default=1
        Testing window length in years.
    num_stocks : int, default=NUM_STOCKS
        Number of tickers to include per split.
    seq_len : int, default=SEQ_LEN
        Sequence length for model input.

    Returns
    -------
    pd.DataFrame
        DataFrame with walk-forward validation results indexed by training start date.
    """

    wf_data = []
    split_counter = 1
    total_trainings = pd.to_datetime(end).year - pd.to_datetime(start).year - train_length

    # Loop through each walk-forward split
    for start_train, end_train, end_test in generate_walk_forward_dates(
        start, end, train_length, test_length
    ):
        timer_start = time.perf_counter()
        print(f'Beginning walk-forward validation {split_counter} of {total_trainings}.'
              f'Train Window: {start_train.strftime("%Y-%m-%d")} to {end_train.strftime("%Y-%m-%d")}.')

        # Select valid tickers with sufficient data
        valid_tickers = generate_valid_tickers(
            start_date=start_train, 
            end_date=end_test,
            num_stocks=num_stocks
        )

        # Generate model inputs for the training period
        X, y, stock_ids, full_df = generate_model_inputs(
            tickers=valid_tickers,
            train_start=start_train,
            train_end=end_train,
            seq_len=seq_len,
            hold_days=HOLD_DAYS,
            verbose=False
        )

        # Train the model
        train_model(X, y, stock_ids, verbose=False)

        # Generate predictions
        model_inference(
            tickers=valid_tickers,
            full_df=full_df,
            feature_dim=X.shape[2],
            hold_days=HOLD_DAYS,
            verbose=False
        )

        # Backtest predictions and compute Sharpe ratio
        sharpe = backtest(
            param_grid,
            input_loc='../data/processed/predicted_df.csv',
            output=False,
            start_date=end_train,
            end_date=end_test,
            sharpe_only=True
        )

        # Save results
        wf_data.append({
            'Train Start': start_train,
            'Sharpe': sharpe
        })

        timer_end = time.perf_counter()
        print(f'Walk-forward validation {split_counter} finished. '
              f'Time taken = {timer_end - timer_start:.2f} seconds.\n')

        split_counter += 1

    # Return results as DataFrame indexed by training start date
    return pd.DataFrame(wf_data).set_index('Train Start').sort_index()




    