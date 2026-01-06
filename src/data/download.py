from typing import List

import pandas as pd
import yfinance as yf

from src.data.config import load_data_config


data_cfg = load_data_config("../configs/data.yaml")


def download_ftse() -> None:
    """
    Download historical daily price data for all FTSE 100 constituents from Yahoo Finance
    and save each ticker as a parquet file.

    Notes
    -----
        - Expects a parquet file with a 'ticker' column.
        - Downloads up to 25 years of daily data, adjusted for splits/dividends.
        - Skips tickers with no data and prints status messages.
    """

    # Load the full list of tickers
    all_tickers_path = data_cfg['paths']['raw']['all_tickers']
    all_tickers = pd.read_parquet(all_tickers_path)['ticker'].to_list()

    for ticker in all_tickers:
        print(f"Downloading {ticker} ...")

        # Download price history
        df = yf.download(ticker,
                         period="25y",
                         interval="1d",
                         auto_adjust=True)

        # Remove multi-index if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel('Ticker', axis=1)
            df.columns.name = None

        # Skip if no data was returned
        if df.empty:
            print(f"No data for {ticker}, skipping.")
            continue

        # Save parquet
        ticker_save_path = data_cfg['paths']['raw']['ftse']
        df.to_parquet(ticker_save_path / f'{ticker}.parquet')