import pandas as pd
import yfinance as yf
from typing import List


def download_ftse() -> None:
    """
    Download historical daily price data for all FTSE 100 constituents from Yahoo Finance
    and save each ticker as a CSV file in ../data/raw/ftse/.

    Notes
    -----
    - Expects a CSV file '../data/raw/ftse100_tickers.csv' with a 'ticker' column.
    - Downloads up to 25 years of daily data, adjusted for splits/dividends.
    - Skips tickers with no data and prints status messages.
    """

    # Load the full list of tickers
    all_tickers: List[str] = pd.read_csv('../data/raw/ftse100_tickers.csv')['ticker'].to_list()

    for ticker in all_tickers:
        print(f"Downloading {ticker} ...")

        # Download price history
        df = yf.download(
            ticker,
            period="25y",
            interval="1d",
            auto_adjust=True
        )

        # Remove multi-index if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel('Ticker', axis=1)
            df.columns.name = None

        # Skip if no data was returned
        if df.empty:
            print(f"No data for {ticker}, skipping.")
            continue

        # Save CSV
        df.to_csv(f"../data/raw/ftse/{ticker}.csv")