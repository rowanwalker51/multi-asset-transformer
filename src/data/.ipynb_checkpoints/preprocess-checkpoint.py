from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from hmmlearn import hmm

import sys
sys.path.append("../")

from utils.params import NUM_STOCKS, SEQ_LEN


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Load raw market and macro datasets from disk and return them as a 
    dictionary of DataFrames.

    Each CSV is expected to have a Date index and a 'Close' column, which is
    renamed to a consistent series identifier (e.g., 'rf', 'ftse'). The function
    does not perform any validation beyond loading and renaming.
    """
    # Risk-free rate series
    rf = (
        pd.read_csv('../data/raw/rf/rf.csv', index_col='Date', parse_dates=True)
        .rename(columns={'Close': 'rf'})
    )

    # FTSE index benchmark
    ftse = (
        pd.read_csv('../data/raw/ftse_index.csv', index_col='Date', parse_dates=True)
        .rename(columns={'Close': 'ftse'})
    )

    # FX rates
    gbp_usd = (
        pd.read_csv('../data/raw/fx/gbp_usd.csv', index_col='Date', parse_dates=True)
        .rename(columns={'Close': 'gbp_usd'})
    )
    gbp_eur = (
        pd.read_csv('../data/raw/fx/gbp_eur.csv', index_col='Date', parse_dates=True)
        .rename(columns={'Close': 'gbp_eur'})
    )

    # Commodities
    gold = (
        pd.read_csv('../data/raw/commodity/gold.csv', index_col='Date', parse_dates=True)
        .rename(columns={'Close': 'gold'})
    )
    oil = (
        pd.read_csv('../data/raw/commodity/oil.csv', index_col='Date', parse_dates=True)
        .rename(columns={'Close': 'oil'})
    )

    # Collect into a single mapping
    data_dict: Dict[str, pd.DataFrame] = {
        'rf': rf,
        'ftse': ftse,
        'gbp_usd': gbp_usd,
        'gbp_eur': gbp_eur,
        'gold': gold,
        'oil': oil,
    }

    return data_dict


asset_data=load_raw_data()


def create_features(ticker: str,
                    hold_days: int,
                    asset_data: Dict[str, pd.DataFrame] = asset_data,
                    input_path: str = '../data/raw/ftse/') -> pd.DataFrame:
    """
    Build the full feature set for a single ticker by combining its price series
    with macro inputs and a range of technical indicators.

    The function:
     - Loads the ticker's price history
     - Merges external macro series (rf, FTSE index, FX, commodities)
     - Computes rolling statistical features (returns, volatility, beta)
     - Builds technical indicators (RSI, moving averages, correlations)
     - Creates the binary label based on forward returns

    Parameters
    ----------
    ticker : str
        The equity ticker to process.
    hold_days : int
        Forward return horizon used to generate the classification label.
    asset_data : Dict[str, pd.DataFrame]
        Preloaded macro and market datasets keyed by name.
    input_path : str
        Directory containing individual ticker CSV files.

    Returns
    -------
    pd.DataFrame
        Feature matrix with engineered predictors and the final label column.
    """

    # Load raw price data for the ticker
    df = pd.read_csv(
        input_path + f'{ticker}.csv',
        index_col='Date',
        parse_dates=True
    )

    # Merge macroeconomic and benchmark inputs
    df = (
        df.join(asset_data['rf'][['rf']], how='left')
          .join(asset_data['ftse'][['ftse']], how='left')
          .join(asset_data['gbp_usd'][['gbp_usd']], how='left')
          .join(asset_data['gbp_eur'][['gbp_eur']], how='left')
          .join(asset_data['gold'][['gold']], how='left')
          .join(asset_data['oil'][['oil']], how='left')
          .dropna()
    )

    # Short, medium, long-term windows
    sml = (5, 21, 60)

    # Forward returns and label
    df[f'Return_{hold_days}d'] = df['Close'].pct_change(hold_days).shift(-hold_days)
    df['Label'] = (df[f'Return_{hold_days}d'] > 0).astype(int)

    # Log returns and smoothed returns
    df['Log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    for w in sml:
        df[f'Log_return_{w}'] = df['Log_return'].ewm(span=w, adjust=False).mean()

    # Price-based moving average features
    for w in sml:
        df[f'Price_{w}'] = df['Close'].rolling(w).mean() / df['Close'] - 1

    # Volume moving averages
    for w in sml:
        df[f'Volume{w}'] = df['Volume'].rolling(w).mean()

    # Price-volume correlation
    df['Vol_price_correlation'] = df['Close'].rolling(21).corr(df['Volume'])

    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Rolling volatility estimates
    for w in sml:
        df[f'Vol_{w}'] = df['Log_return'].rolling(w).std()

    # Temporal features
    df['Day_of_week'] = df.index.dayofweek
    df['Month_of_year'] = df.index.month

    # Benchmark returns and correlation
    df['Log_return_benchmark'] = np.log(df['ftse'] / df['ftse'].shift(1))
    df['Benchmark_correlation'] = df['Log_return'].rolling(21).corr(df['Log_return_benchmark'])

    # Rolling beta vs benchmark
    for w in sml:
        rolling_cov = df['Log_return'].rolling(w).cov(df['Log_return_benchmark'])
        rolling_var = df['Log_return_benchmark'].rolling(w).var()
        df[f'Beta_{w}'] = rolling_cov / rolling_var

    # Clean-up: remove unused columns
    df.dropna(inplace=True)
    df.drop(
        columns=[
            'High', 'Low', 'Open',
            'Log_return',
            'Log_return_benchmark',
            f'Return_{hold_days}d'
        ],
        inplace=True
    )

    # Final feature list
    features = [col for col in df.columns if col != 'Label']

    return df[features + ['Label']]


def generate_valid_tickers(start_date: str,
                           end_date: str,
                           num_stocks: int = NUM_STOCKS,
                           all_tickers_path: str = '../data/raw/ftse100_tickers.csv') -> List[str]:
    """
    Return a list of tickers that have sufficient historical data 
    between `start_date` and `end_date`.

    A ticker is considered valid if its price series contains an
    adequate number of observations within the date window.
    """
    # Load full FTSE ticker list
    all_tickers = pd.read_csv(all_tickers_path)['ticker'].to_list()

    valid_tickers: List[str] = []

    # Number of expected trading days in the window (1 year ≈ 252 days)
    min_fill = (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year) * 252

    for ticker in all_tickers:
        # Load individual price series
        df = pd.read_csv(
            f'../data/raw/ftse/{ticker}.csv',
            index_col='Date',
            parse_dates=True
        )

        # Check if the ticker has enough data in the date range
        if len(df.loc[start_date:end_date]) > min_fill:
            valid_tickers.append(ticker)

    return valid_tickers[:num_stocks]


def hmm_features(path: str = '../data/raw/ftse_index.csv') -> pd.DataFrame:
    """
    Generate technical features for Hidden Markov Model (HMM) regime analysis.

    Features include:
    - Exponentially weighted log returns over multiple windows
    - Price and volume moving averages
    - Price-volume correlation
    - RSI (14-day)
    - Rolling volatility

    Parameters
    ----------
    path : str
        Path to the FTSE index CSV file. The file must have a Date index
        and columns: ['Open', 'High', 'Low', 'Close', 'Volume'].

    Returns
    -------
    pd.DataFrame
        DataFrame containing the engineered features, indexed by Date.
    """

    # Load price data
    df = pd.read_csv(path, index_col='Date', parse_dates=True)

    # Define short, medium, long-term windows
    sml = (5, 21, 60)

    # Log returns and smoothed versions
    df['Log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    for w in sml:
        df[f'Log_return_{w}'] = df['Log_return'].ewm(span=w, adjust=False).mean()

    # Price-based moving averages
    for w in sml:
        df[f'Price_{w}'] = df['Close'].rolling(w).mean() / df['Close'] - 1

    # Volume moving averages
    for w in sml:
        df[f'Volume{w}'] = df['Volume'].rolling(w).mean()

    # Price-volume correlation (21-day rolling)
    df['Vol_price_correlation'] = df['Close'].rolling(21).corr(df['Volume'])

    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Rolling volatility
    for w in sml:
        df[f'Vol_{w}'] = df['Log_return'].rolling(w).std()

    # Drop raw columns to keep only engineered features
    df.dropna(inplace=True)
    df.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume', 'Log_return'], inplace=True)

    return df


def hmm_model(df: pd.DataFrame, n_regimes: int) -> pd.DataFrame:
    """
    Fit a Gaussian Hidden Markov Model (HMM) to the input features and 
    return the posterior probabilities of each regime.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of engineered features (e.g., technical indicators), indexed by Date.
    n_regimes : int
        Number of latent regimes (HMM components) to fit.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (len(df), n_regimes) containing the probability
        of each regime at each time step.
    """

    # Convert to float32 numpy array
    df_hmm = df.values.astype(np.float32)

    # Standardize features (zero mean, unit variance)
    mean = df_hmm.mean(axis=0)
    std = np.clip(df_hmm.std(axis=0), 1e-5, None)  # avoid division by zero
    df_hmm_norm = (df_hmm - mean) / std

    # Fit Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=10000,
        tol=1e-4
    )
    model.fit(df_hmm_norm)

    # Compute posterior probabilities for each regime
    states = model.predict_proba(df_hmm_norm)

    # Construct DataFrame with regime probability columns
    regime_cols = [f'Regime_{i}_prob' for i in range(n_regimes)]
    regime_df = pd.DataFrame(states, columns=regime_cols, index=df.index)

    return regime_df


def generate_model_inputs(
    tickers: List[str],
    train_start: str,
    train_end: str,
    hold_days: int,
    n_regimes: int = 3,
    seq_len: int = SEQ_LEN,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate model-ready inputs for sequence-based transformer models.

    This function:
    - Creates features for each ticker using technical indicators and macro data
    - Computes regime probabilities using HMM
    - Constructs sequences of features for input into a model
    - Normalizes features per sequence
    - Returns full DataFrame for reference

    Parameters
    ----------
    tickers : List[str]
        List of equity tickers to process.
    train_start : str
        Start date for training data (YYYY-MM-DD).
    train_end : str
        End date for training data (YYYY-MM-DD).
    hold_days : int
        Forward horizon to define the target label.
    n_regimes : int, default=3
        Number of HMM regimes.
    seq_len : int, default=SEQ_LEN
        Length of input sequences for the model.
    verbose : bool, default=True
        If True, prints dataset summary.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]
        X : np.ndarray
            Input sequences of shape (num_samples, seq_len, num_features)
        y : np.ndarray
            Target labels of shape (num_samples,)
        stock_ids : np.ndarray
            Stock identifier per sample
        regime_X : np.ndarray
            Sequence of regime probabilities of shape (num_samples, seq_len, n_regimes)
        full_df : pd.DataFrame
            Concatenated feature DataFrame for all tickers, indexed by Date and Ticker
    """

    # Containers for sequences, labels, and regime info
    X, y, stock_ids, regime_X = [], [], [], []
    dfs = {}

    # Generate regime probabilities once for all tickers
    regime_df = hmm_model(hmm_features(), n_regimes)

    for ticker_id, ticker in enumerate(tickers):
        # Create features per ticker
        df = create_features(ticker, hold_days)

        # Merge regime probabilities
        df = df.join(regime_df, how='left')
        df['Ticker'] = ticker
        dfs[ticker] = df

        # Filter to training period
        train_df = df.loc[pd.to_datetime(train_start):pd.to_datetime(train_end)]

        # Determine feature columns (exclude labels, tickers, regime probabilities)
        regime_cols = [f'Regime_{i}_prob' for i in range(n_regimes)]
        features = [col for col in train_df.columns if col not in ['Label', 'Ticker'] + regime_cols]

        # Construct rolling sequences
        for i in range(train_df.shape[0] - seq_len - hold_days):
            seq = train_df[features].iloc[i:i+seq_len].values
            # Normalize per sequence
            seq = (seq - seq.mean(axis=0)) / np.clip(seq.std(axis=0), 1e-5, None)

            X.append(seq)
            y.append(train_df['Label'].iloc[i + seq_len + hold_days - 1])
            stock_ids.append(ticker_id)

            regime_seq = train_df[regime_cols].iloc[i:i+seq_len].values
            regime_X.append(regime_seq)

    # Convert lists to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    stock_ids = np.array(stock_ids, dtype=np.int64)
    regime_X = np.array(regime_X, dtype=np.float32)

    # Full concatenated DataFrame
    full_df = pd.concat(dfs.values())
    full_df = full_df.set_index(['Ticker'], append=True).sort_index()
    full_df = full_df.loc[train_start:]

    # Summary output
    if verbose:
        print('Load successful:')
        print('~~~~~~~~~~~~~~~~~~~~~~')
        print(f'Number of rows: {X.shape[0]}')
        print(f'Number of features: {X.shape[2]}')
        print(f'Number of stocks: {len(np.unique(stock_ids))}')

    return X, y, stock_ids, regime_X, full_df