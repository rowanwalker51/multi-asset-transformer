from typing import Dict, List, Tuple
import time

import pandas as pd
import numpy as np
from hmmlearn import hmm

from src.common.config import CommonConfig, load_yaml, get_config_path
from src.data.config import load_data_config


# Load YAML files
common_cfg = CommonConfig(**load_yaml(get_config_path("common.yaml"))["common"])
data_cfg = load_data_config(get_config_path("data.yaml")


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Load raw market and macro datasets from disk and return them as a 
    dictionary of DataFrames.

    Each parquet is expected to have a Date index and a 'Close' column, which is
    renamed to a consistent series identifier (e.g., 'rf', 'ftse'). The function
    does not perform any validation beyond loading and renaming.
    """
    raw_paths = data_cfg['paths']['raw']
    
    # Risk-free rate
    rf_file = 'rf.parquet'
    rf = (pd.read_parquet(raw_paths['rf'] / rf_file)
            .rename(columns={'Close': 'rf'})
            .set_index('Date'))
    rf.index = pd.to_datetime(rf.index)

    # FTSE index benchmark
    index_file = 'index.parquet'
    ftse = (pd.read_parquet(raw_paths['base'] / index_file)
              .rename(columns={'Close': 'ftse'})
              .set_index('Date'))
    ftse.index = pd.to_datetime(ftse.index)

    # FX rates
    gbp_usd_file = 'gbp_usd.parquet'
    gbp_usd = (pd.read_parquet(raw_paths['fx'] / gbp_usd_file)
                 .rename(columns={'Close': 'gbp_usd'})
                 .set_index('Date'))
    gbp_usd.index = pd.to_datetime(gbp_usd.index)
    
    gbp_eur_file = 'gbp_eur.parquet'
    gbp_eur = (pd.read_parquet(raw_paths['fx'] / gbp_eur_file)
                 .rename(columns={'Close': 'gbp_eur'})
                 .set_index('Date'))
    gbp_eur.index = pd.to_datetime(gbp_eur.index)
    
    # Commodities
    gold_file = 'gold.parquet'
    gold = (pd.read_parquet(raw_paths['commodities'] / gold_file)
              .rename(columns={'Close': 'gold'})
              .set_index('Date'))
    gold.index = pd.to_datetime(gold.index)
    
    oil_file = 'oil.parquet'
    oil = (pd.read_parquet(raw_paths['commodities'] / oil_file)
             .rename(columns={'Close': 'oil'})
             .set_index('Date'))
    oil.index = pd.to_datetime(oil.index)
    
    # Collect into a single mapping
    data_dict = {
        'rf': rf,
        'ftse': ftse,
        'gbp_usd': gbp_usd,
        'gbp_eur': gbp_eur,
        'gold': gold,
        'oil': oil
    }

    return data_dict


def create_features(ticker: str,
                    hold_days: int) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        Feature matrix with predictors and the final label column.
    """
    raw_paths = data_cfg['paths']['raw']
    
    # FTSE index benchmark
    index_file = 'index.parquet'
    ftse = (pd.read_parquet(raw_paths['base'] / index_file)
              .rename(columns={'Close': 'ftse'})
              .set_index('Date'))
    ftse.index = pd.to_datetime(ftse.index)
    
    # Load raw price data for the ticker
    df = pd.read_parquet(raw_paths['ftse'] / f'{ticker}.parquet')
    df.index = pd.to_datetime(df.index)

    # Short, medium, long-term windows
    sml = (5, 21, 60)

    df = df.join(ftse[['ftse']], how='left')

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

    # Remove unused columns
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
                           num_stocks: int = common_cfg.num_stocks) -> List[str]:
    """
    Return a list of tickers that have sufficient historical data 
    between `start_date` and `end_date`.

    A ticker is considered valid if its price series contains an
    adequate number of observations within the date window.
    """
    # Load full FTSE ticker list
    all_tickers_path = data_cfg['paths']['raw']['base']
    all_tickers_file = 'all_tickers.parquet'
    all_tickers = pd.read_parquet(all_tickers_path / all_tickers_file)['ticker'].to_list()

    valid_tickers: List[str] = []

    # Number of expected trading days in the window
    min_fill = (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year) * 252

    ticker_data_path = data_cfg['paths']['raw']['ftse']
    
    for ticker in all_tickers:
        # Load individual price series
        df = pd.read_parquet(ticker_data_path / f'{ticker}.parquet')

        df.index = pd.to_datetime(df.index)

        # Check if the ticker has enough data in the date range
        if len(df.loc[start_date:end_date]) > min_fill:
            valid_tickers.append(ticker)

    return valid_tickers[:num_stocks]


def hmm_features() -> pd.DataFrame:
    """
    Generate technical features for Hidden Markov Model (HMM) regime analysis.

    Features include:
        - Exponentially weighted log returns over multiple windows
        - Price and volume moving averages
        - Price-volume correlation
        - RSI (14-day)
        - Rolling volatility

    Returns
    -------
    pd.DataFrame
        DataFrame containing the engineered features, indexed by Date.
    """
    path = data_cfg['paths']['raw']['base']
    file = 'index.parquet'
    
    # Load price data
    df = pd.read_parquet(path / file).set_index('Date')
    df.index = pd.to_datetime(df.index)

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


def hmm_model(df: pd.DataFrame, 
              n_regimes: int) -> pd.DataFrame:
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

    # Standardise features
    mean = df_hmm.mean(axis=0)
    std = np.clip(df_hmm.std(axis=0), 1e-5, None)
    df_hmm_norm = (df_hmm - mean) / std

    # Fit Gaussian HMM
    model = hmm.GaussianHMM(n_components=n_regimes,
                            covariance_type="full",
                            n_iter=10000,
                            tol=1e-4,
                            random_state=common_cfg.random_seed)

    model.fit(df_hmm_norm)

    # Compute probabilities for each regime
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
    n_regimes: int = common_cfg.n_regimes,
    seq_len: int = common_cfg.seq_len,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate model-ready inputs for sequence-based transformer models.

    Returns
    -------
    X : np.ndarray
        Shape (num_samples, seq_len, num_features)
    y : np.ndarray
        Shape (num_samples,)
    stock_ids : np.ndarray
        Shape (num_samples,)
    regime_X : np.ndarray
        Shape (num_samples, seq_len, n_regimes)
    full_df : pd.DataFrame
        Full feature DataFrame indexed by (Date, Ticker)
    """
    start = time.perf_counter()

    X, y, stock_ids, regime_X = [], [], [], []
    dfs = {}

    # Build per-ticker features
    for ticker in tickers:
        df = create_features(ticker, hold_days)
        df["Ticker"] = ticker
        dfs[ticker] = df

    full_df = pd.concat(dfs.values())

    # Add macro + regime features before sequencing
    regime_df = hmm_model(hmm_features(), n_regimes)
    asset_data = load_raw_data()

    full_df = (
        full_df
        .join(asset_data["rf"][["rf"]], how="left")
        .join(asset_data["gbp_usd"][["gbp_usd"]], how="left")
        .join(asset_data["gbp_eur"][["gbp_eur"]], how="left")
        .join(asset_data["gold"][["gold"]], how="left")
        .join(asset_data["oil"][["oil"]], how="left")
        .dropna()
    )

    full_df = full_df.join(regime_df, how='left')

    full_df = full_df.set_index(["Ticker"], append=True).sort_index()

    # Build sequences
    regime_cols = [f"Regime_{i}_prob" for i in range(n_regimes)]

    for ticker_id, ticker in enumerate(tickers):
        ticker_df = full_df.xs(ticker, level="Ticker")
        train_df = ticker_df.loc[train_start:train_end]

        feature_cols = [
            c for c in train_df.columns
            if c not in regime_cols + ["Label"]
        ]

        for i in range(len(train_df) - seq_len - hold_days):
            seq = train_df[feature_cols].iloc[i:i + seq_len].values
            seq = (seq - seq.mean(axis=0)) / np.clip(seq.std(axis=0), 1e-5, None)

            X.append(seq)
            y.append(train_df["Label"].iloc[i + seq_len + hold_days - 1])
            stock_ids.append(ticker_id)
            regime_X.append(train_df[regime_cols].iloc[i:i + seq_len].values)

    # Convert to arrays
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    stock_ids = np.asarray(stock_ids, dtype=np.int64)
    regime_X = np.asarray(regime_X, dtype=np.float32)

    # Save features
    feature_path = data_cfg["paths"]["processed"]
    file_name = f'processed_{hold_days}.parquet'
    full_df.to_parquet(feature_path / file_name)

    # Sanity checks
    assert X.shape[0] == y.shape[0] == stock_ids.shape[0] == regime_X.shape[0]
    assert X.shape[1] == seq_len
    assert regime_X.shape[2] == n_regimes

    if verbose:
        elapsed = time.perf_counter() - start
        print("Load successful")
        print("----------------")
        print(f"Time taken: {elapsed:.2f}s")
        print(f"Samples: {X.shape[0]}")
        print(f"Features: {X.shape[2]}")
        print(f"Stocks: {len(set(stock_ids))}")

    return X, y, stock_ids, regime_X, full_df