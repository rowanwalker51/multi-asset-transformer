from typing import Tuple, Sequence, Union, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.risk import compute_returns, sharpe_ratio, compute_alpha_beta
from src.common.config import get_config_path
from src.data.config import load_data_config


# Load YAML files
data_cfg = load_data_config(get_config_path("data.yaml"))
                           

from typing import Sequence
import pandas as pd

def add_signal(
    long_threshold: float,
    short_threshold: float,
    horizons: Sequence[int] = (1, 5, 21),
    challenger: bool = False,
    df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Generate trading signals based on ensemble ranking of model predictions.

    Parameters
    ----------
    long_threshold : float
        Threshold above which a long position (1) is taken.
    short_threshold : float
        Threshold below which a short position (-1) is taken.
    horizons : Sequence[int], default=(1,5,21)
        List of prediction horizons to include in ensemble ranking.
    challenger: bool, default=False
        If True, uses the challenger model inference.
    df: pd.DataFrame, default=None
        If provided, uses this DataFrame instead of reading from file - used for testing.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date and Ticker, including:
        - Ensemble ranking
        - Position (-1, 0, 1) based on thresholds
    """

    # If no DataFrame is provided, read from file
    if df is None:
        input_path = data_cfg['paths']['inference']
        input_file = 'inference_challenger.parquet' if challenger else 'inference.parquet'
        df = pd.read_parquet(input_path / input_file)

    # Initialize list to store ranks
    rank_cols = []

    # Compute ranks for each horizon if it exists
    for h in horizons:
        col = f"Prediction_{h}"
        if col not in df.columns:
            # Skip missing prediction columns instead of raising
            continue
        rank_col = f"Rank_{h}"
        df[rank_col] = df.groupby("Date")[col].rank(pct=True)
        rank_cols.append(rank_col)

    if not rank_cols:
        raise ValueError(
            f"No prediction columns found in DataFrame. Expected one of: "
            f"{[f'Prediction_{h}' for h in horizons]}"
        )

    # Compute ensemble rank as mean of available ranks
    df["EnsembleRank"] = df[rank_cols].mean(axis=1)

    # Generate positions based on thresholds
    df["Position"] = 0
    df.loc[df["EnsembleRank"] >= long_threshold, "Position"] = 1
    df.loc[df["EnsembleRank"] <= short_threshold, "Position"] = -1

    return df


def create_backtest_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare price and signal matrices for backtesting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns: 'Date', 'Ticker', 'Close', 'Position'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - prices: pivoted DataFrame of Close prices (index=Date, columns=Ticker)
        - signals: pivoted DataFrame of Position signals (index=Date, columns=Ticker),
                   filled missing values with 0 and cast to integers
    """

    # Pivot Close prices
    prices = pd.pivot_table(df, values='Close', columns='Ticker', index='Date')

    # Pivot Position signals, fill missing values with 0, ensure integer type
    signals = pd.pivot_table(df, values='Position', columns='Ticker', index='Date') \
                 .fillna(0).astype(int)

    return prices, signals


def vol_target_weights(signals: pd.Series,
                       returns: pd.DataFrame,
                       target_vol: float,
                       max_leverage: float,
                       lookback: int = 21,
                       ann_factor: int = 252) -> pd.Series:
    """
    Calculate volatility-targeted portfolio weights based on signals.

    Parameters
    ----------
    signals : pd.Series
        Trading signals (+1, 0, -1) for each ticker, indexed by ticker.
    returns : pd.DataFrame
        Daily returns of tickers (same tickers as signals), indexed by date.
    target_vol : float
        Target annualized portfolio volatility.
    max_leverage : float
        Maximum allowed sum of absolute weights.
    lookback : int, default=21
        Rolling window (in days) to compute volatility and covariance.
    ann_factor : int, default=252
        Annualization factor for volatility and covariance.

    Returns
    -------
    pd.Series
        Scaled weights for each ticker, volatility-targeted and leverage-limited.
    """

    tickers = signals.index

    # Rolling volatility for each ticker
    vol = returns[tickers].rolling(lookback).std().iloc[-1]
    # Replace zero volatility with median to avoid division errors
    vol = vol.replace(0, np.nan).fillna(vol.median())

    # Raw inverse-volatility-weighted signals
    raw = signals / vol

    if raw.abs().sum() == 0:
        return pd.Series(0, index=tickers)

    # Normalize weights to sum to 1 in absolute terms
    weights = raw / raw.abs().sum()

    # Compute current portfolio volatility using covariance matrix
    cov = returns.iloc[-lookback:].cov() * ann_factor
    curr_vol = np.sqrt(weights @ cov @ weights)

    if curr_vol == 0:
        return pd.Series(0, index=tickers)

    # Scale weights to match target volatility
    scaled = weights * (target_vol / curr_vol)

    # Enforce maximum leverage
    total_leverage = scaled.abs().sum()
    if total_leverage > max_leverage:
        scaled *= (max_leverage / total_leverage)

    return scaled


def backtest(param_grid: Dict[str, float],
             start_date: str,
             end_date: str,
             df: pd.DataFrame | None = None,
             lookback: int = 21,
             initial_equity: float = 1000,
             sharpe_only: bool = False,
             output: bool = True,
             optimiser: bool = False,
             challenger: bool = False) -> Union[float, Tuple[pd.Series, pd.Series, pd.Series], Tuple[pd.Series, float]]:
    """
    Perform a backtest of a signal-based trading strategy with risk management rules.

    Parameters
    ----------
    param_grid : dict
        Dictionary containing strategy parameters, e.g. thresholds, vol target, frictions.
    start_date : str
        Backtest start date (YYYY-MM-DD).
    end_date : str
        Backtest end date (YYYY-MM-DD).
    df: pd.DataFrame, default=None
        If provided, uses this DataFrame instead of reading from file - used for testing.
    lookback : int, default=21
        Rolling window for volatility calculation and position sizing.
    initial_equity : float, default=1000
        Starting capital for the strategy.
    sharpe_only : bool, default=False
        If True, returns only the strategy Sharpe ratio.
    output : bool, default=True
        If True, prints results and plots equity curve.
    optimiser : bool, default=False
        If True, returns daily returns and Sharpe ratio for optimization purposes.
    challenger: bool, default=False
        If True, uses the challenger model inference.

    Returns
    -------
    float or tuple
        Depending on options:
        - Sharpe ratio if `sharpe_only=True`
        - Tuple(strategy_equity, benchmark_equity, risk_free) if `sharpe_only=False` and `optimiser=False`
        - Tuple(returns, sharpe) if `optimiser=True`
    """

    # Extract strategy parameters
    long_threshold = param_grid['long_threshold']
    short_threshold = param_grid['short_threshold']
    target_vol = param_grid['target_vol']
    slippage_bps = param_grid['slippage']
    commission_bps = param_grid['commission']
    take_profit = param_grid['take_profit']
    stop_loss = param_grid['stop_loss']
    max_hold_days = param_grid['max_hold_days']
    max_drawdown = param_grid['max_drawdown']
    fraction_per_trade = param_grid['leverage']

    if df is not None:
        df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
    # Generate trading signals
    if df is None:
        df = add_signal(long_threshold=long_threshold, 
                        short_threshold=short_threshold,
                        challenger=challenger)
    else:
        if "Position" not in df.columns:
            df = add_signal(long_threshold=long_threshold,
                            short_threshold=short_threshold,
                            df=df)
    
    df = df.loc[start_date:end_date]

    # Prepare prices and position matrices
    prices, signals = create_backtest_data(df)

    # Daily returns
    returns = prices.pct_change(fill_method=None).fillna(0)

    equity_curve = [initial_equity]
    equity = initial_equity

    # Initialize per-ticker tracking
    positions = {t: 0 for t in prices.columns}
    entry_price = {t: None for t in prices.columns}
    holding_days = {t: 0 for t in prices.columns}

    dd_peak = equity  # Peak equity for drawdown tracking

    # Loop through each day in backtest
    start_idx = min(lookback+1, len(prices)-1)
    for i in range(start_idx, len(prices)):
        date = prices.index[i]

        # Portfolio-level drawdown liquidation
        dd = equity / dd_peak - 1
        if dd < -max_drawdown:
            positions = {t: 0 for t in positions}
            entry_price = {t: None for t in entry_price}
            holding_days = {t: 0 for t in holding_days}
        dd_peak = max(dd_peak, equity)

        # Volatility-targeted weights
        todays_signals = signals.loc[date]
        window_returns = returns.iloc[:i]
        target_weights = vol_target_weights(todays_signals,
                                            window_returns,
                                            target_vol=target_vol,
                                            lookback=lookback,
                                            ann_factor=252,
                                            max_leverage=fraction_per_trade)

        # Update positions and apply transaction costs
        for t in positions:
            prev_weight = positions[t]
            new_weight = target_weights[t]
            if prev_weight != new_weight:
                cost = equity * abs(new_weight - prev_weight) * (slippage_bps + commission_bps) / 10000
                equity -= cost
            positions[t] = new_weight
            if new_weight != 0 and entry_price[t] is None:
                entry_price[t] = prices.loc[date, t]
                holding_days[t] = 0

        # Increment holding days
        for t in positions:
            if positions[t] != 0:
                holding_days[t] += 1

        # Apply take-profit, stop-loss and max holding days
        for t in positions:
            if positions[t] == 0 or entry_price[t] is None:
                continue
            current_price = prices.loc[date, t]
            pnl_return = (current_price - entry_price[t]) / entry_price[t] * np.sign(positions[t])
            if pnl_return >= take_profit or pnl_return <= stop_loss or holding_days[t] >= max_hold_days:
                positions[t] = 0
                entry_price[t] = None
                holding_days[t] = 0

        # Daily PnL update
        daily_ret = sum(positions[t] * returns.loc[date, t] for t in positions)
        equity *= (1 + daily_ret)
        equity_curve.append(equity)

    if len(prices) <= lookback:
        curve = pd.Series(equity_curve[:len(prices)], index=prices.index)
    else:
        curve = pd.Series(equity_curve[lookback:], index=prices.index[lookback:])

    # Construct equity DataFrame
    equity_df = pd.DataFrame(curve).rename(columns={0: 'Strategy_Equity'})
    equity_df.index = pd.to_datetime(equity_df.index)

    # Benchmark & risk-free
    benchmark_path = data_cfg['paths']['raw']['base']
    benchmark_file = 'index.parquet'
    benchmark = pd.read_parquet(benchmark_path / benchmark_file).set_index('Date')[['Close']]
    benchmark.columns = ['Benchmark_Close']
    benchmark.index = pd.to_datetime(benchmark.index)

    rf_path = data_cfg['paths']['raw']['rf']
    rf_file = 'rf.parquet'
    rf = pd.read_parquet(rf_path / rf_file).set_index('Date').rename(columns={'Close':'rf'})
    rf.index = pd.to_datetime(rf.index)

    equity_df = equity_df.join(benchmark, how='left').join(rf, how='left')
    equity_df["Benchmark_Equity"] = (1 + equity_df["Benchmark_Close"].pct_change(fill_method=None)).cumprod() * initial_equity

    strategy_equity = equity_df['Strategy_Equity']
    benchmark_equity = equity_df['Benchmark_Equity']
    rf_series = equity_df['rf']

    # Compute returns for Sharpe ratio / optimisation
    strategy_returns = compute_returns(strategy_equity)

    if optimiser:
        return strategy_returns, sharpe_ratio(strategy_returns, rf_series)
    if sharpe_only:
        return sharpe_ratio(strategy_returns, rf_series)

    if output:
        plt.style.use('ggplot')
        print("Strategy final equity:", f'{strategy_equity.iloc[-1]:,.0f}')
        print("Buy-and-hold final equity:", f'{benchmark_equity.iloc[-1]:,.0f}')
        print('\n')
        print('Sharpe Ratio:', f'{sharpe_ratio(strategy_returns, rf_series):,.2f}')
        alpha, beta = compute_alpha_beta(strategy_equity, benchmark_equity, rf_series)
        print('Alpha (annualised):', f'{alpha:,.2f}%')
        print('Beta:', f'{beta:,.2f}')

        equity_df[['Strategy_Equity', 'Benchmark_Equity']].plot(figsize=(12,6))
        plt.title("Strategy vs FTSE 100")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.show()

    return strategy_equity, benchmark_equity, rf_series


def optimise_sharpe(params: Dict[str, Any],
                    trials: int,
                    start_date: str,
                    end_date: str,
                    df: pd.DataFrame | None = None,
                    challenger: bool = False) -> Tuple[Dict[str, Any], float]:
    """
    Randomly search over parameter space to find the combination 
    that maximises strategy Sharpe ratio.

    Parameters
    ----------
    params : dict
        Dictionary of parameters to optimise. Each value can be a list/array of options 
        or a fixed scalar.
    trials : int
        Number of random trials to perform.
    start_date : str
        Backtest start date in 'YYYY-MM-DD' format.
    end_date : str
        Backtest end date in 'YYYY-MM-DD' format.
    df: pd.DataFrame, default=None
        If provided, uses this DataFrame instead of reading from file - used for testing.
    challenger: bool, default=False
        If True, uses the challenger model inference.

    Returns
    -------
    best_params : dict
        Parameter combination with highest Sharpe ratio.
    best_sharpe : float
        Corresponding Sharpe ratio for the best parameter set.
    """
    
    results = []

    for i in range(1, trials + 1):
        # Randomly pick a value from each parameter list, or use fixed scalar
        chosen = {k: np.random.choice(v) if isinstance(v, (list, np.ndarray)) else v
                  for k, v in params.items()}
        
        # Run backtest in optimiser mode (returns daily returns and Sharpe)
        returns, sharpe = backtest(param_grid=chosen,
                                   start_date=start_date,
                                   end_date=end_date,
                                   output=False,
                                   optimiser=True,
                                   challenger=challenger,
                                   df=df)

        results.append({**chosen, 'Sharpe': sharpe})

        # Progress reporting
        if i == 1:
            print(f'Starting optimisation for {trials} trials...\n')
        if i % 10 == 0 or i == trials:
            print(f'Trial {i}/{trials} completed.')

    # Compile results into DataFrame and select the best Sharpe
    results_df = pd.DataFrame(results).sort_values(by='Sharpe', ascending=False).reset_index(drop=True)
    best_row = results_df.iloc[0]
    
    best_params = best_row.drop('Sharpe').to_dict()
    best_sharpe = best_row['Sharpe']

    return best_params, best_sharpe