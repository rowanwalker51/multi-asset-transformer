from typing import Tuple, Sequence, Union, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import risk


def add_signal(long_threshold: float,
               short_threshold: float,
               horizons: Sequence[int] = (1, 5, 21),
               input_loc: str = '../data/processed/predicted_df.csv') -> pd.DataFrame:
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
    input_loc : str, default='../data/processed/predicted_df.csv'
        Path to the CSV file containing model predictions.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date and Ticker, including:
        - Ensemble ranking
        - Position (-1, 0, 1) based on thresholds
    """

    # Load predicted probabilities
    df = pd.read_csv(input_loc)

    # Compute rank within each date for each horizon
    for h in horizons:
        df[f"Rank_{h}"] = df.groupby("Date")[f"Prediction_{h}"].rank(pct=True)

    # Compute ensemble mean rank across horizons
    df["Ensemble"] = df[[f"Rank_{h}" for h in horizons]].mean(axis=1)

    # Set MultiIndex for alignment
    df = df.set_index(['Date', 'Ticker']).sort_index()

    # Initialize positions
    df['Position'] = 0
    df.loc[df['Ensemble'] >= long_threshold, 'Position'] = 1
    df.loc[df['Ensemble'] <= short_threshold, 'Position'] = -1

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
                       target_vol: float = 0.10,
                       lookback: int = 21,
                       ann_factor: int = 252,
                       max_leverage: float = 1.0) -> pd.Series:
    """
    Calculate volatility-targeted portfolio weights based on signals.

    Parameters
    ----------
    signals : pd.Series
        Trading signals (+1, 0, -1) for each ticker, indexed by ticker.
    returns : pd.DataFrame
        Daily returns of tickers (same tickers as signals), indexed by date.
    target_vol : float, default=0.10
        Target annualized portfolio volatility.
    lookback : int, default=21
        Rolling window (in days) to compute volatility and covariance.
    ann_factor : int, default=252
        Annualization factor for volatility and covariance.
    max_leverage : float, default=1.0
        Maximum allowed sum of absolute weights.

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
             lookback: int = 21,
             initial_equity: float = 1000,
             sharpe_only: bool = False,
             output: bool = True,
             optimiser: bool = False,
             input_loc: str = '../data/processed/predicted_df.csv',
             benchmark_loc: str = '../data/raw/ftse_index.csv',
             rf_loc: str = '../data/raw/rf/rf.csv') -> Union[float, Tuple[pd.Series, pd.Series, pd.Series], Tuple[pd.Series, float]]:
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

    # Generate trading signals
    df = add_signal(long_threshold=long_threshold, 
                    short_threshold=short_threshold,
                    input_loc=input_loc)
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
    for i in range(lookback+1, len(prices)):
        date = prices.index[i]

        # --- Portfolio-level drawdown liquidation ---
        dd = equity / dd_peak - 1
        if dd < -max_drawdown:
            positions = {t: 0 for t in positions}
            entry_price = {t: None for t in entry_price}
            holding_days = {t: 0 for t in holding_days}
        dd_peak = max(dd_peak, equity)

        # --- Volatility-targeted weights ---
        todays_signals = signals.loc[date]
        window_returns = returns.iloc[:i]
        target_weights = vol_target_weights(
            todays_signals,
            window_returns,
            target_vol=target_vol,
            lookback=lookback,
            ann_factor=252,
            max_leverage=fraction_per_trade
        )

        # --- Update positions and apply transaction costs ---
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

        # --- Increment holding days ---
        for t in positions:
            if positions[t] != 0:
                holding_days[t] += 1

        # --- Apply TP/SL and max holding days ---
        for t in positions:
            if positions[t] == 0 or entry_price[t] is None:
                continue
            current_price = prices.loc[date, t]
            pnl_return = (current_price - entry_price[t]) / entry_price[t] * np.sign(positions[t])
            if pnl_return >= take_profit or pnl_return <= stop_loss or holding_days[t] >= max_hold_days:
                positions[t] = 0
                entry_price[t] = None
                holding_days[t] = 0

        # --- Daily PnL update ---
        daily_ret = sum(positions[t] * returns.loc[date, t] for t in positions)
        equity *= (1 + daily_ret)
        equity_curve.append(equity)

    # Construct equity DataFrame
    curve = pd.Series(equity_curve, index=prices.index[lookback:])
    equity_df = pd.DataFrame(curve).rename(columns={0: 'Strategy_Equity'})
    equity_df.index = pd.to_datetime(equity_df.index)

    # Benchmark & risk-free
    benchmark = pd.read_csv(benchmark_loc, index_col='Date', parse_dates=True)[['Close']]
    benchmark.columns = ['Benchmark_Close']
    rf = pd.read_csv(rf_loc, index_col='Date', parse_dates=True).rename(columns={'Close':'rf'})

    equity_df = equity_df.join(benchmark, how='left').join(rf, how='left')
    equity_df["Benchmark_Equity"] = (1 + equity_df["Benchmark_Close"].pct_change(fill_method=None)).cumprod() * initial_equity

    strategy_equity = equity_df['Strategy_Equity']
    benchmark_equity = equity_df['Benchmark_Equity']
    rf_series = equity_df['rf']

    # Compute returns for Sharpe ratio / optimisation
    strategy_returns = risk.compute_returns(strategy_equity)

    if optimiser:
        return strategy_returns, risk.sharpe_ratio(strategy_returns, rf_series)
    if sharpe_only:
        return risk.sharpe_ratio(strategy_returns, rf_series)

    if output:
        plt.style.use('ggplot')
        print("Strategy final equity:", f'{strategy_equity.iloc[-1]:,.0f}')
        print("Buy-and-hold final equity:", f'{benchmark_equity.iloc[-1]:,.0f}')
        print('\n')
        print('Sharpe Ratio:', f'{risk.sharpe_ratio(strategy_returns, rf_series):,.2f}')
        alpha, beta = risk.compute_alpha_beta(strategy_equity, benchmark_equity, rf_series)
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
                    end_date: str) -> Tuple[Dict[str, Any], float]:
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
        returns, sharpe = backtest(
            param_grid=chosen,
            start_date=start_date,
            end_date=end_date,
            output=False,
            optimiser=True
        )

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