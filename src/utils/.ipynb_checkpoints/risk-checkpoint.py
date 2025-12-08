from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
import statsmodels.api as sm


# Return Calculations

def compute_returns(prices: pd.Series) -> pd.Series:
    """Compute simple daily returns from a price series."""
    return prices.pct_change(fill_method=None).dropna()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series."""
    return np.log(prices / prices.shift(1)).dropna()



# Volatility and Ratios

def annualized_vol(returns: pd.Series, periods: int = 252) -> float:
    """Compute annualized volatility."""
    return returns.std() * np.sqrt(periods)


def sharpe_ratio(returns: pd.Series, rf: pd.Series, periods: int = 252) -> float:
    """Compute annualized Sharpe ratio."""
    rf_annual = rf / 100
    rf_daily = (1 + rf_annual)**(1/252) - 1
    excess_returns = returns - (rf / 100 / periods)
    return (excess_returns.mean() / excess_returns.std(ddof=1)) * np.sqrt(periods)


def sortino_ratio(returns: pd.Series, rf: pd.Series, periods: int = 252) -> float:
    """Compute annualized Sortino ratio."""
    rf_annual = rf / 100
    rf_daily = (1 + rf_annual)**(1/252) - 1
    excess_returns = returns - rf_daily
    downside_std = excess_returns[excess_returns < 0].std(ddof=1)
    mean_excess = excess_returns.mean()
    return mean_excess / downside_std * np.sqrt(periods)



# Drawdown Metrics

def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Compute drawdown curve."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return drawdown
    

def max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown from equity curve."""
    return (equity / equity.cummax() - 1).min()


def drawdown_duration(equity: pd.Series) -> int:
    """Compute maximum drawdown duration in days."""
    dd = equity / equity.cummax() - 1
    durations = (dd != 0).astype(int)
    return durations.groupby((durations != durations.shift()).cumsum()).sum().max()



# Trade Statistics

def win_rate(returns: pd.Series) -> float:
    return (returns > 0).mean()


def loss_rate(returns: pd.Series) -> float:
    return (returns < 0).mean()


def payoff_ratio(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = -returns[returns < 0]
    return wins.mean() / losses.mean() if len(losses) > 0 else np.nan


def profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return wins / losses if losses != 0 else np.nan


def tail_ratio(returns: pd.Series) -> float:
    top5 = np.percentile(returns, 95)
    bottom5 = abs(np.percentile(returns, 5))
    return top5 / bottom5 if bottom5 != 0 else np.nan



# Value at Risk (VaR) / CVaR

def var_historical(returns: pd.Series, level: float = 0.05) -> float:
    return np.percentile(returns, level * 100)


def var_gaussian(returns: pd.Series, level: float = 0.05) -> float:
    return returns.mean() + returns.std() * norm.ppf(level)


def var_cornish_fisher(returns: pd.Series, level: float = 0.05) -> float:
    z = norm.ppf(level)
    s = skew(returns)
    k = kurtosis(returns, fisher=True)
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3*z) * k / 24
        - (2*z**3 - 5*z) * (s**2) / 36
    )
    return returns.mean() + returns.std() * z_cf


def cvar_historical(returns: pd.Series, level: float = 0.05) -> float:
    v = var_historical(returns, level)
    return returns[returns < v].mean()



# Alpha / Beta Calculation

def compute_alpha_beta(strategy: pd.Series, benchmark: pd.Series, rf: pd.Series) -> Tuple[float, float]:
    """
    Compute annualized alpha and beta of strategy vs benchmark, adjusting for risk-free rate.
    Returns:
        alpha (float): annualized alpha
        beta (float): beta coefficient
    """
    strat_returns = compute_returns(strategy).to_frame(name='returns')
    bmark_returns = compute_returns(benchmark).to_frame(name='bmark')
    rf_annual = rf / 100
    rf_daily = (1 + rf_annual)**(1/252) - 1
    rf_returns = rf_daily.to_frame(name='rf')

    df = strat_returns.join(bmark_returns, how='inner') \
                      .join(rf_returns, how='inner') \
                      .replace([np.inf, -np.inf], np.nan).dropna()

    excess_strategy = df['returns'] - df['rf']
    excess_benchmark = df['bmark'] - df['rf']

    X = sm.add_constant(excess_benchmark)
    model = sm.OLS(excess_strategy, X).fit()
    
    alpha_daily = model.params['const']
    beta = model.params[0]
    
    # Annualize alpha
    alpha_annual = (1 + alpha_daily)**252 - 1
    return alpha_annual, beta



# Rolling Metrics

def rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    return (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)


def rolling_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)



# Full Risk Report

def full_risk_report(strategy: pd.Series, benchmark: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """
    Compute a comprehensive risk report including returns, volatility, drawdowns, 
    ratios, VaR, CVaR, skewness, and kurtosis for strategy and benchmark.
    """
    report = {}
    for col, series in zip(['Strategy', 'Benchmark'], [strategy, benchmark]):
        r = compute_returns(series)
        report[col] = {
            "Mean Daily Return": f"{r.mean() * 100:.2f}%",
            "Daily Vol": f"{r.std() * 100:.2f}%",
            "Annualized Vol": f"{annualized_vol(r) * 100:.2f}%",
            "Sharpe Ratio": round(sharpe_ratio(r, rf), 2),
            "Sortino Ratio": round(sortino_ratio(r, rf), 2),
            "Max Drawdown": f"{max_drawdown(series) * 100:.2f}%",
            "Drawdown days": drawdown_duration(series),
            "Win Rate": f"{win_rate(r) * 100:.2f}%",
            "Historical VaR (5%)": f"{var_historical(r) * 100:.2f}%",
            "Gaussian VaR (5%)": f"{var_gaussian(r) * 100:.2f}%",
            "Cornish-Fisher VaR (5%)": f"{var_cornish_fisher(r) * 100:.2f}%",
            "CVaR (5%)": f"{cvar_historical(r) * 100:.2f}%"
        }               
    return pd.DataFrame(report)


def alpha_beta_table(strategy: pd.Series, benchmark: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """Computes table with annualised Alpha and Beta to Benchmark."""
    alpha, beta = compute_alpha_beta(strategy, benchmark, rf)
    data = {"Metric": ["Annualised Alpha", "Beta to Benchmark"],
            "Value": [f"{alpha * 100:.2f}%", f"{beta:.2f}"]}
    
    return pd.DataFrame(data).set_index('Metric')