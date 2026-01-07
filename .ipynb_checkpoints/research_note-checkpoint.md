# Multi-Horizon Probabilistic Ranking for Systematic Equity Trading

**Author:** Rowan Walker  

---

## Objective

The objective of this research is to evaluate whether averaging probabilistic forecasts across multiple forward horizons and acting solely on cross-sectional rankings leads to more robust equity trading performance than single-horizon classification approaches, once realistic portfolio construction and transaction costs are applied.

Rather than relying on absolute probability thresholds, the strategy uses relative ranking information, which is less sensitive to calibration drift and regime changes commonly observed in financial time series.

---

## Data and Feature Set

The model is trained on a cross-section of FTSE 100 equities. Input features include:

- Equity-level price, return and volume features
- Rolling technical and statistical indicators
- Market benchmark features
- FX and commodity macro proxies
- Latent market regime features inferred using a Gaussian Hidden Markov Model

Data is partitioned using strict time-based splits to prevent look-ahead bias. All datasets are stored in Parquet format for performance and schema stability.

---

## Model Architecture

A multi-horizon time-series transformer is implemented in PyTorch. The model produces probabilistic forecasts over multiple forward horizons within a single architecture.

Key components include:

- Linear projection of numerical features
- Learned stock embeddings for cross-sectional differentiation
- Regime embeddings derived from HMM posterior probabilities
- Rotary positional embeddings
- A CLS token used for sequence-level aggregation

The architecture is designed to handle multiple assets within a unified model while preserving time-series ordering.

---

## Signal Construction

For each asset, the model outputs class probabilities across several forward horizons. These probabilities are averaged to form a single composite probability per asset.

Assets are ranked cross-sectionally by the averaged probability. Portfolio signals are defined as:

- **Long:** top *x%* of ranked assets  
- **Short:** bottom *x%* of ranked assets  

No absolute probability thresholds are applied. This ranking-based approach reduces sensitivity to miscalibration and improves stability across market regimes.

---

## Portfolio Construction and Risk Management

Signals are converted into positions using a volatility-targeted, risk parity framework. The portfolio construction process includes:

- Volatility targeting at the portfolio level
- Risk parity allocation across active positions
- Stop-loss and take-profit rules
- Explicit drawdown constraints

Execution assumptions incorporate transaction costs, slippage and turnover effects to ensure realistic performance estimates.

---

## Benchmark Model

A challenger model is implemented using XGBoost with time-series aware cross-validation. The challenger model uses a standardised preprocessing pipeline and produces probabilistic outputs.

Both the transformer and the challenger model are evaluated using identical signal construction, ranking logic and portfolio construction rules to ensure a fair comparison.

---

## Results

The multi-horizon transformer consistently outperforms the XGBoost challenger out-of-sample after transaction costs. Averaging probabilities across horizons improves ranking stability and reduces drawdowns relative to single-horizon approaches.

Performance improvements are primarily observed at the portfolio level rather than in isolated classification metrics, highlighting the importance of signal construction and risk management.

---

## Conclusion

In noisy financial time series, relative ranking of averaged multi-horizon probabilistic forecasts is more robust than acting on point forecasts or absolute probability thresholds.

The results suggest that integrating model outputs into a disciplined ranking-based signal framework with realistic portfolio construction is critical for achieving stable risk-adjusted returns.