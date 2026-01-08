# Multi-Horizon Probabilistic Ranking for Systematic Equity Trading

**Author:** Rowan Walker  

---

## Objective

This research evaluates whether **averaging probabilistic forecasts across multiple forward horizons** and acting solely on **cross-sectional rankings** leads to more robust equity trading performance than single-horizon classification approaches.  

By relying on **relative rankings** rather than absolute probability thresholds, the strategy is less sensitive to calibration drift and regime shifts commonly observed in financial time series.

---

## Data and Feature Set

The model is trained on a cross-section of FTSE 100 equities. Input features include:

- Equity-level price, return, and volume features  
- Rolling technical and statistical indicators  
- Market benchmark features  
- FX and commodity macro proxies  
- Latent market regime features inferred using a Gaussian Hidden Markov Model  

Data is partitioned using **strict time-based splits** to prevent look-ahead bias. All datasets are stored in **Parquet format** for performance and schema stability.

---

## Model Architecture

A **multi-horizon time-series transformer** is implemented in PyTorch. It produces probabilistic forecasts over multiple forward horizons within a single architecture.

Key components:

- Linear projection of numerical features  
- Learned stock embeddings for cross-sectional differentiation  
- Regime embeddings derived from HMM posterior probabilities  
- Rotary positional embeddings  
- CLS token for sequence-level aggregation  

The architecture handles multiple assets simultaneously while preserving strict time-series ordering.

---

## Signal Construction

For each asset, the model outputs class probabilities across multiple horizons. These are averaged to produce a **composite probability per asset**.  

Assets are ranked cross-sectionally, and portfolio signals are generated as:

- **Long:** top *x%* of ranked assets  
- **Short:** bottom *x%* of ranked assets  

No absolute probability thresholds are applied. This ranking-based approach improves robustness and reduces sensitivity to miscalibration.

---

## Portfolio Construction and Risk Management

Signals are converted into positions using a **volatility-targeted, risk parity framework**. The portfolio construction pipeline includes:

- Volatility targeting at the portfolio level  
- Risk parity allocation across active positions  
- Stop-loss and take-profit rules  
- Explicit drawdown constraints  

Execution assumptions model **transaction costs, slippage, and turnover effects**, ensuring realistic performance estimates.

---

## Backtesting and Validation

A **custom backtesting engine** supports:

- Walk-forward validation  
- Equity curve generation  
- Sharpe ratio computation  
- Parameter optimisation  

---

## Benchmark Model

A challenger model using **XGBoost** with time-series aware cross-validation is included.  

- Standardised preprocessing pipeline  
- Probabilistic outputs for direct comparison  
- Identical ranking and portfolio construction rules applied  

This allows a **fair and consistent comparison** with the transformer.

---

## Results

- Multi-horizon transformer **outperforms XGBoost** out-of-sample after transaction costs  
- Averaging probabilities across horizons **stabilises rankings** and **reduces drawdowns**  
- Performance improvements are primarily observed at the **portfolio level**  
- Relative ranking of multi-horizon probabilistic forecasts is more robust than point forecasts or absolute thresholds  

---

## Conclusion

In noisy financial time series, **cross-sectional ranking of averaged multi-horizon forecasts** combined with disciplined portfolio construction yields **more stable, risk-adjusted returns** than traditional single-horizon or absolute-threshold approaches.  

Backtesting and automated tests ensure that results are **reproducible and robust**, supporting confident deployment of the strategy in realistic scenarios.