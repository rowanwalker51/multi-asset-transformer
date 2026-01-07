# Multi-Asset, Multi-Horizon Transformer for Systematic Equity Trading

## Overview

This repository contains a full end-to-end systematic trading research pipeline built around a multi-horizon time-series transformer. The model produces probabilistic forecasts over several forward horizons, which are averaged and converted into cross-sectional rankings. Portfolio signals are formed by taking long positions in the top x% of ranked assets and short positions in the bottom x%, with portfolio weights determined via volatility targeting and risk parity.

The project prioritises time-series correctness, realistic execution assumptions and portfolio-level evaluation over isolated predictive metrics.

---

## Research Motivation

Point forecasts and single-horizon classifiers are unstable in financial time series and prone to calibration drift. This project explores whether averaging probabilistic forecasts across multiple horizons and acting only on relative ranking information leads to more robust trading performance once realistic portfolio construction and transaction costs are applied.

---

## Pipeline Summary

### Data Ingestion and Feature Engineering

- Explicit separation of raw, processed and inference datasets
- Centralised YAML configuration for data paths, model parameters and experiments
- Feature sets include:
  - Equity price and volume features
  - Market benchmark inputs
  - FX and commodity macro proxies
  - Rolling technical and statistical indicators
- Market regimes inferred using a Gaussian Hidden Markov Model
- Parquet used throughout for performance and schema stability

---

### Model Architecture

#### Multi-Horizon Time-Series Transformer

- Implemented in PyTorch
- Single encoder architecture producing probabilistic forecasts over multiple forward horizons
- Core components:
  - Linear projection of numerical features
  - Stock embeddings for cross-sectional differentiation
  - Regime embeddings derived from HMM posterior probabilities
  - Rotary positional embeddings
  - CLS token for sequence-level aggregation
- Designed to handle multiple assets within a unified model

---

### Training and Inference

- Deterministic training with fixed random seeds
- Separate training and inference loops
- Walk-forward retraining supported
- All intermediate artefacts and inference outputs saved explicitly for auditability

---

### Signal Construction

- The model outputs class probabilities for each asset across several forward horizons
- Horizon-level probabilities are averaged to form a single composite probability per asset
- Assets are ranked cross-sectionally by the averaged probability
- Signals are constructed as:
  - Long: top x% of ranked assets
  - Short: bottom x% of ranked assets
- No absolute probability thresholds are used, reducing sensitivity to calibration drift

---

### Portfolio Construction and Risk Management

- Positions derived from ranked signals
- Portfolio construction includes:
  - Volatility targeting
  - Risk parity allocation
  - Stop-loss and take-profit rules
  - Drawdown constraints
- Execution assumptions explicitly model:
  - Transaction costs
  - Slippage
  - Turnover effects

---

### Benchmarking

- Challenger model implemented using XGBoost
- Time-series aware cross-validation
- Standardised preprocessing pipeline
- Probabilistic outputs for direct comparison
- Identical signal construction and portfolio logic applied to both models
- Transformer outperforms the challenger out-of-sample after costs

---

### Backtesting and Validation

- Custom backtesting engine
- Walk-forward validation supported
- Performance metrics include:
  - Returns and volatility
  - Sharpe ratio
  - Maximum drawdown

---

### Interactive Analysis

- Dash-based GUI for:
  - Backtest parameter tuning
  - Signal threshold selection (x%)
  - Risk and allocation constraints
- Designed for rapid iteration without code changes

---

## Project Structure
multi-asset-transformer/
│
├── configs/           # YAML configs
├── data/
│   ├── raw/
│   ├── processed/
│   └── inference/
├── src/
│   ├── common/        # General configs
|   ├── data/          # Ingestion and feature engineering
│   ├── gui/           # GUI dashboard
│   ├── inference/     # Model inference logic
│   ├── models/        # Transformer architecture
│   ├── training/      # Model training
│   └── utils/         # Back test logic, risk suite and seeding
├── results/           # Trained models
├── notebooks/         # Research notebooks
└── README.md


---

## Design Principles

- Time-series correctness over convenience
- Explicit prevention of data leakage
- Relative ranking preferred to absolute probability thresholds
- Portfolio-level performance as the primary objective
- Clear separation between modelling, signal construction and portfolio logic

---

## Results Summary

- Averaging probabilities across horizons produces more stable rankings
- Cross-sectional selection improves robustness across regimes
- Volatility targeting and risk parity materially reduce drawdowns
- Transformer outperforms a tuned XGBoost benchmark after transaction costs