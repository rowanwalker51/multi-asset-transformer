# Quantitative Time Series Trading Project

This repository contains a deep learning-based trading framework for equity and FX markets. The project focuses on feature engineering, sequence modeling with Transformers, regime awareness using Hidden Markov Models, and backtesting with risk metrics.

## Project Structure

- `data/raw/` – Raw market and macroeconomic data (FTSE 100, FX, commodities, risk-free rates).  
- `data/processed/` – Processed features, predictions, and signals.  
- `results/model/` – Trained PyTorch Transformer models.  
- `utils/` – Helper functions and parameters.  
- `src/` – Core scripts for feature generation, model training, inference, and backtesting.  

## Key Components

1. **Data Loading & Feature Engineering**  
   - Pulls historical market data and computes technical indicators, log returns, volatility, RSI, beta, and temporal features.  

2. **Regime Detection**  
   - Hidden Markov Model-based regime probabilities are computed from macro and market data.  

3. **Time Series Transformer**  
   - Multi-stock sequence modeling using Transformer architecture with rotary positional embeddings (RoPE), stock embeddings, and regime embeddings.  

4. **Model Training & Prediction**  
   - Handles sequence batching, normalization, and outputs probabilistic predictions of asset movements.  

5. **Walk-Forward Validation & Backtesting**  
   - Out-of-sample validation with configurable train/test windows.  
   - Volatility-targeted portfolio allocation, slippage, commission, take-profit/stop-loss, and max holding rules.  
   - Calculates performance metrics including Sharpe ratio, alpha, beta, drawdown, and rolling risk measures.  

6. **Risk Analysis**  
   - Comprehensive risk reporting: Sharpe, Sortino, max drawdown, VAR/CVaR, win/loss rates, profit factor, tail ratios, alpha/beta.  