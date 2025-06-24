import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from arch import arch_model

from risk_analysis import (
    get_risk_metrics,
    compute_portfolio_risk,
    compute_benchmark_metrics,
    optimize_portfolio,
    suggest_portfolio_tweaks,
    compute_volatility_table,
    compute_concentration_risk,
    compute_longitudinal_volatility,
)

# --- New: Relative Volatility Ratio Table ---
def compute_relative_volatility_table(vol_table):
    tickers = vol_table.index
    ratio_df = pd.DataFrame(index=tickers, columns=tickers)
    for i in tickers:
        for j in tickers:
            ratio_df.loc[i, j] = vol_table.loc[i, "Yearly"] / vol_table.loc[j, "Yearly"]
    return ratio_df.astype(float).round(2)

# --- New: Tracking Error ---
def compute_tracking_error(portfolio_returns, benchmark_returns):
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[1] != 2:
        return np.nan
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return np.std(diff) * np.sqrt(252)

# --- New: GARCH Volatility Forecast ---
def forecast_volatility_garch(returns):
    results = {}
    for ticker in returns.columns:
        r = returns[ticker].dropna() * 100  # convert to percentage for better stability
        if len(r) > 100:
            try:
                am = arch_model(r, vol='Garch', p=1, q=1)
                res = am.fit(disp="off")
                forecast = res.forecast(horizon=5)
                vol = np.sqrt(forecast.variance.values[-1, :])
                results[ticker] = vol
            except Exception as e:
                results[ticker] = [np.nan] * 5
        else:
            st.warning(f"GARCH not run for {ticker} (less than 100 daily observations)")
            results[ticker] = [np.nan] * 5
    forecast_df = pd.DataFrame(results, index=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]).T
    return forecast_df.round(2)

# Ensure the new functions are accessible to display_risk_and_optimization()
__all__ = [
    "get_risk_metrics",
    "compute_portfolio_risk",
    "compute_benchmark_metrics",
    "optimize_portfolio",
    "suggest_portfolio_tweaks",
    "compute_volatility_table",
    "compute_concentration_risk",
    "compute_longitudinal_volatility",
    "compute_relative_volatility_table",
    "compute_tracking_error",
    "forecast_volatility_garch",
]
