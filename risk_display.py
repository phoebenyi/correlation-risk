import pandas as pd
import streamlit as st
import yfinance as yf
from risk_analysis import (
    get_risk_metrics,
    compute_portfolio_risk,
    compute_benchmark_metrics,
    optimize_portfolio,
    suggest_portfolio_tweaks,
    compute_volatility_table
)
import numpy as np


def display_risk_and_optimization(returns_clean, start, end, portfolio_weights=None):
    st.subheader("📉 Risk Metrics per Asset (VaR, CVaR, Sharpe)")
    risk_df = get_risk_metrics(returns_clean)

    with st.expander("ℹ️ Formula Help (VaR, CVaR, Sharpe)"):
        st.markdown("""
- **VaR (95%)**: Maximum expected loss with 95% confidence  
  \( \text{VaR}_{0.05} = \text{Percentile}_{5\%}(R) \)

- **CVaR (95%)**: Expected loss *beyond* the 95% VaR  
  \( \text{CVaR}_{0.05} = E[R | R \leq \text{VaR}_{0.05}] \)

- **Sharpe Ratio** (Annualized):  
  \( \text{Sharpe} = \frac{E[R] - R_f}{\sigma} \cdot \sqrt{252} \)
        """)
    st.dataframe(risk_df.round(4))

    st.subheader("📊 Volatility Table by Frequency (Daily → Yearly)")
    vol_table = compute_volatility_table(returns_clean)
    st.dataframe(vol_table.reset_index().rename(columns={'index': 'Ticker'}).round(4))

    # Portfolio Optimization
    st.subheader("🧠 Portfolio Optimization")
    models = {
        "Mean-Variance (MV)": "MV",
        "Conditional VaR (CVaR)": "CVaR",
        "Drawdown at Risk (DaR)": "DaR",
        "Equal Volatility (EV)": "EV"
    }
    selected_model = st.selectbox("Select Optimization Model", list(models.keys()))
    
    if portfolio_weights is not None:
        weights = portfolio_weights.values
        display_weights = pd.DataFrame({
            "Weight": portfolio_weights
        }).T
        st.info("✅ Using weights from uploaded portfolio.")
        w_final = portfolio_weights
    else:
        w_opt = optimize_portfolio(returns_clean)
        weights = w_opt.values.flatten()
        display_weights = w_opt.T
        st.info("⚙️ Using optimized weights.")
        w_final = w_opt.squeeze()

    st.dataframe(display_weights.round(4))

    port_std, port_var, port_cvar = compute_portfolio_risk(returns_clean, weights)

    st.subheader("📊 Optimized Portfolio Risk Summary")
    st.metric("Annualized Std Dev", f"{port_std * np.sqrt(252):.2%}")
    st.metric("VaR (95%)", f"{port_var:.2%}")
    st.metric("CVaR (95%)", f"{port_cvar:.2%}")

    weighted_returns = returns_clean.dot(weights)
    cumulative_returns = (1 + weighted_returns).cumprod()
    st.subheader("📈 Cumulative Return of Optimized Portfolio")
    st.line_chart(cumulative_returns)

    st.subheader("🔁 Rolling Portfolio Volatility (20d & 60d)")
    roll_vol_20d = weighted_returns.rolling(20).std() * np.sqrt(252)
    roll_vol_60d = weighted_returns.rolling(60).std() * np.sqrt(252)
    roll_vol_df = pd.DataFrame({
        "20d Rolling Volatility": roll_vol_20d,
        "60d Rolling Volatility": roll_vol_60d
    }).dropna()
    st.line_chart(roll_vol_df)

    st.subheader("📉 Benchmark Comparison (Beta, Correlation)")
    benchmarks = {
        "Nasdaq-100": "^NDX",
        "S&P 500": "^GSPC",
        "SNCP": "^STI"
    }
    for name, symbol in benchmarks.items():
        try:
            bm = yf.download(symbol, start=start, end=end)["Close"].pct_change().dropna()
            beta, corr = compute_benchmark_metrics(weighted_returns, bm)
            st.markdown(f"**{name}**  \n📐 Beta: `{beta:.3f}`  \n🔗 Correlation: `{corr:.3f}`")
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch benchmark {name}: {e}")

    st.subheader("💬 Risk Contribution Suggestions")
    suggestions = suggest_portfolio_tweaks(w_final, returns_clean)
    for s in suggestions:
        st.markdown(f"- {s}")