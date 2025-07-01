import io
import pandas as pd
import streamlit as st
import yfinance as yf
from risk_analysis import (
    get_risk_metrics,
    compute_portfolio_risk,
    compute_benchmark_metrics,
    optimize_portfolio,
    suggest_portfolio_tweaks,
    compute_volatility_table,
    compute_longitudinal_volatility,
    compute_concentration_risk
)
import numpy as np
import plotly.express as px

from advanced_volatility import (
    compute_relative_volatility_table,
    compute_tracking_error,
    forecast_volatility_garch,
)

def get_portfolio_metric_summary(portfolio_returns, label, scaling):
    temp_df = pd.DataFrame({label: portfolio_returns})
    metrics_df = get_risk_metrics(temp_df)
    row = metrics_df.loc[label]
    return {
        "Sharpe": round(portfolio_returns.mean() / portfolio_returns.std() * scaling, 4),
        "Sortino": round(
            portfolio_returns.mean() / portfolio_returns[portfolio_returns < 0].std() * scaling
            if portfolio_returns[portfolio_returns < 0].std() > 0 else np.nan, 4
        ),
        "VaR (95%)": round(np.percentile(portfolio_returns, 5), 4),
        "CVaR (95%)": round(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(), 4),
        "Std Dev": round(portfolio_returns.std() * scaling, 4),
        "Median Return": round(np.median(portfolio_returns), 4)
    }

def render_risk_metrics(returns_clean, scaling):
    st.subheader("📉 Risk Metrics per Asset (VaR, CVaR, Sharpe, Sortino)")
    risk_df = get_risk_metrics(returns_clean, scaling=scaling)
    risk_df = risk_df[["VaR_0.05", "CVaR_0.05", "Sharpe", "Sortino", "Median"]]
    risk_df.rename(columns={
        "VaR_0.05": "VaR (5%) ↓",
        "CVaR_0.05": "CVaR (5%) ↓",
        "Sharpe": "Sharpe ↑",
        "Sortino": "Sortino ↑",
        "Median": "Median Return"
    }, inplace=True)
    st.dataframe(risk_df.round(4))

def render_volatility_tables(returns_clean):
    st.subheader("📊 Volatility Table by Frequency (Daily → Yearly)")
    vol_table = compute_volatility_table(returns_clean)
    st.dataframe(vol_table.reset_index().rename(columns={"index": "Ticker"}).round(4))

def render_portfolio_optimization(returns_clean, portfolio_weights, scaling):
    st.subheader("🧠 Portfolio Optimization")
    models = {
        "Mean-Variance (MV)": "MV",
        "Conditional VaR (CVaR)": "CVaR",
        "Drawdown at Risk (DaR)": "DaR",
        "Equal Volatility (EV)": "EV"
    }
    selected_model = st.selectbox("Select Optimization Model", list(models.keys()))
    uploaded_weights = None
    if portfolio_weights is not None and isinstance(portfolio_weights, pd.Series):
        uploaded_weights = portfolio_weights.dropna()
        weights = uploaded_weights.values
        st.info("✅ Using uploaded portfolio weights.")
    else:
        model_key = models[selected_model]
        try:
            w_opt = optimize_portfolio(returns_clean, method=model_key)
            weights = w_opt.values.flatten()
            uploaded_weights = pd.Series(weights, index=returns_clean.columns)
            st.info("⚙️ Using optimized weights.")
        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
            return

    display_weights = pd.DataFrame({"Ticker": uploaded_weights.index, "Weight": uploaded_weights.values})
    st.dataframe(display_weights.round(4))

def render_return_visuals(returns_clean, portfolio_weights, scaling):
    st.subheader("📈 Cumulative Return Comparison")
    opt_weights = optimize_portfolio(returns_clean).values.flatten()
    opt_returns = returns_clean.dot(opt_weights)
    opt_cumret = (1 + opt_returns).cumprod()

    if portfolio_weights is not None:
        uploaded_weights_aligned = portfolio_weights.reindex(returns_clean.columns).fillna(0)
        up_returns = returns_clean.dot(uploaded_weights_aligned)
        up_cumret = (1 + up_returns).cumprod()
        st.line_chart(pd.DataFrame({
            "Uploaded": up_cumret,
            "Optimized": opt_cumret
        }))
    else:
        st.line_chart(pd.DataFrame({
            "Optimized": opt_cumret
        }))

def render_rolling_volatility(returns_clean, portfolio_weights):
    st.subheader("🔁 Rolling Portfolio Volatility")
    if portfolio_weights is not None:
        series = returns_clean.dot(portfolio_weights)
    else:
        series = returns_clean.dot(optimize_portfolio(returns_clean).values.flatten())

    roll_20 = series.rolling(20).std() * np.sqrt(252)
    roll_60 = series.rolling(60).std() * np.sqrt(252)
    st.line_chart(pd.DataFrame({
        "20d Vol": roll_20,
        "60d Vol": roll_60
    }).dropna())

def display_risk_and_optimization(returns_clean, start, end, portfolio_weights=None):
    st.subheader("📐 Sharpe & Sortino Calculation Mode")
    metric_mode = st.radio("Sharpe/Sortino Mode", ["Annualized", "Monthly"], index=0)
    scaling = np.sqrt(252) if metric_mode == "Annualized" else np.sqrt(12)

    tabs = st.tabs([
        "📉 Risk Metrics", "📊 Volatility", "🧠 Optimization",
        "📈 Returns", "🔁 Rolling Vol"
    ])

    with tabs[0]:
        render_risk_metrics(returns_clean, scaling)

    with tabs[1]:
        render_volatility_tables(returns_clean)

    with tabs[2]:
        render_portfolio_optimization(returns_clean, portfolio_weights, scaling)

    with tabs[3]:
        render_return_visuals(returns_clean, portfolio_weights, scaling)

    with tabs[4]:
        render_rolling_volatility(returns_clean, portfolio_weights)

def render_concentration_metrics(weights):
    from risk_analysis import compute_concentration_risk
    hhi, hhi_norm = compute_concentration_risk(weights)
    st.metric("Herfindahl Index (HHI)", f"{hhi:.4f}")
    st.metric("Normalized HHI", f"{hhi_norm:.2%}")

def show_benchmark_metrics(portfolio_name, portfolio_returns, benchmarks, start, end):
    import yfinance as yf
    from risk_analysis import compute_benchmark_metrics

    metrics = []

    for name, symbol in benchmarks.items():
        try:
            bm_raw = yf.download(symbol, start=start, end=end)["Close"]
            bm_weekly = bm_raw.resample("W-FRI").last().pct_change().dropna()

            # Portfolio weekly returns
            port_weekly = portfolio_returns.resample("W-FRI").mean().dropna()
            aligned = pd.concat([port_weekly, bm_weekly], axis=1).dropna()

            beta, corr = compute_benchmark_metrics(portfolio_returns, bm_weekly)

            R_p = (1 + aligned.iloc[:, 0]).prod() ** (52 / len(aligned)) - 1
            R_m = (1 + aligned.iloc[:, 1]).prod() ** (52 / len(aligned)) - 1
            alpha = R_p - beta * R_m

            metrics.append({
                "Benchmark": name,
                "Alpha": alpha,
                "Beta": beta,
                "Correlation": corr,
                "Emoji": "🟢" if alpha > 0 else "🔴"
            })

        except Exception as e:
            st.warning(f"⚠️ Failed to fetch benchmark {name} for {portfolio_name}: {e}")

    return metrics
