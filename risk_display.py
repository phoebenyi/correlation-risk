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
    with st.expander("❓ What Do These Risk Metrics Mean?"):
        st.markdown(r"""
        ### 🧠 Risk Metrics Explained

        #### 💥 Value at Risk (VaR)
        VaR tells you the **worst expected loss** over a day, with 95% confidence.

        **Formula:**
        $$
        \text{VaR}_{\alpha} = \text{Percentile}_{\alpha}(\text{Returns})
        $$

        👉 **Lower is better** — means fewer extreme losses.

        ---

        #### 🔻 Conditional VaR (CVaR)
        Average loss **if the worst 5% happens**.

        **Formula:**
        $$
        \text{CVaR}_{\alpha} = \mathbb{E}[\text{Returns} \mid \text{Returns} \leq \text{VaR}_{\alpha}]
        $$

        👉 **Lower is better** — reflects depth of worst-case scenarios.

        ---

        #### 📈 Sharpe Ratio
        Return per unit of **total** risk.

        **Formula:**
        $$
        \text{Sharpe} = \frac{\mathbb{E}[R - R_f]}{\sigma} \times \sqrt{252}
        $$

        👉 **Higher is better**. Sharpe > 1 = good, > 2 = excellent.

        ---

        #### 🎯 Sortino Ratio
        Return per unit of **downside** risk only.

        **Formula:**
        $$
        \text{Sortino} = \frac{\mathbb{E}[R - R_f]}{\sigma_{\text{down}}} \times \sqrt{252}
        $$

        👉 **Higher is better** — penalizes only bad volatility.

        ---

        #### 📊 Median Return
        The "typical" return (less affected by outliers).

        👉 **Higher is better**.
        """)
        st.markdown("""
        ---

        ### 🛠️ How to Use These Metrics

        | Metric | Use it to… | Ideal |
        |--------|------------|-------|
        | VaR / CVaR | Spot worst-case losses | ↓ Lower |
        | Sharpe | Evaluate overall return vs. risk | ↑ Higher |
        | Sortino | Focus on downside protection | ↑ Higher |
        | Median Return | Estimate typical performance | ↑ Higher |
        """)
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
    with st.expander("❓ What is Volatility and How to Use It?"):
        st.markdown(r"""
    ### 📉 Volatility Overview

    Volatility = how much prices move around — the “bumpiness” of an asset.

    #### 🧮 Formula:
    $$
    \text{Volatility}_{\text{period}} = \text{Std Dev of Returns} \times \sqrt{\text{periods}}
    $$

    - Daily volatility uses √1  
    - Weekly = √5  
    - Monthly = √21  
    - Yearly = √252

    #### 📌 Interpretation:
    - **Higher = riskier**, but not always bad
    - Good for spotting **stable vs. explosive** stocks
    - Use with Sharpe/Sortino to balance risk and reward
    """)
    with st.expander("❓ What Is the Relative Volatility Matrix?"):
        st.markdown(r"""
    ### 📊 Relative Volatility Matrix

    This compares **how risky one stock is relative to another**.

    ---

    ### 🧮 Formula:
    $$
    \text{Relative Vol}_{i,j} = \frac{ \text{Volatility of } i }{ \text{Volatility of } j }
    $$

    E.g., if A/B = 2.0 → A is twice as volatile as B (based on yearly volatility)

    🧠 **Use it to**:
    - Spot overly risky assets
    - Normalize risk profiles across a portfolio
    """)
    with st.expander("❓ What Is Rolling Volatility?"):
        st.markdown(r"""
    ### 🔁 Rolling Volatility

    Shows how volatility changes over time using a rolling window (e.g. 20d, 60d).

    ---

    ### 🧮 Formula:
    $$
    \text{Rolling Vol}_{t} = \text{std}(R_{t-N:t}) \times \sqrt{252}
    $$

    🧠 **Use it to**:
    - Detect rising/falling risk
    - Time rebalancing or hedging decisions
    """)
    with st.expander("❓ What Is Volatility Evolution?"):
        st.markdown(r"""
    ### 🕰️ Volatility Evolution

    Shows how long-term volatility has changed across different timeframes:

    - 3 months (63d)
    - 6 months (126d)
    - 1 year (252d)

    🧠 **Use it to**:
    - See whether risk is increasing
    - Detect regime shifts in asset behavior
    """)
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

def show_benchmark_metrics(portfolio_returns, benchmarks, start, end):
    import yfinance as yf
    from risk_analysis import compute_benchmark_metrics
    import streamlit as st

    metrics = []

    for name, symbol in benchmarks.items():
        try:
            bm_raw = yf.download(symbol, start=start, end=end)["Close"]
            bm_weekly = bm_raw.resample("W-FRI").last().pct_change().dropna()

            port_weekly = portfolio_returns.resample("W-FRI").mean().dropna()

            # Align data
            aligned = pd.concat([port_weekly, bm_weekly], axis=1).dropna()

            if st.session_state.get("debug_mode", False):
                st.write(f"📊 {name} aligned rows: {aligned.shape[0]}")
                st.write(f"Portfolio weekly range: {port_weekly.index.min()} to {port_weekly.index.max()}")
                st.write(f"Benchmark weekly range: {bm_weekly.index.min()} to {bm_weekly.index.max()}")

            if aligned.empty:
                st.warning(f"⚠️ No overlap with benchmark: {name}")
                continue

            beta, corr = compute_benchmark_metrics(port_weekly, bm_weekly)

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
            st.warning(f"❌ Failed to process {name}: {e}")

    return metrics

def render_relative_volatility_table(returns_clean):
    from advanced_volatility import compute_volatility_table, compute_relative_volatility_table
    st.subheader("🔁 Relative Volatility Matrix (A/B Ratio of Yearly Volatility)")
    vol_table = compute_volatility_table(returns_clean)
    rel_vol_df = compute_relative_volatility_table(vol_table)
    st.dataframe(rel_vol_df)

def render_return_histogram(returns_clean, portfolio_weights):
    st.subheader("📊 Portfolio Return Histogram")
    import plotly.express as px

    if portfolio_weights is not None:
        port_returns = returns_clean.dot(portfolio_weights)
        label = "Uploaded Portfolio"
    else:
        opt_weights = optimize_portfolio(returns_clean).values.flatten()
        port_returns = returns_clean.dot(opt_weights)
        label = "Optimized Portfolio"

    fig = px.histogram(port_returns, nbins=50, title=f"{label} Return Distribution", marginal="rug")
    st.plotly_chart(fig, use_container_width=True)

def render_tracking_error(returns_clean, portfolio_weights, benchmark_symbol="^GSPC"):
    st.subheader("📉 Tracking Error vs Benchmark")

    import yfinance as yf
    from advanced_volatility import compute_tracking_error

    try:
        benchmark = yf.download(benchmark_symbol, start=returns_clean.index.min(), end=returns_clean.index.max())["Close"]
        benchmark_returns = benchmark.pct_change().dropna()

        if portfolio_weights is not None:
            port_returns = returns_clean.dot(portfolio_weights)
        else:
            opt_weights = optimize_portfolio(returns_clean).values.flatten()
            port_returns = returns_clean.dot(opt_weights)

        te = compute_tracking_error(port_returns, benchmark_returns)
        st.metric("Tracking Error", f"{te:.4%}")
    except Exception as e:
        st.warning(f"Failed to compute tracking error: {e}")

def render_garch_volatility_forecast(returns_clean):
    st.subheader("🔮 GARCH 5-Day Volatility Forecast")
    with st.expander("❓ What is GARCH Forecasting?"):
        st.markdown(r"""
    ### 🔮 GARCH – Forecasting Future Volatility

    **GARCH = Generalized Autoregressive Conditional Heteroskedasticity**  
    It’s a fancy way to model and **predict volatility** based on past patterns.

    ---

    ### 💡 What it Does:
    - Models “volatility clustering”  
    - Predicts **next 5 days of volatility**
    - Widely used in:  
    - Risk models (e.g. VaR)  
    - Options pricing  
    - Forecasting market stress

    ---

    ### 🧮 Formula (simplified GARCH(1,1)):
    $$
    \sigma_t^2 = \alpha_0 + \alpha_1 \cdot \epsilon_{t-1}^2 + \beta_1 \cdot \sigma_{t-1}^2
    $$

    - \( \epsilon_{t-1}^2 \): Squared return shock yesterday  
    - \( \sigma_{t-1}^2 \): Previous day’s forecasted variance

    ---

    ### 🛠️ How to Use It:
    - Detect rising/falling market risk  
    - Stress-test short-term portfolios  
    - Pair with VaR / CVaR for scenario analysis
    """)
    from advanced_volatility import forecast_volatility_garch
    forecast_df = forecast_volatility_garch(returns_clean)
    st.dataframe(forecast_df)

def render_risk_summary_table(returns_clean, portfolio_weights, scaling):
    st.subheader("🧠 Risk Summary Table")

    if portfolio_weights is not None:
        port_returns = returns_clean.dot(portfolio_weights)
        label = "Uploaded"
    else:
        opt_weights = optimize_portfolio(returns_clean).values.flatten()
        port_returns = returns_clean.dot(opt_weights)
        label = "Optimized"

    summary = get_portfolio_metric_summary(port_returns, label, scaling)
    st.table(pd.DataFrame([summary]))

def render_longitudinal_volatility_table(returns_clean):
    from risk_analysis import compute_longitudinal_volatility
    st.subheader("🕰️ Volatility Evolution (3m, 6m, 1y)")
    df = compute_longitudinal_volatility(returns_clean)
    st.dataframe(df)
