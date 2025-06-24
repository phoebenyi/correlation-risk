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
import plotly.express as px
from risk_analysis import compute_concentration_risk
from risk_analysis import compute_benchmark_metrics

from advanced_volatility import (
    compute_relative_volatility_table,
    compute_tracking_error,
    forecast_volatility_garch,
)

def show_benchmark_metrics(portfolio_name, portfolio_returns, benchmarks, start, end, label_emoji=False):
    metrics = []

    for name, symbol in benchmarks.items():
        if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
            portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
        try:
            bm_raw = yf.download(symbol, start=start, end=end)["Close"]
            bm_weekly = bm_raw.resample("W-FRI").last().pct_change().dropna()
            beta, corr = compute_benchmark_metrics(portfolio_returns, bm_weekly)

            # Optional safety check: are we using actual returns?
            if portfolio_returns.max() > 10:
                st.warning(f"⚠️ Portfolio returns may not be return series. Check upstream data.")

            # Ensure we're working with actual returns
            port_weekly = portfolio_returns.resample("W-FRI").mean().dropna()

            aligned = pd.concat([port_weekly, bm_weekly], axis=1).dropna()
            R_p = (1 + aligned.iloc[:, 0]).prod() ** (52 / len(aligned)) - 1
            R_m = (1 + aligned.iloc[:, 1]).prod() ** (52 / len(aligned)) - 1
            alpha = R_p - beta * R_m
            emoji = "🟢" if alpha > 0 else "🔴" if label_emoji else ""

            metrics.append({
                "Benchmark": name,
                "Alpha": alpha,
                "Beta": beta,
                "Correlation": corr,
                "Emoji": emoji
            })

            if label_emoji:
                st.markdown(f"""
                ### {name}
                - 📈 **Alpha**: {alpha:.2%} {emoji}  
                - 📐 **Beta**: {beta:.3f}  
                - 🔗 **Correlation**: {corr:.3f}
                """)

        except Exception as e:
            st.warning(f"⚠️ Failed to fetch benchmark {name} for {portfolio_name}: {e}")

    return metrics

def display_risk_and_optimization(returns_clean, start, end, portfolio_weights=None):
    st.subheader("📉 Risk Metrics per Asset (VaR, CVaR, Sharpe)")
    risk_df = get_risk_metrics(returns_clean)

    with st.expander("ℹ️ Formula Help (VaR, CVaR, Sharpe)"):
        st.markdown("**VaR (95%)**: Maximum expected loss with 95% confidence")
        st.latex(r"\text{VaR}_{0.05} = \text{Percentile}_{5\%}(R)")

        st.markdown("**CVaR (95%)**: Expected loss *beyond* the 95% VaR")
        st.latex(r"\text{CVaR}_{0.05} = E[R \mid R \leq \text{VaR}_{0.05}]")

        st.markdown("**Sharpe Ratio (Annualized)**:")
        st.latex(r"\text{Sharpe} = \frac{E[R] - R_f}{\sigma} \cdot \sqrt{252}")
    risk_df = risk_df[["VaR_0.05", "CVaR_0.05", "Sharpe", "Median"]]
    st.dataframe(risk_df.round(4))

    st.subheader("📊 Volatility Table by Frequency (Daily → Yearly)")

    with st.expander("ℹ️ What This Table Shows & How It's Calculated"):
        st.markdown("""
        **Volatility** measures how much asset returns fluctuate — it reflects **investment risk**.

        This table shows **annualized volatility** (standard deviation of returns) for each asset, calculated from different timeframes:

        | Frequency | Meaning                    | Annualization Factor |
        |-----------|----------------------------|----------------------|
        | Daily     | Based on daily returns     | √252 trading days    |
        | Weekly    | Based on weekly returns    | √52 weeks            |
        | Monthly   | Based on monthly returns   | √12 months           |
        | Quarterly | Based on quarterly returns | √4 quarters          |
        | Yearly    | Based on yearly returns    | 1 (no scaling)       |

        #### 📐 How It's Calculated

        1. **Group returns** by the selected frequency  
        (e.g., group daily returns into months for "Monthly")

        2. **Compute standard deviation** of those grouped returns
        """)
        st.markdown("3. **Annualize** it using this formula:")
        st.latex(r"\text{Annualized Volatility} = \text{Std Dev} \times \sqrt{\text{Periods per Year}}")
    vol_table = compute_volatility_table(returns_clean)
    st.dataframe(vol_table.reset_index().rename(columns={'index': 'Ticker'}).round(4))

    with st.expander("📈 Advanced Volatility Analysis"):
        st.markdown("Includes relative volatility, GARCH forecast, and tracking error vs benchmark.")

        st.subheader("📊 Relative Volatility Ratio (Yearly Vol A / Vol B)")
        rel_vol_table = compute_relative_volatility_table(vol_table)
        st.dataframe(rel_vol_table.style.format("{:.2f}"))

        st.subheader("🔮 5-Day GARCH Volatility Forecast")
        garch_forecast = forecast_volatility_garch(returns_clean)
        st.dataframe(garch_forecast.style.format("{:.2f}"))

        st.subheader("📉 Tracking Error vs Benchmark")
        benchmark_symbol = st.selectbox("Select Benchmark for Tracking Error", ["^GSPC", "^NDX", "^NYA"], index=0)
        try:
            bm = yf.download(benchmark_symbol, start=start, end=end)["Close"].pct_change().dropna()
            weighted_returns = returns_clean.dot(portfolio_weights)
            tracking_error = compute_tracking_error(weighted_returns, bm)
            st.metric(f"Tracking Error (vs {benchmark_symbol})", f"{tracking_error:.2%}")
        except Exception as e:
            st.warning(f"Failed to compute tracking error: {e}")

    st.subheader("📆 Longitudinal Volatility (Rolling Windows)")
    with st.expander("ℹ️ What This Shows"):
        st.markdown("""
    This table shows **recent volatility** for each asset based on different trailing periods:

    - **63d** ≈ 3 months
    - **126d** ≈ 6 months
    - **252d** ≈ 1 year

    Volatility is computed as:
    \\[
    \\text{Rolling Volatility} = \\text{StdDev}(R_{t:t-w}) \\times \\sqrt{252}
    \\]
    """)

    from risk_analysis import compute_longitudinal_volatility
    long_vol = compute_longitudinal_volatility(returns_clean)
    st.dataframe(long_vol.pivot(index="Ticker", columns="Window", values="Volatility").round(4))

    st.subheader("🧠 Portfolio Optimization")
    models = {
        "Mean-Variance (MV)": "MV",
        "Conditional VaR (CVaR)": "CVaR",
        "Drawdown at Risk (DaR)": "DaR",
        "Equal Volatility (EV)": "EV"
    }
    selected_model = st.selectbox("Select Optimization Model", list(models.keys()))

    with st.expander("ℹ️ What is Portfolio Optimization & What Do These Models Do?"):
        st.markdown("""
Portfolio optimization helps determine **how much to allocate to each asset** to achieve specific goals, such as:

- Maximizing return for a given level of risk
- Minimizing exposure to extreme losses
- Equalizing volatility contributions

#### 🧠 Optimization Models:
- **Mean-Variance (MV)**: Balances risk and return using Modern Portfolio Theory. Seeks to maximize Sharpe Ratio or minimize variance.
- **Conditional VaR (CVaR)**: Focuses on worst-case outcomes. Reduces expected losses in the most extreme 5% of cases.
- **Drawdown at Risk (DaR)**: Minimizes large dips from previous highs, protecting against prolonged downturns.
- **Equal Volatility (EV)**: Assigns weights so all assets contribute equally to portfolio risk.

#### ⚙️ How It Works (Simplified):
1. Estimate expected returns and risk metrics
2. Define an objective (e.g., minimize CVaR)
3. Use a solver to compute optimal asset weights
4. Use those weights to evaluate and visualize the new portfolio
""")

    with st.expander("📐 Math Formulations of Optimization Models"):
        st.markdown("### 📊 Mean-Variance (MV) Optimization")
        st.markdown("Objective: Maximize Sharpe Ratio")
        st.latex(r"\max_w \frac{E[R_p] - R_f}{\sqrt{w^T \Sigma w}}")
        st.markdown("Or minimize portfolio variance:")
        st.latex(r"\min_w w^T \Sigma w")
        st.markdown("---")

        st.markdown("### 🔻 Conditional Value-at-Risk (CVaR)")
        st.markdown("Objective: Minimize expected loss in worst-case scenarios")
        st.latex(r"\min_w \text{CVaR}_\alpha(R_p) = E[R_p \mid R_p \leq \text{VaR}_\alpha]")
        st.markdown("---")

        st.markdown("### 📉 Drawdown at Risk (DaR)")
        st.markdown("Objective: Minimize expected drawdown from peak")
        st.latex(r"\min_w \text{DaR}_\alpha(R_p)")
        st.markdown("---")

        st.markdown("### ⚖️ Equal Volatility (EV)")
        st.markdown("Objective: Ensure each asset contributes equally to portfolio volatility")
        st.latex(r"w_i \cdot (\Sigma w)_i = \text{constant for all } i")

    with st.expander("🔒 Weight Constraints & Risk Controls"):
        st.markdown("In portfolio optimization, constraints help manage risk and enforce investor preferences:")

        st.markdown("- **No Shorting**:")
        st.latex(r"w_i \geq 0")
        st.markdown("Prevents negative weights (i.e., borrowing to short sell assets).")

        st.markdown("- **Fully Invested Portfolio**:")
        st.latex(r"\sum_i w_i = 1")
        st.markdown("Ensures 100% of capital is allocated.")

        st.markdown("- **Max Asset Allocation Cap** (optional):")
        st.latex(r"w_i \leq 0.3")
        st.markdown("Limits concentration risk — e.g., no more than 30% in any one stock.")

        st.markdown("These are enforced in the optimization solver (e.g., using `riskfolio-lib`).")
    
    if portfolio_weights is not None and isinstance(portfolio_weights, pd.Series):
        portfolio_weights = portfolio_weights.dropna()
        portfolio_weights = portfolio_weights[portfolio_weights > 0]
        weights = portfolio_weights.values
        display_weights = pd.DataFrame({"Ticker": portfolio_weights.index, "Weight": portfolio_weights.values})
        display_weights.set_index("Ticker", inplace=True)
        st.info("✅ Using weights from uploaded portfolio.")
        w_final = portfolio_weights
    else:
        model_key = models[selected_model]
        w_opt = optimize_portfolio(returns_clean, method=model_key)
        weights = w_opt.values.flatten()
        display_weights = w_opt.T
        st.info("⚙️ Using optimized weights.")
        w_final = w_opt.squeeze()

    st.dataframe(display_weights.round(4))

    st.subheader("📊 Portfolio Weights Breakdown")

    try:
        # If it's an optimized portfolio (DataFrame), transform it
        if isinstance(display_weights, pd.DataFrame) and display_weights.shape[0] == 1:
            display_weights_clean = display_weights.T.reset_index()
            display_weights_clean.columns = ["Ticker", "Weight"]
        elif isinstance(display_weights, pd.DataFrame):
            display_weights_clean = display_weights.reset_index()
            display_weights_clean.columns = ["Ticker", "Weight"]
        else:
            # If it's a Series, convert directly
            display_weights_clean = pd.DataFrame({
                "Ticker": portfolio_weights.index,
                "Weight": portfolio_weights.values
            })

        # Clean and filter
        display_weights_clean = display_weights_clean.dropna()
        display_weights_clean = display_weights_clean[display_weights_clean["Weight"] > 0]
        display_weights_clean = display_weights_clean.sort_values(by="Weight", ascending=False)

        st.markdown("### 🧪 Debug: Editable Portfolio Weights")
        edited_weights = st.data_editor(display_weights_clean, num_rows="dynamic", key="editable_weights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🥧 Pie Chart")
            fig_pie = px.pie(edited_weights, names="Ticker", values="Weight")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("### 📊 Bar Chart")
            fig_bar = px.bar(edited_weights, x="Ticker", y="Weight", text="Weight", labels={"Weight": "Weight %"})
            st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to render portfolio weight breakdown: {e}")

    # Optimized Portfolio
    opt_weights = w_final
    opt_returns = returns_clean.dot(opt_weights)
    opt_cumret = (1 + opt_returns).cumprod()

    opt_std = opt_returns.std() * np.sqrt(252)
    opt_sharpe = (opt_returns.mean() * 252) / (opt_returns.std() * np.sqrt(252))
    opt_var = np.percentile(opt_returns, 5)
    opt_cvar = opt_returns[opt_returns <= opt_var].mean()
    opt_median = np.median(opt_returns)

    # Uploaded Portfolio (if any)
    if portfolio_weights is not None:
        up_weights = portfolio_weights.values
        up_returns = returns_clean.dot(up_weights)
        up_cumret = (1 + up_returns).cumprod()

        up_std = up_returns.std() * np.sqrt(252)
        up_sharpe = (up_returns.mean() * 252) / (up_returns.std() * np.sqrt(252))
        up_var = np.percentile(up_returns, 5)
        up_cvar = up_returns[up_returns <= up_var].mean()

        # Comparison Table
        st.subheader("📊 Uploaded vs Optimized Portfolio Performance")
        compare_df = pd.DataFrame({
            "Metric": ["Annualized Std Dev", "Sharpe Ratio", "VaR (95%)", "CVaR (95%)", "Median Return"],
            "Uploaded Portfolio": [up_std, up_sharpe, up_var, up_cvar, np.median(up_returns)],
            "Optimized Portfolio": [opt_std, opt_sharpe, opt_var, opt_cvar, opt_median]
        })
        st.dataframe(compare_df.set_index("Metric").applymap(lambda x: f"{x:.2%}" if "Dev" in str(x) or "VaR" in str(x) else round(x, 4)))

        # Side-by-side Cumulative Returns
        st.subheader("📈 Cumulative Return Comparison")
        cum_df = pd.DataFrame({
            "Uploaded Portfolio": up_cumret,
            "Optimized Portfolio": opt_cumret
        }).dropna()
        st.line_chart(cum_df)
    else:
        st.info("📂 No uploaded portfolio weights provided — only showing optimized results.")

    # Optimized Portfolio Risk Summary (FIXED VARIABLES)
    st.subheader("📊 Optimized Portfolio Risk Summary")
    hhi, hhi_norm = compute_concentration_risk(weights)

    st.metric("Herfindahl Index (HHI)", f"{hhi:.4f}")
    st.metric("Normalized HHI", f"{hhi_norm:.2%}")

    with st.expander("ℹ️ What is Concentration Risk?"):
        st.markdown("""
    - **HHI (Herfindahl-Hirschman Index)** measures how concentrated your portfolio is.
    - A value close to **1** indicates high concentration (e.g., most capital in one asset).
    - A value closer to **1/N** (where N is number of assets) suggests even diversification.

    \\[
    \\text{HHI} = \\sum w_i^2 \\quad\\text{(higher = more concentrated)}
    \\]

    \\[
    \\text{Normalized HHI} = \\frac{HHI - \\frac{1}{N}}{1 - \\frac{1}{N}}
    \\quad\\in [0, 1]
    \\]
    """)
    with st.expander("ℹ️ What These Metrics Mean & How They're Calculated"):
        st.markdown("**📊 Optimized Portfolio Risk Summary** provides key metrics to understand the risk-return profile of the optimized portfolio:")

        st.markdown("**• Annualized Std Dev (Volatility):** Measures how much the portfolio's returns fluctuate annually.")
        st.latex(r"\text{Volatility}_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}")

        st.markdown("**• VaR (Value-at-Risk) at 95% Confidence:** Maximum expected **loss** on a single day under normal market conditions.")
        st.latex(r"\text{VaR}_{0.05} = \text{Percentile}_{5\%}(R)")

        st.markdown("**• CVaR (Conditional VaR) at 95% Confidence:** Expected **loss** given that VaR is breached — captures tail risk.")
        st.latex(r"\text{CVaR}_{0.05} = \mathbb{E}[R \mid R \leq \text{VaR}_{0.05}]")
    st.metric("Annualized Std Dev", f"{opt_std:.2%}")
    st.metric("VaR (95%)", f"{opt_var:.2%}")
    st.metric("CVaR (95%)", f"{opt_cvar:.2%}")

    portfolio_daily_returns = returns_clean.dot(weights)
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    st.subheader("📈 Cumulative Return of Optimized Portfolio")
    st.line_chart(cumulative_returns)

    st.subheader("🔁 Rolling Portfolio Volatility (20d & 60d)")
    with st.expander("ℹ️ Why Rolling Volatility?"):
        st.markdown("""
    **Rolling volatility** shows how your portfolio's risk changes over time.

    - 20-day volatility → short-term view
    - 60-day volatility → smoother long-term view
    """)
        st.markdown("Formula:")
        st.latex(r"\text{Rolling Volatility} = \text{StdDev}(R_t, \text{window}) \times \sqrt{252}")
    roll_vol_20d = portfolio_daily_returns.rolling(20).std() * np.sqrt(252)
    roll_vol_60d = portfolio_daily_returns.rolling(60).std() * np.sqrt(252)
    roll_vol_df = pd.DataFrame({
        "20d Rolling Volatility": roll_vol_20d,
        "60d Rolling Volatility": roll_vol_60d
    }).dropna()
    st.line_chart(roll_vol_df)

    benchmarks = {
        "Nasdaq-100": "^NDX",
        "S&P 500": "^GSPC",
        "NYSE": "^NYA"
    }

    st.subheader("📉 Benchmark Comparison (Alpha, Beta, Correlation)")

    with st.expander("ℹ️ What Do These Metrics Mean?"):
        st.markdown("### 📐 Alpha, Beta, Correlation")

        st.markdown("**• Alpha (α)**: Excess return above what the market explains (positive α = outperformance):")
        st.latex(r"\alpha = R_p - \beta R_m")

        st.markdown("**• Beta (β)**: Sensitivity to market movements:")
        st.latex(r"\beta = \frac{\text{Cov}(R_p, R_m)}{\text{Var}(R_m)}")

        st.markdown("**• Correlation**: Measures directional co-movement with the benchmark, in range [-1, +1].")

    st.subheader("📊 Alpha/Beta/Correlation Table (Optimized & Uploaded Portfolios)")
    st.caption("📘 Alpha and Beta are calculated using *weekly (Friday-to-Friday)* returns over the past 3 months.")
    combined_data = []

    opt_metrics = show_benchmark_metrics("Optimized Portfolio", portfolio_daily_returns, benchmarks, start, end, label_emoji=True)

    if portfolio_weights is not None:
        uploaded_daily_returns = returns_clean.dot(portfolio_weights)
        up_metrics = show_benchmark_metrics("Uploaded Portfolio", uploaded_daily_returns, benchmarks, start, end, label_emoji=False)
    else:
        up_metrics = []

    for row in opt_metrics:
        combined_data.append({
            "Benchmark": row["Benchmark"],
            "Alpha (Optimized)": row["Alpha"],
            "Beta (Optimized)": row["Beta"],
            "Correlation (Optimized)": row["Correlation"]
        })

    if portfolio_weights is not None:
        for idx, row in enumerate(up_metrics):
            combined_data[idx]["Alpha (Uploaded)"] = row["Alpha"]
            combined_data[idx]["Beta (Uploaded)"] = row["Beta"]
            combined_data[idx]["Correlation (Uploaded)"] = row["Correlation"]

    combined_df = pd.DataFrame(combined_data)
    st.dataframe(combined_df.round(4))

    benchmark_df = pd.DataFrame(opt_metrics)
    benchmark_df_display = benchmark_df.copy()
    benchmark_df_display["Alpha"] = benchmark_df_display["Alpha"].apply(lambda x: f"{x:.2%}")
    benchmark_df_display["Beta"] = benchmark_df_display["Beta"].apply(lambda x: f"{x:.3f}")
    benchmark_df_display["Correlation"] = benchmark_df_display["Correlation"].apply(lambda x: f"{x:.3f}")

    if combined_data:
        st.subheader("📄 Exportable Alpha/Beta/Correlation Table (Optimized Portfolio)")

        st.download_button(
            "⬇️ Download Benchmark Alpha/Beta/Correlation CSV",
            pd.DataFrame(combined_data).to_csv(index=False),
            file_name="alpha_beta_correlation_comparison.csv",
            mime="text/csv"
        )

        top_alpha_row = max(opt_metrics, key=lambda x: x["Alpha"])
        st.success(f"🏆 **Top Benchmark Outperformance**: {top_alpha_row['Benchmark']} with Alpha = {top_alpha_row['Alpha']:.2%}")
                
    st.subheader("💬 Risk Contribution Suggestions")
    with st.expander("ℹ️ How These Suggestions Are Generated"):
        st.markdown("""
    These tips are generated by evaluating **how much each asset contributes to portfolio risk**.

    Steps:
    1. Calculate marginal risk (volatility or CVaR) of each asset.
    2. Compare each asset's risk vs weight.
    3. Highlight imbalances (e.g., heavy weights with high risk).

    The model suggests:
    - Reducing oversized or high-risk positions
    - Balancing risk exposure more evenly across assets
    - Avoiding overconcentration

    These are heuristic-based adjustments, not hard rules.
        """)
    suggestions = suggest_portfolio_tweaks(w_final, returns_clean)
    for s in suggestions:
        st.markdown(f"- {s}")