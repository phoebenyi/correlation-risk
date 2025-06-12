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

    st.subheader("📊 Portfolio Weights Breakdown")

    display_weights_clean = display_weights.T.reset_index()
    display_weights_clean.columns = ["Ticker", "Weight"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🥧 Pie Chart")
        fig_pie = px.pie(display_weights_clean, names="Ticker", values="Weight")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("### 📊 Bar Chart")
        fig_bar = px.bar(display_weights_clean, x="Ticker", y="Weight", text="Weight", labels={"Weight": "Weight %"})
        st.plotly_chart(fig_bar, use_container_width=True)

    # Optimized Portfolio
    opt_weights = w_final
    opt_returns = returns_clean.dot(opt_weights)
    opt_cumret = (1 + opt_returns).cumprod()

    opt_std = opt_returns.std() * np.sqrt(252)
    opt_sharpe = (opt_returns.mean() * 252) / (opt_returns.std() * np.sqrt(252))
    opt_var = np.percentile(opt_returns, 5)
    opt_cvar = opt_returns[opt_returns <= opt_var].mean()

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
            "Metric": ["Annualized Std Dev", "Sharpe Ratio", "VaR (95%)", "CVaR (95%)"],
            "Uploaded Portfolio": [up_std, up_sharpe, up_var, up_cvar],
            "Optimized Portfolio": [opt_std, opt_sharpe, opt_var, opt_cvar]
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

    weighted_returns = returns_clean.dot(weights)
    cumulative_returns = (1 + weighted_returns).cumprod()
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
    with st.expander("ℹ️ Beta vs Correlation – What's the Difference?"):
        st.markdown("**Beta** measures how sensitive the portfolio is to market movements:")
        st.latex(r"\beta = \frac{\text{Cov}(R_p, R_b)}{\text{Var}(R_b)}")

        st.markdown("""
    - β = 1 → moves in sync with benchmark  
    - β < 1 → less sensitive  
    - β > 1 → more volatile

    **Correlation** shows direction of movement between portfolio and benchmark.  
    Value ranges from -1 (opposite) to +1 (perfect sync).
    """)
    for name, symbol in benchmarks.items():
        try:
            bm = yf.download(symbol, start=start, end=end)["Close"].pct_change().dropna()
            beta, corr = compute_benchmark_metrics(weighted_returns, bm)
            st.markdown(f"**{name}**  \n📐 Beta: `{beta:.3f}`  \n🔗 Correlation: `{corr:.3f}`")
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch benchmark {name}: {e}")

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