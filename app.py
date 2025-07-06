import streamlit as st
from supabase import create_client
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io
import plotly.express as px
import numpy as np
import itertools

from portfolio_upload import load_portfolio_from_csv
from returns_analysis import compute_returns, normalize_prices
from price_display import display_raw_price_data, display_normalized_price_data, offer_price_data_download

from correlation_analysis import (
    compute_pairwise_correlation,
    flatten_correlation_matrix,
    get_top_correlations,
    plot_correlation_heatmap
)

from rolling_correlation import display_rolling_correlation_viewer

from risk_analysis import (
    get_risk_metrics,
    compute_volatility_table,
    compute_portfolio_risk,
    compute_benchmark_metrics,
    optimize_portfolio,
    suggest_portfolio_tweaks,
)

from covariance_analysis import compute_covariance_matrix, plot_covariance_heatmap

from antifragility_analysis import compute_antifragility_scores, display_antifragility_table

import datetime

from risk_display import render_concentration_metrics, show_benchmark_metrics

def render_concentration_metrics(portfolio_weights):
    from risk_analysis import compute_concentration_risk

    hhi, hhi_norm = compute_concentration_risk(portfolio_weights.values)
    st.metric("Herfindahl Index (HHI)", f"{hhi:.4f}")
    st.metric("Normalized HHI", f"{hhi_norm * 100:.2f}%")

def render_results(df, returns, df_norm, tickers, start, end, portfolio_weights):
    returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()

    tab_labels = [
    "📊 Prices",
    "📈 Correlation Matrix",
    "📉 Covariance Matrix",
    "📉 Volatility Overview",
    "🧮 Risk Metrics (Asset-Level)",
    "📐 Sharpe & Sortino (Portfolio)",
    "⚙️ Optimization & Suggestions",
    "📊 Cumulative Returns",
    "📉 Alpha/Beta/Tracking Error",
    "🧬 Antifragility"
    ]

    tabs = st.tabs(tab_labels)
    scaling = np.sqrt(252) # Annualization factor for daily returns
    
    # Price Visuals
    with tabs[0]:
        display_raw_price_data(df)
        offer_price_data_download(df)
        display_normalized_price_data(df_norm)

    # Correlation Matrix
    with tabs[1]:
        st.subheader("📍 Key Correlation Highlights")
        corr_type = st.session_state.get("corr_type", "Pearson")
        corr = compute_pairwise_correlation(returns, method=corr_type)
        st.session_state["corr"] = corr
        corr_flat = flatten_correlation_matrix(corr)
        top_pos, top_neg, strong_corr = get_top_correlations(corr_flat)

        st.markdown("### 🔝 Top 5 Positively Correlated Pairs")
        st.dataframe(top_pos.reset_index(drop=True).round(3))

        st.markdown("### 🔻 Top 5 Negatively Correlated Pairs")
        st.dataframe(top_neg.reset_index(drop=True).round(3))

        st.markdown("### ⚠️ Pairs with |Correlation| > 0.8")
        st.dataframe(strong_corr.reset_index(drop=True).round(3))

        st.subheader(f"📌 {corr_type} Correlation Matrix")
        fig = plot_correlation_heatmap(corr, corr_type)
        st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap_main")

        # Cross-Sector Correlation (TWSC-style)
        if "portfolio_classifications" in st.session_state:
            class_map = st.session_state["portfolio_classifications"]
            if class_map is not None:
                st.subheader("🧠 Thematic / Sector Correlation Analysis (TWSC)")
                tickers_with_class = [t for t in returns.columns if t in class_map.index]
                returns_classified = returns[tickers_with_class]
                class_map_filtered = class_map.loc[tickers_with_class]

                corr = compute_pairwise_correlation(returns_classified)
                corr_values = corr.stack().reset_index()
                corr_values.columns = ["Ticker A", "Ticker B", "Correlation"]
                corr_values["Class A"] = corr_values["Ticker A"].map(class_map_filtered)
                corr_values["Class B"] = corr_values["Ticker B"].map(class_map_filtered)
                corr_values = corr_values[corr_values["Ticker A"] != corr_values["Ticker B"]]

                grouped_corr = corr_values.groupby(["Class A", "Class B"])["Correlation"].mean().unstack().round(2)
                st.markdown("### 🧾 Average Correlation Between Groups")
                st.dataframe(grouped_corr.fillna("-"))

                try:
                    fig = px.imshow(
                        grouped_corr,
                        text_auto=True,
                        title="Cross-Group Correlation Heatmap (TWSC)",
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Failed to render sector heatmap: {e}")

        display_rolling_correlation_viewer(returns, tickers)

    # Covariance Matrix
    with tabs[2]:
        st.subheader("📉 Covariance Matrix")
        try:
            cov_matrix = compute_covariance_matrix(returns)
            st.dataframe(cov_matrix.round(4))
            buffer_cov = io.StringIO()
            cov_matrix.to_csv(buffer_cov)
            st.download_button("⬇️ Download Covariance Matrix", buffer_cov.getvalue(), "covariance_matrix.csv", "text/csv")
            plot_covariance_heatmap(cov_matrix)
        except Exception as e:
            st.warning(f"Failed to compute covariance matrix: {e}")

    # Volatility
    with tabs[3]:
        from risk_display import (
            render_volatility_tables,
            render_relative_volatility_table,
            render_rolling_volatility,
            render_tracking_error,
            render_longitudinal_volatility_table,
            render_garch_volatility_forecast
        )
        st.subheader("📉 Volatility Overview")
        render_volatility_tables(returns_clean)
        render_relative_volatility_table(returns_clean)
        render_rolling_volatility(returns_clean, portfolio_weights)
        render_longitudinal_volatility_table(returns_clean)
        render_garch_volatility_forecast(returns_clean)

    # Risk Metrics
    with tabs[4]:
        from risk_display import render_risk_metrics, render_return_histogram
        render_risk_metrics(returns_clean, scaling)
        render_return_histogram(returns_clean, portfolio_weights)

    # Sharpe & Sortino Ratios
    with tabs[5]:
        st.subheader("📊 Sharpe & Sortino Ratios")

        # ℹ️ Explanatory Section
        with st.expander("❓ What Are Sharpe & Sortino Ratios?"):
            st.markdown(r"""
            ### 🧮 **Sharpe Ratio – Total Risk-Adjusted Return**

            The **Sharpe Ratio** answers:
            > “How much return am I earning per unit of total risk?”

            $$
            \text{Sharpe Ratio} = \frac{ \mathbb{E}[R_p - R_f] }{ \sigma_p } \times \sqrt{252}
            $$

            - $R_p$: Portfolio return  
            - $R_f$: Risk-free rate (e.g., 2%)  
            - $\sigma_p$: Standard deviation of **all** returns  
            - $\sqrt{252}$: Annualizes daily returns

            **Interpretation:**
            - Sharpe > 1 → good
            - Sharpe > 2 → excellent
            - Sharpe < 1 → high risk for return earned

            ---

            ### 🎯 **Sortino Ratio – Downside Risk-Adjusted Return**

            The **Sortino Ratio** only penalizes **downside** volatility:

            $$
            \text{Sortino Ratio} = \frac{ \mathbb{E}[R_p - R_f] }{ \sigma_{\text{down}} } \times \sqrt{252}
            $$

            - $\sigma_{\text{down}}$: Standard deviation of **negative** excess returns only

            **Interpretation:**
            - Sortino is usually higher than Sharpe
            - Focuses on “bad volatility” only
            - Useful for capital preservation and conservative investing

            """)

        if portfolio_weights is not None:
            st.download_button("📥 Export returns_clean (aligned)", returns_clean[portfolio_weights.index].to_csv(), "returns_clean_aligned.csv")
            st.download_button("📥 Export portfolio_weights", portfolio_weights.to_csv(), "weights.csv")

        # Actual metric calculation
        scaling_annual = np.sqrt(252)
        scaling_monthly = np.sqrt(12)

        def compute_ratios(returns, scale, risk_free_rate=0.02 / 252):
            excess_returns = returns - risk_free_rate
            mean_excess = excess_returns.mean()
            std_total = returns.std()
            std_downside = excess_returns[excess_returns < 0].std()
            sharpe = mean_excess / std_total * scale if std_total > 0 else np.nan
            sortino = mean_excess / std_downside * scale if std_downside > 0 else np.nan
            return sharpe, sortino

        if portfolio_weights is not None:
            uploaded_returns = returns_clean.dot(portfolio_weights)
            sharpe_ann, sortino_ann = compute_ratios(uploaded_returns, scaling_annual)
            sharpe_month, sortino_month = compute_ratios(uploaded_returns, scaling_monthly)

            st.markdown("### 📈 Uploaded Portfolio")
            st.metric("Sharpe (Annualized)", f"{sharpe_ann:.4f}")
            st.metric("Sortino (Annualized)", f"{sortino_ann:.4f}")
            st.metric("Sharpe (Monthly)", f"{sharpe_month:.4f}")
            st.metric("Sortino (Monthly)", f"{sortino_month:.4f}")
        else:
            st.info("📂 No uploaded portfolio weights provided. Showing optimized portfolio metrics.")
            opt_weights = optimize_portfolio(returns_clean).values.flatten()
            opt_returns = returns_clean.dot(opt_weights)
            sharpe_ann, sortino_ann = compute_ratios(opt_returns, scaling_annual)
            sharpe_month, sortino_month = compute_ratios(opt_returns, scaling_monthly)

            st.markdown("### ⚙️ Optimized Portfolio")
            st.metric("Sharpe (Annualized)", f"{sharpe_ann:.4f}")
            st.metric("Sortino (Annualized)", f"{sortino_ann:.4f}")
            st.metric("Sharpe (Monthly)", f"{sharpe_month:.4f}")
            st.metric("Sortino (Monthly)", f"{sortino_month:.4f}")

    # Optimization & Suggestions
    with tabs[6]:
        from risk_display import render_portfolio_optimization, render_concentration_metrics
        st.subheader("⚙️ Optimization & Risk Suggestions")
        st.caption("Generated by analyzing marginal volatility and downside risk.")
        if portfolio_weights is not None:
            st.subheader("📊 Concentration Risk (HHI)")
            render_concentration_metrics(portfolio_weights)
            suggestions = suggest_portfolio_tweaks(portfolio_weights, returns_clean)
            st.markdown("### 💬 Suggestions")
            for s in suggestions:
                st.markdown(f"- {s}")
        else:
            st.info("📂 No uploaded portfolio weights available for concentration analysis.")

    # Cumulative Returns
    with tabs[7]:
        from risk_display import render_return_visuals
        st.subheader("📊 Cumulative Return & Benchmark Analysis")
        render_return_visuals(returns_clean, portfolio_weights, scaling)
    
    # Benchmark
    with tabs[8]:
        from risk_display import render_tracking_error
        st.subheader("📉 Alpha/Beta/Correlation vs Benchmark")
        st.caption("📘 Alpha and Beta are calculated using weekly (Friday-to-Friday) returns over the past 3 months.")

        default_benchmarks = {
            "S&P 500": "^GSPC",
            "Nasdaq-100": "^NDX",
            "Dow Jones": "^DJI",
            "STI": "^STI"
        }

        user_input = st.sidebar.text_input("➕ Add Benchmarks (format: ^DJI:Dow, ^STI:STI)", key="benchmark_input")
        user_benchmarks = {}
        if user_input:
            for entry in user_input.split(","):
                if ":" in entry:
                    ticker, label = entry.split(":")
                    user_benchmarks[label.strip()] = ticker.strip()

        benchmarks = {**default_benchmarks, **user_benchmarks}
        selected_benchmark_label = st.selectbox("📈 Benchmark for Tracking Error", list(benchmarks.keys()))
        benchmark_symbol = benchmarks[selected_benchmark_label]

        st.subheader("📏 Tracking Error")

        with st.expander("❓ What is Tracking Error?"):
            st.markdown("""
            ### 📏 Tracking Error

            Measures how closely your portfolio tracks a benchmark.

            $$
            \text{TE} = \text{std}(R_p - R_b) \times \sqrt{252}
            $$

            - $R_p$: Portfolio return  
            - $R_b$: Benchmark return

            **Low TE** → tightly follows benchmark  
            **High TE** → deviates from benchmark  
            """)

        render_tracking_error(returns_clean, portfolio_weights, benchmark_symbol)

        st.markdown("---")
        
        if portfolio_weights is not None:
            portfolio_daily = returns_clean.dot(portfolio_weights)

            if portfolio_daily.empty:
                st.error("❌ Portfolio returns are empty!")
            else:
                benchmark_results = show_benchmark_metrics(portfolio_daily, benchmarks, start, end)
                if benchmark_results:
                    df_benchmark = pd.DataFrame(benchmark_results)

                    # Handle edge case: if somehow Alpha/Beta/Correlation are NaN
                    if df_benchmark.empty or df_benchmark[["Alpha", "Beta", "Correlation"]].isnull().all().all():
                        st.warning("⚠️ Computed benchmark results are empty or contain only NaNs.")
                    else:
                        st.dataframe(df_benchmark.round(4))
                        best = max(benchmark_results, key=lambda x: x["Alpha"])
                        st.success(f"🏆 Top Benchmark Outperformance: {best['Benchmark']} with Alpha = {best['Alpha']:.2%}")
                else:
                    st.warning("⚠️ No benchmark metrics calculated. Possibly no overlapping data.")

    # Antifragility Analysis
    with tabs[9]:
        st.subheader("🧬 Antifragility Analysis")
        try:
            scores_df = compute_antifragility_scores(returns)
            st.caption("🔍 Antifragility Score = - (correlation × downside capture). Higher = more resilient to drawdowns.")
            st.dataframe(scores_df.style.highlight_max(axis=0).format("{:.2f}"))

            buffer_anti = io.StringIO()
            scores_df.to_csv(buffer_anti)
            st.download_button("⬇️ Download Antifragility Scores", buffer_anti.getvalue(), "antifragility_scores.csv", "text/csv")
        except Exception as e:
            st.warning(f"Failed to compute antifragility scores: {e}")
                        
# Securely load from .streamlit/secrets.toml
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
# supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_client():
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    user = st.session_state.get("user")
    if user:
        client.auth.set_session(user["access_token"], user["refresh_token"])
    return client

supabase = get_client()

try:
    import riskfolio as rp
    riskfolio_available = True
except ImportError:
    riskfolio_available = False

st.set_page_config(page_title="Correlation & Risk Dashboard", layout="wide")
st.title("📈 Dynamic Stock Correlation & Risk Analysis")

# ✅ Show toast if group was just saved
if "last_saved_group" in st.session_state:
    st.success(f"✅ Group '{st.session_state['last_saved_group']}' saved successfully!")
    del st.session_state["last_saved_group"]

# ---------------------------------------------
# Sidebar Configuration Inputs
# ---------------------------------------------

st.sidebar.header("Configuration")
st.sidebar.title("🔐 Login")

auth_action = st.sidebar.radio("Choose:", ["Login", "Signup"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button(auth_action):
    try:
        if auth_action == "Signup":
            user = supabase.auth.sign_up({"email": email, "password": password})
            st.sidebar.success("✅ Please check your email to confirm your account. After confirming, you can log in.")
        else:
            auth_result = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            session = supabase.auth.get_session()
            if session:
                if auth_result.user.confirmed_at is None:
                    st.sidebar.warning("📧 Please confirm your email before logging in.")
                else:
                    st.session_state["user"] = {
                        "id": auth_result.user.id,
                        "email": auth_result.user.email,
                        "access_token": session.access_token,
                        "refresh_token": session.refresh_token
                    }
                    st.sidebar.success("✅ Logged in!")
                    st.rerun()
            else:
                st.sidebar.error("❌ Login failed: No valid session.")

    except Exception as e:
        st.sidebar.error(f"{auth_action} failed: {e}")

# Logout Button
if "user" in st.session_state and st.sidebar.button("Logout"):
    st.session_state.pop("user", None)
    supabase.auth.sign_out()
    st.rerun()

if "user" in st.session_state:
    user = st.session_state.get("user")
    uid = user.get("id") if user else None
    token = user.get("access_token") if user else None

    st.sidebar.success("✅ Logged in successfully!")
    try:
        email = user.get("email", "Unknown")
        st.sidebar.markdown(f"**Logged in as:** {email}👤")
    except Exception:
        st.sidebar.warning("⚠️ Failed to retrieve user email.")

    
    st.sidebar.subheader("📁 Your Groups")

    uid = st.session_state["user"].get("id")
    groups_resp = supabase.table("groups").select("*").eq("user_id", uid).execute()
    groups = groups_resp.data

    group_names = sorted([g["group_name"] for g in groups])
    group_lookup = {g["group_name"]: g for g in groups}

    with st.sidebar.expander("➕ Create New Group"):
                new_group_name = st.text_input("New Group Name")
                new_group_tickers = st.text_input("Tickers (comma-separated)", help="Example: AAPL, MSFT, TSLA")

                if st.button("Create Group"):
                    if new_group_name and new_group_tickers:
                        tickers_list = [t.strip().upper() for t in new_group_tickers.split(",") if t.strip()]
                        response = supabase.table("groups").insert({
                            "user_id": uid,
                            "group_name": new_group_name,
                            "tickers": tickers_list
                        }).execute()

                        if response.data:
                            st.session_state["last_saved_group"] = new_group_name
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to save group. Response: {response}")
                    else:
                        st.warning("Please enter both a group name and ticker list.")
    
    with st.sidebar.expander("📤 Upload Your Portfolio (CSV) or Use a Group"):
        st.markdown("CSV must include **Ticker** and **Shares** columns. Example:")
        st.code("Ticker,Shares\nAAPL,50\nTSLA,30")
        source_choice = st.radio("Choose Input Method", ["Upload CSV", "Use Group"], key="input_method_radio")

        tickers = []
        portfolio_weights = None

        if source_choice == "Upload CSV":
            portfolio_df, portfolio_weights, classifications = load_portfolio_from_csv()
            if portfolio_df is not None:
                tickers = portfolio_df["Ticker"].tolist()
                st.session_state["portfolio_classifications"] = classifications

        elif source_choice == "Use Group":
            selected_group = st.selectbox("Select Group", group_names)
            
            if selected_group:
                group_obj = group_lookup[selected_group]
                tickers = group_obj["tickers"]
                st.session_state["tickers"] = tickers

                if tickers:
                    st.markdown("### ⚖️ Edit Weights (optional)")

                    # Provide a default weight dictionary (equal weight)
                    default_weights = {t: 1 / len(tickers) for t in tickers}

                    # Use Streamlit number inputs to allow editing weights
                    edited_weights = {}
                    for t in tickers:
                        edited_weights[t] = st.number_input(
                            label=f"Weight for {t}",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(default_weights[t]),
                            step=0.01,
                            key=f"weight_input_{t}"
                        )

                    # Normalize weights to sum to 1
                    total = sum(edited_weights.values())

                    # ✅ Move the check here, after total is defined:
                    if abs(total - 1.0) > 0.01:
                        st.warning("Weights will be normalized to sum to 100%.")

                    if total == 0:
                        st.warning("⚠️ Total weight is 0. Please adjust weights.")
                        portfolio_weights = None
                        st.session_state["portfolio_weights"] = None
                    else:
                        normalized_weights = {t: w / total for t, w in edited_weights.items()}
                        portfolio_weights = pd.Series(normalized_weights)
                        st.session_state["portfolio_weights"] = portfolio_weights

                        st.caption(f"🎯 Total Weight (before normalization): {total:.2f}")
                        st.dataframe(pd.Series(normalized_weights, name="Normalized Weight").round(4))
                    
                    # Pie chart of weights
                    st.markdown("📊 **Weight Allocation Pie Chart**")
                    pie_df = pd.DataFrame({
                        "Ticker": list(normalized_weights.keys()),
                        "Weight": list(normalized_weights.values())
                    })

                    fig = px.pie(pie_df, values="Weight", names="Ticker", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("📈 **Weight Allocation Bar Chart**")
                    fig_bar = px.bar(pie_df, x="Ticker", y="Weight", text="Weight", title="Portfolio Weights")
                    fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                    fig_bar.update_layout(yaxis_tickformat=".0%", uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig_bar, use_container_width=True)

                if group_obj["user_id"] == st.session_state["user"]["id"]:
                    with st.sidebar.expander("✏️ Edit/Delete Group"):
                        updated_tickers = st.text_input("Edit Tickers", ",".join(group_obj["tickers"]))
                        if st.button("Update Group"):
                            supabase.table("groups").update({
                                "tickers": [t.strip().upper() for t in updated_tickers.split(",")]
                            }).eq("id", group_obj["id"]).execute()
                            st.rerun()
                        if st.button("❌ Delete Group"):
                            supabase.table("groups").delete().eq("id", group_obj["id"]).execute()
                            st.rerun()

else:
    st.sidebar.info("Please log in to manage groups.")

# Date range
import datetime
today = datetime.date.today()
default_start = today - pd.DateOffset(years=5)
start = st.sidebar.date_input("Start Date", value=default_start.date(), min_value=datetime.date(2000, 1, 1), max_value=today)
end = st.sidebar.date_input("End Date", value=today, min_value=datetime.date(2000, 1, 1), max_value=today)
threshold_days = st.sidebar.slider(
    "⚙️ Tolerance for start/end mismatch (days)",
    0, 10, 3,
    help="Recommended: 3 days to account for weekends, market holidays, market closures or data source lag. Set to 0 when using 'Daily' frequency for stricter matching."
)

# Return frequency and type
freq = st.sidebar.selectbox("Return Frequency", ["Daily", "Monthly", "Yearly"])
abs_or_pct = st.sidebar.radio("Return Type", ["% Change (Relative)", "Price (Absolute)"])

# Overlap logic for YoY
overlap_window = st.sidebar.selectbox("Overlap Windows (for Yearly)?", ["Yes", "No"])

# Correlation method
corr_type = st.sidebar.selectbox("Correlation Method", ["Pearson", "Kendall", "Spearman"])

# ---------------------------------------------
# Begin Analysis
# ---------------------------------------------
if st.sidebar.button("🔍 Run Analysis"):
    with st.spinner("Fetching and analyzing data..."):
        data = {}
        incomplete_data_notes = []
        with st.status("📥 Downloading data...", expanded=True) as status:
            failed = []
            for ticker in tickers:
                stock = yf.download(ticker, start=start, end=end, group_by="column", auto_adjust=True)

                if not stock.empty:
                    # Check if the date range is incomplete
                    actual_start = stock.index.min().date()
                    actual_end = stock.index.max().date()
                    user_start = pd.to_datetime(start).date()
                    user_end = pd.to_datetime(end).date()

                    if (actual_start - user_start).days > threshold_days or (user_end - actual_end).days > threshold_days:
                        reason = "IPO, delisting, or missing Yahoo data"
                        incomplete_data_notes.append({
                            "Ticker": ticker,
                            "Available From": actual_start,
                            "Available To": actual_end,
                            "Requested From": user_start,
                            "Requested To": user_end,
                            "Reason": reason
                        })

                    # Extract price series
                    if isinstance(stock.columns, pd.MultiIndex):
                        try:
                            val = stock["Close"][ticker].dropna()
                        except KeyError:
                            st.warning(f"⚠️ 'Close' prices not found for {ticker}.")
                            failed.append(ticker)
                            continue
                    else:
                        if "Close" in stock.columns:
                            val = stock["Close"]
                        else:
                            st.error(f"{ticker} has no valid price columns.")
                            failed.append(ticker)
                            continue

                    if isinstance(val, pd.Series):
                        data[ticker] = val
                    elif isinstance(val, pd.DataFrame) and val.shape[1] == 1:
                        data[ticker] = val.iloc[:, 0]
                    else:
                        st.error(f"{ticker} has invalid format. Skipping.")
                        failed.append(ticker)
                else:
                    st.error(f"{ticker} returned no data.")
                    failed.append(ticker)
            status.update(label=f"✅ Download complete. ({len(tickers) - len(failed)} success, {len(failed)} failed)", state="complete")

        if incomplete_data_notes:
            st.markdown("### ⚠️ Some stocks have limited data")
            with st.expander("Click to view details"):
                df_missing = pd.DataFrame(incomplete_data_notes)
                st.dataframe(df_missing)

        if len(data) == 0:
            st.error("❌ No valid data downloaded.")
        else:
            df = pd.DataFrame(data)
            df_norm = normalize_prices(df)
            returns = compute_returns(df, freq, abs_or_pct, overlap_window)
            st.write(f"📊 Data range in df: {df.index.min().date()} to {df.index.max().date()}")
            buffer_prices = io.StringIO()
            df.to_csv(buffer_prices)
            buffer_prices.seek(0)
            st.download_button("⬇️ Download Price Data CSV", data=buffer_prices.getvalue(), file_name="prices.csv", mime="text/csv")

            st.session_state["df"] = df
            st.session_state["returns"] = returns
            st.session_state["df_norm"] = df_norm
            st.session_state["tickers"] = returns.columns.tolist()
            st.session_state["start"] = start
            st.session_state["end"] = end
            st.session_state["portfolio_weights"] = portfolio_weights
            st.session_state["corr_type"] = corr_type
            st.session_state["analysis_complete"] = True

if st.session_state.get("analysis_complete"):
    render_results(
        df=st.session_state["df"],
        returns=st.session_state["returns"],
        df_norm=st.session_state["df_norm"],
        tickers=st.session_state["tickers"],
        start=st.session_state["start"],
        end=st.session_state["end"],
        portfolio_weights=st.session_state.get("portfolio_weights")
    )