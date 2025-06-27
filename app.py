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

def show_benchmark_metrics(label, portfolio_returns, benchmarks, start, end, label_emoji=True):
    results = []
    for name, ticker in benchmarks.items():
        try:
            bm = yf.download(ticker, start=start, end=end)["Close"]
            bm_returns = bm.pct_change().dropna()
            beta, corr = compute_benchmark_metrics(portfolio_returns, bm_returns)
            alpha = portfolio_returns.mean() - bm_returns.mean()
            results.append({
                "Portfolio": label + (" 🟢" if alpha > 0 else " 🔴") if label_emoji else label,
                "Benchmark": name,
                "Alpha": round(alpha, 4),
                "Beta": round(beta, 4),
                "Correlation": round(corr, 4),
                "Emoji": "🟢" if alpha > 0 else "🔴"
            })
        except Exception as e:
            st.warning(f"Failed to compute metrics for benchmark {name}: {e}")
    return results

def render_concentration_metrics(portfolio_weights):
    from risk_analysis import compute_concentration_risk

    hhi, hhi_norm = compute_concentration_risk(portfolio_weights.values)
    st.metric("Herfindahl Index (HHI)", f"{hhi:.4f}")
    st.metric("Normalized HHI", f"{hhi_norm * 100:.2f}%")

def render_results(df, returns, df_norm, tickers, start, end, portfolio_weights):
    returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    tab_labels = [
    "📊 Prices",                          
    "📈 Correlation",                    
    "📉 Covariance",                     
    "🧬 Antifragility",                  
    "⚙️ Sharpe/Sortino Settings",        
    "📐 Risk Metrics",                   
    "📊 Volatility",                    
    "🧠 Optimization",                  
    "📈 Cumulative Returns",            
    "🔁 Rolling Volatility",            
    "📉 Alpha/Beta",                    
    "📊 Concentration Risk (HHI)",      
    "💬 Risk Suggestions"               
    ]
    tabs = st.tabs(tab_labels)
    
    # Price Visuals
    with tabs[0]:
        st.subheader("📊 Raw & Normalized Price Comparison")
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

    with tabs[3]:
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

    # Sharpe & Sortino Toggle
    with tabs[4]:
        st.subheader("📐 Sharpe & Sortino Settings")
        metric_mode = st.radio("Sharpe/Sortino Mode", ["Annualized", "Monthly"], index=0)
        scaling = np.sqrt(252) if metric_mode == "Annualized" else np.sqrt(12)
        st.session_state["sharpe_scaling"] = scaling

    # Risk Metrics
    with tabs[5]:
        from risk_display import render_risk_metrics
        render_risk_metrics(returns_clean, scaling)

    # Volatility Table
    with tabs[6]:
        from risk_display import render_volatility_tables
        render_volatility_tables(returns_clean)

    # Optimization
    with tabs[7]:
        from risk_display import render_portfolio_optimization
        render_portfolio_optimization(returns_clean, portfolio_weights, scaling)

    # Cumulative Return
    with tabs[8]:
        from risk_display import render_return_visuals
        render_return_visuals(returns_clean, portfolio_weights, scaling)

    # Rolling Volatility
    with tabs[9]:
        from risk_display import render_rolling_volatility
        render_rolling_volatility(returns_clean, portfolio_weights)
    
    # Alpha/Beta/Correlation
    with tabs[10]:
        st.subheader("📉 Alpha/Beta/Correlation Benchmark Metrics")
        st.caption("📘 Alpha and Beta are calculated using weekly (Friday-to-Friday) returns over the past 3 months.")

        benchmarks = {
            "Nasdaq-100": "^NDX",
            "S&P 500": "^GSPC",
            "NYSE": "^NYA"
        }

        if portfolio_weights is not None:
            portfolio_returns = returns_clean.dot(portfolio_weights)
            benchmark_results = show_benchmark_metrics("Uploaded Portfolio", portfolio_returns, benchmarks, start, end)
            if benchmark_results:
                df_benchmark = pd.DataFrame(benchmark_results).drop(columns="Emoji")
                st.dataframe(df_benchmark.round(4))
                best = max(benchmark_results, key=lambda x: x["Alpha"])
                st.success(f"🏆 Top Benchmark Outperformance: {best['Benchmark']} with Alpha = {best['Alpha']:.2%}")

    with tabs[11]:
        st.subheader("📊 Concentration Risk (HHI)")
        if portfolio_weights is not None:
            render_concentration_metrics(portfolio_weights)
        else:
            st.info("📂 No uploaded portfolio weights available for concentration analysis.")

    with tabs[12]:
        st.subheader("💬 Risk Contribution Suggestions")
        st.caption("Generated by analyzing marginal volatility and downside risk.")
        if portfolio_weights is not None:
            suggestions = suggest_portfolio_tweaks(portfolio_weights, returns_clean)
            for s in suggestions:
                st.markdown(f"- {s}")
                        
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
                portfolio_weights = None
                st.session_state["portfolio_weights"] = None
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