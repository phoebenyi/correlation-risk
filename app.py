import streamlit as st
from supabase import create_client
import pandas as pd
import yfinance as yf
# import seaborn as sns
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

from risk_display import display_risk_and_optimization

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

with st.sidebar.expander("📤 Upload Custom Portfolio (CSV)"):
    st.markdown("CSV must include **Ticker** and **Shares** columns. Optionally: Purchase Date, Price at Purchase.")
    st.markdown("Example:")
    st.code("Ticker,Shares\nAAPL,50\nTSLA,30\nMSFT,20")
    portfolio_df, portfolio_weights = load_portfolio_from_csv()

tickers = []

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

                        if response.status_code == 201:
                            st.success("✅ Group created successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to create group: {response}")
                    else:
                        st.warning("Please enter both a group name and ticker list.")

    with st.sidebar.expander("📤 Upload Excel File of Stocks"):
        uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
        if uploaded_file:
            try:
                df_uploaded = pd.read_excel(uploaded_file)
                if df_uploaded.shape[1] >= 1:
                    uploaded_tickers = df_uploaded.iloc[:, 0].dropna().astype(str).str.upper().tolist()

                    # Preview tickers
                    st.markdown("### 📋 Parsed Tickers:")
                    st.code(", ".join(uploaded_tickers) if uploaded_tickers else "No tickers found")

                    # Default group name = filename
                    import os
                    suggested_name = os.path.splitext(uploaded_file.name)[0]
                    custom_name = st.text_input("Group Name", value=suggested_name)

                    # Prevent duplicate names
                    if custom_name in group_names:
                        st.warning("⚠️ Group name already exists. Please choose a unique name.")
                    elif st.button("Save as Group"):
                        if custom_name and uploaded_tickers:
                            response = supabase.table("groups").insert({
                                "user_id": uid,
                                "group_name": custom_name,
                                "tickers": uploaded_tickers
                            }).execute()

                            if response.status_code == 201:
                                st.success(f"✅ Group '{custom_name}' created with {len(uploaded_tickers)} tickers.")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to create group: {response}")
                        else:
                            st.warning("Please provide a valid group name and tickers.")

                else:
                    st.error("❌ Excel must have at least 1 column with tickers in Column A.")

            except Exception as e:
                st.error(f"❌ Failed to process Excel file: {e}")

        st.markdown("📌 **Put all stock symbols in Column A (first column)**.\nExample:\nAAPL\nMSFT\nTSLA")

    selected_group = st.sidebar.selectbox("Select Group", group_names)
    if selected_group:
        group_obj = group_lookup[selected_group]
        tickers = group_obj["tickers"]
        # Only allow editing if user is the group owner
        if "user" in st.session_state and group_obj["user_id"] == st.session_state["user"]["id"]:
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
            st.write(f"📊 Data range in df: {df.index.min().date()} to {df.index.max().date()}")
            buffer_prices = io.StringIO()
            df.to_csv(buffer_prices)
            buffer_prices.seek(0)
            st.download_button("⬇️ Download Price Data CSV", data=buffer_prices.getvalue(), file_name="prices.csv", mime="text/csv")

            # ---------------------------------------------
            # Price History Visualization
            # ---------------------------------------------

            display_raw_price_data(df)
            offer_price_data_download(df)
            df_norm = normalize_prices(df)
            display_normalized_price_data(df_norm)

            # ---------------------------------------------
            # Return Calculation based on frequency & type
            # ---------------------------------------------
            returns = compute_returns(df, freq, abs_or_pct, overlap_window)

            st.session_state["df"] = df
            st.session_state["returns"] = returns
            st.session_state["tickers"] = returns.columns.tolist()
            st.session_state["df_norm"] = df_norm

            # ---------------------------------------------
            # Correlation Matrix
            # ---------------------------------------------
            corr = compute_pairwise_correlation(returns, method=corr_type)
            st.session_state["corr"] = corr
            st.session_state["analysis_complete"] = True

            st.subheader("📍 Key Correlation Highlights")

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

            # CSV Export
            buffer = io.StringIO()
            corr.to_csv(buffer)
            csv = buffer.getvalue().encode("utf-8")
            st.download_button("⬇️ Download Correlation Matrix CSV", data=csv, file_name="correlation_matrix.csv", mime="text/csv")

if st.session_state.get("analysis_complete", False):
    st.subheader("📊 Raw Price History Comparison")
    st.line_chart(st.session_state["df"])

    st.subheader("📊 Normalised Price History Comparison")
    st.line_chart(st.session_state["df_norm"])

    st.subheader(f"📌 {corr_type} Correlation Matrix")
    fig = px.imshow(
        st.session_state["corr"],
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=f"{corr_type} Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap_postrun")

    # Rolling correlation section
    display_rolling_correlation_viewer(st.session_state["returns"], st.session_state["tickers"])

    # ---------------------------------------------
    # Risk Metrics and Portfolio Optimization
    # ---------------------------------------------
    df = st.session_state["df"]
    df_norm = st.session_state["df_norm"]
    returns = st.session_state["returns"]
    tickers = st.session_state["tickers"]

    if riskfolio_available and len(tickers) > 1:
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns_clean.shape[1] < 2 or returns_clean.shape[0] < 3:
            st.warning("⚠️ Not enough data to compute risk metrics or optimize portfolio. Need ≥ 2 assets and ≥ 3 return periods.")
        else:
            display_risk_and_optimization(returns_clean, start, end, portfolio_weights)
    elif not riskfolio_available:
        st.warning("Install `riskfolio-lib` to enable risk metrics and optimization.")