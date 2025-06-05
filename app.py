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

# Securely load from .streamlit/secrets.toml
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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
                        "access_token": session.access_token,
                        "email": auth_result.user.email
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

    groups_resp = supabase.table("groups").select("*").or_(f"user_id.eq.{uid},is_shared.eq.true").execute()
    # st.write("📦 Groups fetched:", groups_resp.data)
    groups = groups_resp.data

    group_names = [g["group_name"] for g in groups]
    group_lookup = {g["group_name"]: g for g in groups}

    selected_group = st.sidebar.selectbox("Select Group", group_names)
    if selected_group:
        tickers = group_lookup[selected_group]["tickers"]

    with st.sidebar.expander("➕ Create New Group"):
        new_name = st.text_input("Group Name")
        new_tickers = st.text_input("Tickers (comma-separated)")
        shared = st.checkbox("Make Public?")
        if st.button("Create Group"):
            tickers_list = [t.strip().upper() for t in new_tickers.split(",") if t.strip()]
            try:
                response = supabase.table("groups").insert({
                    "user_id": uid,
                    "group_name": new_name,
                    "tickers": tickers_list,
                    "is_shared": shared
                }).execute()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to create group: {e}")

    if selected_group:
        with st.sidebar.expander("✏️ Edit/Delete Group"):
            updated_tickers = st.text_input("Edit Tickers", ",".join(group_lookup[selected_group]["tickers"]))
            share_toggle = st.checkbox("Public?", value=group_lookup[selected_group]["is_shared"])
            if st.button("Update Group"):
                supabase.table("groups").update({
                    "tickers": [t.strip().upper() for t in updated_tickers.split(",")],
                    "is_shared": share_toggle
                }).eq("id", group_lookup[selected_group]["id"]).execute(headers={"Authorization": f"Bearer {token}"})
                st.rerun()
            if st.button("❌ Delete Group"):
                supabase.table("groups").delete().eq("id", group_lookup[selected_group]["id"]).execute()
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

# Rolling correlation
window = st.sidebar.slider("Rolling Correlation Window (days)", 20, 180, 60)

rolling_pairs = [(a, b) for i, a in enumerate(tickers) for b in tickers[i+1:]] if len(tickers) >= 2 else []
pair = st.sidebar.selectbox(
    "🔁 Choose Pair for Rolling Correlation",
    rolling_pairs,
    format_func=lambda x: f"{x[0]} vs {x[1]}",
    key="rolling_pair",
    help="Select any pair of stocks to see how their correlation changes over time"
)
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

            st.subheader("📊 Raw Price History Comparison")
            st.write("📉 **Raw Prices** – Actual trading prices. ✅ Useful for valuation, ❌ hard to compare across different price ranges.")
            st.write("📉 Using raw 'Close' prices (not adjusted for splits/dividends)")
            st.subheader("📊 Latest Price Data Snapshot")
            with st.expander("🔍 View Latest Raw Price Table"):
                st.dataframe(df.tail(10))
            st.line_chart(df)

            st.subheader("📊 Normalised Price History Comparison")
            df_norm = df.apply(lambda x: x / x.dropna().iloc[0] * 100 if x.dropna().shape[0] > 0 else x)
            st.write("📈 **Normalized Prices** – All lines start at 100. ✅ Great for comparing relative performance, ❌ loses actual price context.")
            with st.expander("🔍 View Latest Normalized Price Table"):
                st.dataframe(df_norm.tail(10).round(2))
            st.line_chart(df_norm)

            # st.subheader("📊 Log Price History Comparison")
            # import matplotlib.pyplot as plt
            # st.write("📊 **Log Prices** – Price on a logarithmic scale. ✅ Better for visualizing exponential growth, ❌ can distort small moves.")
            # with st.expander("🔍 View Latest Log Price Table"):
            #     st.dataframe(df.tail(10))
            # fig, ax = plt.subplots(figsize=(9, 4))
            # df.plot(ax=ax, logy=True)
            # ax.set_title("Log Price Chart")
            # ax.grid(True)
            # ax.legend(
            #     loc="upper center",
            #     bbox_to_anchor=(0.5, -0.3),
            #     ncol=min(4, len(df.columns)),
            #     frameon=False,
            #     fontsize=8
            # )
            # st.pyplot(fig, clear_figure=True)

            # ---------------------------------------------
            # Return Calculation based on frequency & type
            # ---------------------------------------------
            if freq == "Daily":
                returns = df.pct_change() if abs_or_pct == "% Change (Relative)" else df.diff()
                returns = returns.dropna(how="all")

            elif freq == "Monthly":
                monthly_prices = df.ffill().resample("M").last()
                if abs_or_pct == "% Change (Relative)":
                    returns = pd.concat([monthly_prices[col].pct_change() for col in monthly_prices.columns], axis=1)
                else:
                    returns = pd.concat([monthly_prices[col].diff() for col in monthly_prices.columns], axis=1)
                returns.columns = monthly_prices.columns


            elif freq == "Yearly":
                if overlap_window == "Yes":
                    if abs_or_pct == "% Change (Relative)":
                        returns = pd.concat([df[col].pct_change(252) for col in df.columns], axis=1)
                    else:
                        returns = pd.concat([df[col].diff(252) for col in df.columns], axis=1)
                    returns.columns = df.columns
                else:
                    yearly_prices = df.ffill().resample("YE").last()
                    if abs_or_pct == "% Change (Relative)":
                        returns = pd.concat([yearly_prices[col].pct_change() for col in yearly_prices.columns], axis=1)
                    else:
                        returns = pd.concat([yearly_prices[col].diff() for col in yearly_prices.columns], axis=1)
                    returns.columns = yearly_prices.columns


            # ---------------------------------------------
            # Correlation Matrix
            # ---------------------------------------------
            tickers = returns.columns.tolist()
            pairwise_corr = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

            for i, j in itertools.combinations(tickers, 2):
                x = returns[i].dropna()
                y = returns[j].dropna()
                overlap = x.index.intersection(y.index)
                if len(overlap) > 1:
                    corr_val = x[overlap].corr(y[overlap], method=corr_type.lower())
                    pairwise_corr.loc[i, j] = corr_val
                    pairwise_corr.loc[j, i] = corr_val

            # Fill diagonal with 1s
            np.fill_diagonal(pairwise_corr.values, 1.0)

            corr = pairwise_corr

            st.subheader("📍 Key Correlation Highlights")

            # ---------------------------------------------
            # Top +ve and -ve Correlated Pairs
            # ---------------------------------------------

            # Unstack matrix and remove self-pairs and duplicates
            corr_pairs = corr.where(~np.eye(len(corr), dtype=bool))  # mask diagonal
            corr_flat = corr_pairs.unstack().dropna().reset_index()
            corr_flat.columns = ['Stock A', 'Stock B', 'Correlation']
            corr_flat = corr_flat[corr_flat['Stock A'] < corr_flat['Stock B']]  # remove duplicates

            # Top +ve correlations
            top_pos = corr_flat.sort_values(by='Correlation', ascending=False).head(5)
            st.markdown("### 🔝 Top 5 Positively Correlated Pairs")
            st.dataframe(top_pos.reset_index(drop=True).round(3))

            # Top -ve correlations
            top_neg = corr_flat.sort_values(by='Correlation', ascending=True).head(5)
            st.markdown("### 🔻 Top 5 Negatively Correlated Pairs")
            st.dataframe(top_neg.reset_index(drop=True).round(3))

            # Highlight strong correlations
            threshold = 0.8
            strong_corr = corr_flat[abs(corr_flat["Correlation"]) > threshold]
            st.markdown(f"### ⚠️ Pairs with |Correlation| > {threshold}")
            st.dataframe(strong_corr.reset_index(drop=True).round(3))

            st.subheader(f"📌 {corr_type} Correlation Matrix")

            # correlation matrix table commented out to avoid repetitive display
            # st.dataframe(corr.round(3))

            fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title=f"{corr_type} Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)

            # removed seaborn because it's a static image and headers cannot be sticky
            # fig, ax = plt.subplots(figsize=(8, 6))
            # sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            # st.pyplot(fig)

            # ---------------------------------------------
            # CSV Export
            # ---------------------------------------------
            buffer = io.StringIO()
            corr.to_csv(buffer)
            csv = buffer.getvalue().encode("utf-8")
            st.download_button("⬇️ Download Correlation Matrix CSV", data=csv, file_name="correlation_matrix.csv", mime="text/csv")

            # ---------------------------------------------
            # Rolling Correlation — dynamic pair selector
            # ---------------------------------------------
            if len(tickers) >= 2 and pair and pair[0] in returns.columns and pair[1] in returns.columns:
                t1, t2 = pair
                st.subheader(f"🔁 Rolling Correlation: {t1} vs {t2}")
                try:
                    aligned = returns[[t1, t2]].dropna()
                    roll_corr = aligned[t1].rolling(window).corr(aligned[t2])
                    st.line_chart(roll_corr.dropna())
                except Exception as e:
                    st.error(f"Rolling correlation failed: {e}")
    st.success("✅ Analysis complete!")
else:
    st.info("👈 Select settings on the left and click 'Run Analysis'.")

            # REMOVED riskfolio-lib section as it is irrelevant to the current task
            # ---------------------------------------------
            # Risk Metrics and Portfolio Optimization
            # ---------------------------------------------
            # if riskfolio_available and len(tickers) > 1:

            #     def get_risk_metrics(returns, alpha=0.05, rf=0.0):
            #             metrics = {}
            #             for col in returns.columns:
            #                 r = returns[col].dropna()
            #                 if len(r) == 0:
            #                     continue  # skip this asset
            #                 var = np.percentile(r, 100 * alpha)
            #                 cvar = r[r <= var].mean()
            #                 sharpe = (r.mean() - rf) / r.std() * np.sqrt(252)
            #                 metrics[col] = {"VaR_0.05": var, "CVaR_0.05": cvar, "Sharpe": sharpe}
            #             return pd.DataFrame(metrics).T
                
            #     st.subheader("📉 Risk Metrics (VaR, CVaR, Sharpe)")
            #     returns_clean = returns.replace([np.inf, -np.inf], np.nan)
            #     min_valid_obs = 3
            #     returns_clean = returns_clean.dropna(axis=1, thresh=min_valid_obs)  # drop columns with too few valid obs
            #     returns_clean = returns_clean.dropna()  # drop any remaining NaN rows

            #     if returns_clean.shape[1] < 2 or returns_clean.shape[0] < 3:
            #         st.warning("⚠️ Not enough data to compute risk metrics or optimize portfolio. Need ≥ 2 assets and ≥ 3 return periods.")
            #     else:
            #         port = rp.Portfolio(returns=returns_clean)
            #         port.assets_stats(method_mu='hist', method_cov='hist')

            #         # Compute VaR, CVaR, Sharpe
            #         risk = get_risk_metrics(returns_clean)
            #         st.dataframe(risk.round(4))

            #         st.subheader("🧠 Portfolio Optimization (Max Sharpe)")
            #         w = port.optimization(model="Classic", rm="MV", obj="Sharpe", hist=True)
            #         st.dataframe(w.T.round(4))

            #         port_weights = w[w > 0].index.tolist()
            #         selected_weights = w.loc[port_weights].values.flatten()

            #         weighted_returns = returns[port_weights].mul(selected_weights, axis=1).sum(axis=1)
            #         cumulative_returns = (1 + weighted_returns).cumprod()

            #         st.subheader("📈 Optimized Portfolio Cumulative Returns")
            #         st.line_chart(cumulative_returns)

            #         st.subheader("📉 Drawdown Chart")
            #         drawdown = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
            #         st.line_chart(drawdown)

            #         st.subheader("🔁 Monthly Rebalanced Portfolio Returns")
            #         rebalance_returns = weighted_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
            #         st.line_chart((1 + rebalance_returns).cumprod())

            # elif not riskfolio_available:
            #     st.warning("Install `riskfolio-lib` to enable risk metrics.")
#     st.success("✅ Analysis complete!")
# else:
#     st.info("👈 Select settings on the left and click 'Run Analysis'.")