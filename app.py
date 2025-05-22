import streamlit as st
import pandas as pd
import yfinance as yf
# import seaborn as sns
import matplotlib.pyplot as plt
import io
import plotly.express as px
import numpy as np
import itertools

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

# Dynamic ticker entry
default_tickers = ["ANET", "FN", "ALAB", "NVDA"]  # Arista, Fabrinet, Astera Labs, NVIDIA
user_input = st.sidebar.text_input("Enter Tickers (comma-separated)", value=",".join(default_tickers))
tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]

# Date range
start = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
threshold_days = st.sidebar.slider(
    "⚙️ Tolerance for start/end mismatch (days)",
    0, 10, 3,
    help="Recommended: 3 days to account for weekends, market holidays, market closures or data source lag. Set to 0 when using 'Daily' frequency for stricter matching."
)

# Return frequency and type
freq = st.sidebar.selectbox("Return Frequency", ["Daily", "Monthly", "Yearly"])
abs_or_pct = st.sidebar.radio("Return Type", ["% Change (Relative)", "Price Difference (Absolute)"])

# Overlap logic for YoY
overlap = st.sidebar.selectbox("Overlap Windows (for Yearly)?", ["Yes", "No"])

# Correlation method
corr_type = st.sidebar.selectbox("Correlation Method", ["Pearson", "Kendall", "Spearman"])

# Rolling correlation
window = st.sidebar.slider("Rolling Correlation Window (days)", 20, 180, 60)

# ---------------------------------------------
# Begin Analysis
# ---------------------------------------------
if st.sidebar.button("🔍 Run Analysis"):
    with st.spinner("Fetching and analyzing data..."):
        data = {}
        incomplete_data_notes = []
        for ticker in tickers:
            st.write(f"📥 Downloading {ticker}...")
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
                        continue
                else:
                    if "Close" in stock.columns:
                        val = stock["Close"]
                    else:
                        st.error(f"{ticker} has no valid price columns.")
                        continue

                if isinstance(val, pd.Series):
                    data[ticker] = val
                elif isinstance(val, pd.DataFrame) and val.shape[1] == 1:
                    data[ticker] = val.iloc[:, 0]
                else:
                    st.error(f"{ticker} has invalid format. Skipping.")
            else:
                st.error(f"{ticker} returned no data.")
        if incomplete_data_notes:
            st.markdown("### ⚠️ Some stocks have limited data")
            with st.expander("Click to view details"):
                df_missing = pd.DataFrame(incomplete_data_notes)
                st.dataframe(df_missing)

        if len(data) == 0:
            st.error("❌ No valid data downloaded.")
        else:
            df = pd.DataFrame(data).ffill()
            st.write(f"📊 Data range in df: {df.index.min().date()} to {df.index.max().date()}")
            buffer_prices = io.StringIO()
            df.to_csv(buffer_prices)
            buffer_prices.seek(0)
            st.download_button("⬇️ Download Price Data CSV", data=buffer_prices.getvalue(), file_name="prices.csv", mime="text/csv")

            if freq == "Yearly" and overlap == "No":
                temp = df.resample("Y").last()
                returns = temp.pct_change().dropna()
                st.caption("🧠 Using non-overlapping year-end returns (Excel-style).")
            elif freq == "Yearly" and overlap == "Yes":
                returns = df.pct_change(252).dropna()
            elif freq == "Monthly":
                temp = df.resample("M").last()
                returns = temp.pct_change().dropna()
            else:
                returns = df.pct_change()

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
            df_norm = df / df.iloc[0] * 100
            st.write("📈 **Normalized Prices** – All lines start at 100. ✅ Great for comparing relative performance, ❌ loses actual price context.")
            with st.expander("🔍 View Latest Normalized Price Table"):
                st.dataframe(df_norm.tail(10).round(2))
            st.line_chart(df_norm)

            st.subheader("📊 Log Price History Comparison")
            import matplotlib.pyplot as plt
            st.write("📊 **Log Prices** – Price on a logarithmic scale. ✅ Better for visualizing exponential growth, ❌ can distort small moves.")
            with st.expander("🔍 View Latest Log Price Table"):
                st.dataframe(df.tail(10))

            fig, ax = plt.subplots(figsize=(9, 4))
            df.plot(ax=ax, logy=True)
            ax.set_title("Log Price Chart")
            ax.grid(True)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.3),
                ncol=min(4, len(df.columns)),
                frameon=False,
                fontsize=8
            )
            st.pyplot(fig, clear_figure=True)

            # ---------------------------------------------
            # Return Calculation based on frequency & type
            # ---------------------------------------------
            if freq == "Daily":
                temp = df
                returns = temp.pct_change().dropna() if abs_or_pct == "% Change (Relative)" else temp.diff().dropna()

            elif freq == "Monthly":
                temp = df.resample("M").last()
                returns = temp.pct_change().dropna() if abs_or_pct == "% Change (Relative)" else temp.diff().dropna()

            elif freq == "Yearly":
                if overlap == "Yes":
                    # Use rolling 252-day change on full daily data (not resampled)
                    returns = df.pct_change(252).dropna() if abs_or_pct == "% Change (Relative)" else df.diff(252).dropna()
                else:
                    temp = df.resample("Y").last()
                    returns = temp.pct_change().dropna() if abs_or_pct == "% Change (Relative)" else temp.diff().dropna()

            # ---------------------------------------------
            # Correlation Matrix
            # ---------------------------------------------
            returns = df.pct_change()

            tickers = returns.columns.tolist()
            pairwise_corr = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

            for i, j in itertools.combinations(tickers, 2):
                x = returns[i]
                y = returns[j]

                # Temporary DataFrame to find overlapping non-NA pairs
                pair_df = pd.concat([x, y], axis=1, keys=[i, j]).dropna()

                if len(pair_df) > 1:
                    corr_val = pair_df[i].corr(pair_df[j], method=corr_type.lower())
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
            st.dataframe(corr.round(3))

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
            if len(tickers) >= 2:
                st.sidebar.markdown("### 🔁 Select Pair for Rolling Correlation")
                pair = st.sidebar.selectbox(
                    "Choose pair to plot",
                    [(a, b) for i, a in enumerate(tickers) for b in tickers[i+1:]],
                    format_func=lambda x: f"{x[0]} vs {x[1]}"
                )

                t1, t2 = pair
                st.subheader(f"🔁 Rolling Correlation: {t1} vs {t2}")
                try:
                    roll_corr = returns[t1].rolling(window).corr(returns[t2])
                    st.line_chart(roll_corr.dropna())
                except Exception as e:
                    st.error(f"Rolling correlation failed: {e}")

            # ---------------------------------------------
            # Risk Metrics and Portfolio Optimization
            # ---------------------------------------------
            if riskfolio_available and len(tickers) > 1:
                st.subheader("📉 Risk Metrics (VaR, CVaR, Sharpe)")
                returns_clean = returns.replace([np.inf, -np.inf], np.nan)

                # Drop columns (assets) that are mostly NaNs — but keep if they have enough data
                min_valid_obs = 30
                returns_clean = returns_clean.loc[:, returns_clean.notna().sum() > min_valid_obs]

                # Drop rows where remaining assets have missing values
                returns_clean = returns_clean.dropna()
                
                if len(returns_clean) < 30:
                    st.warning("⚠️ Fewer than 30 data points — risk metrics may be unreliable.")
                port = rp.Portfolio(returns=returns_clean)
                port.assets_stats(method_mu='hist', method_cov='hist')

                # Compute VaR, CVaR, Sharpe
                def get_risk_metrics(returns, alpha=0.05, rf=0.0):
                    metrics = {}
                    for col in returns.columns:
                        r = returns[col].dropna()
                        var = np.percentile(r, 100 * alpha)
                        cvar = r[r <= var].mean()
                        sharpe = (r.mean() - rf) / r.std() * np.sqrt(252)
                        metrics[col] = {"VaR_0.05": var, "CVaR_0.05": cvar, "Sharpe": sharpe}
                    return pd.DataFrame(metrics).T

                risk = get_risk_metrics(returns)
                st.dataframe(risk.round(4))

                st.subheader("🧠 Portfolio Optimization (Max Sharpe)")
                w = port.optimization(model="Classic", rm="MV", obj="Sharpe", hist=True)
                st.dataframe(w.T.round(4))

                port_weights = w[w > 0].index.tolist()
                selected_weights = w.loc[port_weights].values.flatten()

                weighted_returns = returns[port_weights].mul(selected_weights, axis=1).sum(axis=1)
                cumulative_returns = (1 + weighted_returns).cumprod()

                st.subheader("📈 Optimized Portfolio Cumulative Returns")
                st.line_chart(cumulative_returns)

                st.subheader("📉 Drawdown Chart")
                drawdown = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
                st.line_chart(drawdown)

                st.subheader("🔁 Monthly Rebalanced Portfolio Returns")
                rebalance_returns = weighted_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
                st.line_chart((1 + rebalance_returns).cumprod())

            elif not riskfolio_available:
                st.warning("Install `riskfolio-lib` to enable risk metrics.")
    st.success("✅ Analysis complete!")
else:
    st.info("👈 Select settings on the left and click 'Run Analysis'.")
