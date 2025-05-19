import streamlit as st
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import io

try:
    import riskfolio.Portfolio as pf
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

        # Download each ticker individually
        for ticker in tickers:
            st.write(f"📥 Downloading {ticker}...")
            stock = yf.download(ticker, start=start, end=end, group_by="column", auto_adjust=True)

            if not stock.empty:
                if "Adj Close" in stock.columns:
                    data[ticker] = stock["Adj Close"].squeeze()
                elif "Close" in stock.columns:
                    st.warning(f"{ticker} missing 'Adj Close'. Using 'Close' instead.")
                    data[ticker] = stock["Close"].squeeze()
                else:
                    st.error(f"{ticker} has no valid price columns.")
            else:
                st.error(f"{ticker} returned no data.")

        if len(data) == 0:
            st.error("❌ No valid data downloaded.")
        else:
            df = pd.DataFrame(data).dropna()
            st.subheader("📊 Price History")
            st.line_chart(df)

            # ---------------------------------------------
            # Return Calculation based on frequency & type
            # ---------------------------------------------
            if freq == "Daily":
                temp = df
            elif freq == "Monthly":
                temp = df.resample("M").last()
            elif freq == "Yearly":
                temp = df.resample("Y").last()

            if abs_or_pct == "% Change (Relative)":
                if freq == "Yearly" and overlap == "Yes":
                    returns = df.pct_change(252).dropna()
                else:
                    returns = temp.pct_change().dropna()
            else:
                returns = temp.diff().dropna()

            # Ignore October due to options pollution
            returns = returns[returns.index.month != 10]

            # Warn if data points too few
            if len(returns) < 30:
                st.warning("⚠️ Fewer than 30 data points — correlation may be unreliable.")

            # ---------------------------------------------
            # Correlation Matrix
            # ---------------------------------------------
            corr = returns.corr(method=corr_type.lower())
            st.subheader(f"📌 {corr_type} Correlation Matrix")
            st.dataframe(corr.round(3))

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)

            # ---------------------------------------------
            # CSV Export
            # ---------------------------------------------
            buffer = io.StringIO()
            corr.to_csv(buffer)
            csv = buffer.getvalue().encode("utf-8")
            st.download_button("⬇️ Download Correlation Matrix CSV", data=csv, file_name="correlation_matrix.csv", mime="text/csv")

            # ---------------------------------------------
            # Rolling Correlation Example (if 2+ tickers)
            # ---------------------------------------------
            if len(tickers) >= 2:
                st.subheader(f"🔁 Rolling Correlation: {tickers[0]} vs {tickers[1]}")
                try:
                    roll_corr = returns[tickers[0]].rolling(window).corr(returns[tickers[1]])
                    st.line_chart(roll_corr.dropna())
                except Exception as e:
                    st.error(f"Rolling correlation failed: {e}")

            # ---------------------------------------------
            # Risk Metrics and Portfolio Optimization
            # ---------------------------------------------
            if riskfolio_available and len(tickers) > 1:
                st.subheader("📉 Risk Metrics (VaR, CVaR, Sharpe)")
                port = pf.Portfolio(returns=returns)
                port.assets_stats(method_mu='hist', method_cov='hist')
                risk = port.risk_measures(method='hist', rf=0)
                st.dataframe(risk[["VaR_0.05", "CVaR_0.05", "Sharpe"]].round(4))

                st.subheader("🧠 Portfolio Optimization (Max Sharpe)")
                w = port.optimization(model="Classic", rm="MV", obj="Sharpe", hist=True)
                st.dataframe(w.T.round(4))

                port_weights = w[w > 0].index.tolist()
                weighted_returns = returns[port_weights].mul(w.T[port_weights].values, axis=1).sum(axis=1)
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
