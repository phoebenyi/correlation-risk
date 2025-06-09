import streamlit as st
import pandas as pd
import yfinance as yf


def load_portfolio_from_csv():
    """
    Upload and parse a user-provided portfolio CSV with columns:
    Ticker, Shares [, Purchase Date, Price at Purchase]
    Returns cleaned DataFrame and computed weights based on latest prices.
    """
    st.subheader("📤 Upload Your Portfolio CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="portfolio_csv")

    if not uploaded_file:
        return None, None

    try:
        df = pd.read_csv(uploaded_file)
        required_cols = {"Ticker", "Shares"}
        if not required_cols.issubset(set(df.columns)):
            st.error("❌ CSV must contain at least 'Ticker' and 'Shares' columns.")
            return None, None

        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
        df = df.dropna(subset=["Ticker", "Shares"])

        tickers = df["Ticker"].tolist()
        prices = yf.download(tickers, period="1d", group_by="ticker", auto_adjust=True, threads=False)

        latest_prices = {}
        for ticker in tickers:
            try:
                if isinstance(prices.columns, pd.MultiIndex):
                    latest_prices[ticker] = prices["Close"][ticker].dropna().iloc[-1]
                else:
                    latest_prices[ticker] = prices["Close"].dropna().iloc[-1]
            except:
                st.warning(f"⚠️ Could not fetch price for {ticker}")

        df["Latest Price"] = df["Ticker"].map(latest_prices)
        df["Market Value"] = df["Shares"] * df["Latest Price"]
        total_value = df["Market Value"].sum()
        df["Weight"] = df["Market Value"] / total_value

        st.success("✅ Portfolio loaded and weights calculated.")
        st.dataframe(df[["Ticker", "Shares", "Latest Price", "Market Value", "Weight"]].round(4))

        return df, df.set_index("Ticker")["Weight"]

    except Exception as e:
        st.error(f"❌ Failed to process CSV: {e}")
        return None, None
