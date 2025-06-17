import streamlit as st
import pandas as pd
import yfinance as yf

def load_portfolio_from_csv(key="portfolio_csv"):
    """
    Upload and parse a user-provided portfolio CSV with columns:
    Required: Ticker, Shares
    Optional: Classification
    """
    st.subheader("📤 Upload Your Portfolio CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key=key)

    if not uploaded_file:
        return None, None, None

    try:
        df = pd.read_csv(uploaded_file)
        required_cols = {"Ticker", "Shares"}
        if not required_cols.issubset(set(df.columns)):
            st.error("❌ CSV must contain at least 'Ticker' and 'Shares' columns.")
            return None, None, None

        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
        df = df.dropna(subset=["Ticker", "Shares"])

        tickers = df["Ticker"].tolist()
        latest_prices = {}

        for ticker in tickers:
            try:
                data = yf.download(ticker, period="1d", auto_adjust=True, progress=False)
                if not data.empty and "Close" in data.columns:
                    close_price = data["Close"].dropna()
                    latest_prices[ticker] = float(close_price.iloc[-1]) if not close_price.empty else np.nan
                else:
                    st.warning(f"⚠️ Could not fetch price for {ticker}")
            except Exception as e:
                st.warning(f"⚠️ Error fetching {ticker}: {e}")

        df["Latest Price"] = df["Ticker"].map(latest_prices)
        df["Market Value"] = df["Shares"] * df["Latest Price"]
        total_value = df["Market Value"].sum()
        df["Weight"] = df["Market Value"] / total_value

        st.success("✅ Portfolio loaded and weights calculated.")
        
        display_cols = ["Ticker", "Shares", "Latest Price", "Market Value", "Weight"]
        if "Classification" in df.columns:
            display_cols.append("Classification")

        st.dataframe(df[display_cols].round(4))

        df = df.drop_duplicates(subset=["Ticker"])
        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

        classifications = df.set_index("Ticker")["Classification"] if "Classification" in df.columns else None
        return df, df.set_index("Ticker")["Weight"], classifications

    except Exception as e:
        st.error(f"❌ Failed to process CSV: {e}")
        return None, None, None