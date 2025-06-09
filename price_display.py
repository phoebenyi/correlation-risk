import streamlit as st
import pandas as pd
import io


def display_raw_price_data(df):
    st.subheader("📊 Raw Price History Comparison")
    st.write("📉 **Raw Prices** – Actual trading prices. ✅ Useful for valuation, ❌ hard to compare across different price ranges.")
    st.write("📉 Using raw 'Close' prices (not adjusted for splits/dividends)")
    st.subheader("📊 Latest Price Data Snapshot")
    with st.expander("🔍 View Latest Raw Price Table"):
        st.dataframe(df.tail(10))
    st.line_chart(df)


def display_normalized_price_data(df_norm):
    st.subheader("📊 Normalised Price History Comparison")
    df_norm = df.apply(lambda x: x / x.dropna().iloc[0] * 100 if x.dropna().shape[0] > 0 else x)
    st.write("📈 **Normalized Prices** – All lines start at 100. ✅ Great for comparing relative performance, ❌ loses actual price context.")
    with st.expander("🔍 View Latest Normalized Price Table"):
        st.dataframe(df_norm.tail(10).round(2))
    st.line_chart(df_norm)


def offer_price_data_download(df):
    buffer_prices = io.StringIO()
    df.to_csv(buffer_prices)
    buffer_prices.seek(0)
    st.download_button("⬇️ Download Price Data CSV", data=buffer_prices.getvalue(), file_name="prices.csv", mime="text/csv")