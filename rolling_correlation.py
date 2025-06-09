import streamlit as st


def display_rolling_correlation_viewer(returns, tickers):
    st.subheader("🔁 Rolling Correlation Viewer")
    window = st.slider("Rolling Window (days)", 20, 180, 60, key="rolling_window")
    rolling_pairs = [(a, b) for i, a in enumerate(tickers) for b in tickers[i+1:]]

    if rolling_pairs:
        pair = st.selectbox("Choose Pair", rolling_pairs, format_func=lambda x: f"{x[0]} vs {x[1]}", key="rolling_pair_main")
        t1, t2 = pair
        try:
            aligned = returns[[t1, t2]].dropna()
            roll_corr = aligned[t1].rolling(window).corr(aligned[t2])
            st.line_chart(roll_corr.dropna())
        except Exception as e:
            st.error(f"Rolling correlation failed: {e}")