import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def display_rolling_correlation_viewer(returns, tickers):
    st.subheader("🔁 Rolling Correlation Viewer")
    window = st.slider("Rolling Window (days)", 20, 180, 60, key="rolling_window")

    pairs = [(a, b) for i, a in enumerate(tickers) for b in tickers[i+1:]]
    if not pairs:
        st.warning("Not enough tickers.")
        return

    t1, t2 = st.selectbox("Choose Pair", pairs, format_func=lambda x: f"{x[0]} vs {x[1]}")

    aligned = returns[[t1, t2]].dropna().copy()
    aligned = aligned[~aligned.index.duplicated()].sort_index()

    # Handle sudden spikes: z-score filter (optional, set zcap to None to disable)
    zcap = 4.0
    aligned = aligned[(np.abs((aligned - aligned.mean()) / aligned.std()) < zcap).all(axis=1)]

    if len(aligned) < window:
        st.warning(f"Not enough data after cleaning. {len(aligned)} rows.")
        return

    # Custom correlation calc
    def rolling_corr_custom(df, win):
        corr_vals = []
        idx = []
        for i in range(win, len(df)+1):
            sub = df.iloc[i-win:i]
            x = sub[t1]
            y = sub[t2]
            if x.std() > 1e-6 and y.std() > 1e-6:
                corr_vals.append(x.corr(y))
            else:
                corr_vals.append(np.nan)
            idx.append(sub.index[-1])
        return pd.Series(corr_vals, index=idx)

    rolling_corr = rolling_corr_custom(aligned, window)

    with st.expander("📈 Aligned Returns Used"):
        st.line_chart(aligned)

    st.markdown(f"### 📊 Rolling Correlation: {t1} vs {t2} ({window}-day window)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_corr.index, rolling_corr.values, color="blue", label=f"{t1} vs {t2} ({window}d)")
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Correlation")
    ax.set_title(f"Rolling Correlation (window={window}) – Excel-style, filtered")
    ax.legend()
    st.pyplot(fig)
