import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

def display_rolling_correlation_viewer(returns, tickers):
    st.subheader("🔁 Rolling Correlation Viewer")
    window = st.slider("Rolling Window (days)", 20, 180, 60, key="rolling_window")
    rolling_pairs = [(a, b) for i, a in enumerate(tickers) for b in tickers[i+1:]]

    if rolling_pairs:
        pair = st.selectbox("Choose Pair", rolling_pairs, format_func=lambda x: f"{x[0]} vs {x[1]}", key="rolling_pair_main")
        t1, t2 = pair
        try:
            aligned = returns[[t1, t2]].dropna()

            # Step 1: Debug chart to visually inspect raw returns
            with st.expander("📉 View raw return alignment for selected pair"):
                st.line_chart(aligned[[t1, t2]])

            # Step 2: Rolling correlation with flat/NaN handling
            roll_corr = []
            for i in range(window, len(aligned)):
                window_data = aligned.iloc[i - window:i]
                x = window_data[t1]
                y = window_data[t2]

                # Skip windows with NaNs
                if x.isnull().any() or y.isnull().any():
                    roll_corr.append(np.nan)
                    continue

                # Step 2: Skip flat/invariant windows
                if x.std() < 1e-6 or y.std() < 1e-6:
                    roll_corr.append(np.nan)
                    continue

                # Append correlation
                roll_corr.append(x.corr(y))

            # Convert to Series with correct index
            roll_corr = pd.Series(roll_corr, index=aligned.index[window:])

            # Optional: Smoothing to reduce jaggedness
            roll_corr = roll_corr.rolling(3, min_periods=1).mean()

            # Optional: Inspect spike-prone region (last 5 windows)
            with st.expander("🧪 Debug: Last 5 Windows Before Final Correlation Point"):
                if len(aligned) > window + 5:
                    st.dataframe(aligned.iloc[-(window + 5):])

            # Plot final correlation chart
            container_width = st.container()
            with container_width:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(roll_corr.index, roll_corr, label=f"{t1} vs {t2} ({window}d)", color="blue")
                ax.axhline(0, color='gray', linestyle='--', linewidth=1)
                ax.axhline(1, color='black', linestyle=':', linewidth=0.5)
                ax.axhline(-1, color='black', linestyle=':', linewidth=0.5)
                ax.set_ylim(-1, 1)
                ax.set_ylabel("Correlation")
                ax.set_title(f"Rolling Correlation ({window}-day): {t1} vs {t2}")
                ax.legend(fontsize=8, loc="upper left")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Rolling correlation failed: {e}")