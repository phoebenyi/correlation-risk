import streamlit as st
import matplotlib.pyplot as plt
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
            roll_corr = aligned[t1].rolling(window).corr(aligned[t2])
            roll_corr = roll_corr.dropna()

            # fig, ax = plt.subplots(figsize=(7, 4))
            # ax.plot(roll_corr.index, roll_corr, label=f"{t1} vs {t2} ({window}d)", color="blue")
            # ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            # ax.axhline(1, color='black', linestyle=':', linewidth=0.5)
            # ax.axhline(-1, color='black', linestyle=':', linewidth=0.5)
            # ax.set_ylim(-1, 1)
            # ax.set_ylabel("Correlation")
            # ax.set_title(f"Rolling Correlation ({window}-day): {t1} vs {t2}")
            # ax.legend(fontsize=8, loc="upper left")
            # fig.tight_layout()
            # st.pyplot(fig)
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
                # buf = io.BytesIO()
                # fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")  # dpi controls scaling
                # st.image(buf, caption=f"{t1} vs {t2} ({window}d)", use_container_width="never", width=480)

        except Exception as e:
            st.error(f"Rolling correlation failed: {e}")