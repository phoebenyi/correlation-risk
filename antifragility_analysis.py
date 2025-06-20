# 📄 correlation-risk/antifragility_analysis.py
import pandas as pd
import streamlit as st

def compute_antifragility_scores(returns_df: pd.DataFrame) -> pd.DataFrame:
    # Assume antifragile = performs well when others perform badly
    scores = {}
    portfolio_return = returns_df.mean(axis=1)

    for col in returns_df.columns:
        asset_returns = returns_df[col]
        correlation = asset_returns.corr(portfolio_return)
        downside_capture = ((asset_returns[portfolio_return < 0]).mean()) / (portfolio_return[portfolio_return < 0].mean())
        
        scores[col] = {
            "Correlation to Portfolio": correlation,
            "Downside Capture Ratio": downside_capture,
            "Antifragility Score": -correlation * downside_capture  # Inverse of fragility
        }

    return pd.DataFrame(scores).T.sort_values("Antifragility Score", ascending=False)

def display_antifragility_table(scores_df: pd.DataFrame):
    st.dataframe(scores_df.style.format("{:.2f}"))
