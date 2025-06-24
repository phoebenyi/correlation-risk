# 📄 correlation-risk/covariance_analysis.py
import pandas as pd
import streamlit as st
import plotly.express as px

def compute_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.cov()


def plot_covariance_heatmap(cov_matrix: pd.DataFrame):
    try:
        fig = px.imshow(
        cov_matrix,
        text_auto=".4f",
        title="Covariance Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Plotly failed to render covariance heatmap: {e}")
