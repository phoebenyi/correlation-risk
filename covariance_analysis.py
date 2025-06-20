# 📄 correlation-risk/covariance_analysis.py
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def compute_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.cov()

def plot_covariance_heatmap(cov_matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, fmt=".4f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
