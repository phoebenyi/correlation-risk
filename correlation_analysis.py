import pandas as pd
import numpy as np
import plotly.express as px
import itertools


def compute_pairwise_correlation(returns, method="pearson"):
    tickers = returns.columns.tolist()
    pairwise_corr = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

    for i, j in itertools.combinations(tickers, 2):
        x = returns[i].dropna()
        y = returns[j].dropna()
        overlap = x.index.intersection(y.index)
        if len(overlap) > 1:
            corr_val = x[overlap].corr(y[overlap], method=method.lower())
            pairwise_corr.loc[i, j] = corr_val
            pairwise_corr.loc[j, i] = corr_val

    np.fill_diagonal(pairwise_corr.values, 1.0)
    return pairwise_corr


def flatten_correlation_matrix(corr):
    corr_pairs = corr.where(~np.eye(len(corr), dtype=bool))
    corr_flat = corr_pairs.unstack().dropna().reset_index()
    corr_flat.columns = ['Stock A', 'Stock B', 'Correlation']
    corr_flat = corr_flat[corr_flat['Stock A'] < corr_flat['Stock B']]
    return corr_flat


def get_top_correlations(corr_flat, top_n=5):
    top_pos = corr_flat.sort_values(by='Correlation', ascending=False).head(top_n)
    top_neg = corr_flat.sort_values(by='Correlation', ascending=True).head(top_n)
    strong = corr_flat[abs(corr_flat["Correlation"]) > 0.8]
    return top_pos, top_neg, strong


def plot_correlation_heatmap(corr, method):
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=f"{method} Correlation Matrix"
    )
    return fig
