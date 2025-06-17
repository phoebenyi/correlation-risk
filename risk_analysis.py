import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp

# -------------------------
# 🔢 Risk Metric Calculations
# -------------------------
def get_risk_metrics(returns, alpha=0.05, rf=0.0):
    """
    Computes VaR, CVaR, Sharpe Ratio, and Median Return for each asset.
    Formulas:
      - VaR_α = Percentile(returns, α)
      - CVaR_α = Mean of returns below VaR_α
      - Sharpe = (mean - risk_free) / std * sqrt(252)
    """
    metrics = {}
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) == 0:
            continue
        var = np.percentile(r, 100 * alpha)
        cvar = r[r <= var].mean()
        sharpe = (r.mean() - rf) / r.std() * np.sqrt(252)
        median = np.median(r)
        metrics[col] = {
            "VaR_0.05": var,
            "CVaR_0.05": cvar,
            "Sharpe": sharpe,
            "Median": median
        }
    return pd.DataFrame(metrics).T

# -------------------------
# 📈 Volatility Calculations
# -------------------------
def compute_volatility_table(returns):
    """
    Compute daily/weekly/monthly/quarterly/yearly volatility per asset.
    Formulas:
      - Weekly = Daily × sqrt(5)
      - Monthly = Daily × sqrt(21)
      - Quarterly = Daily × sqrt(63)
      - Yearly = Daily × sqrt(252)
    """
    vol_table = {}
    for col in returns.columns:
        daily = returns[col].std()
        vol_table[col] = {
            "Daily": daily,
            "Weekly": daily * np.sqrt(5),
            "Monthly": daily * np.sqrt(21),
            "Quarterly": daily * np.sqrt(63),
            "Yearly": daily * np.sqrt(252)
        }
    return pd.DataFrame(vol_table).T.round(4)

# -------------------------
# 📉 Portfolio Statistics
# -------------------------
def compute_portfolio_risk(returns, weights):
    """
    Compute portfolio standard deviation, VaR, CVaR, and Median Return.
    """
    cov_matrix = returns.cov()
    portfolio_std = np.sqrt(weights @ cov_matrix @ weights.T)
    port_returns = returns.dot(weights)
    var = np.percentile(port_returns, 5)
    cvar = port_returns[port_returns <= var].mean()
    median = np.median(port_returns)
    return portfolio_std, var, cvar, median

# -------------------------
# 🧠 Benchmark Comparison
# -------------------------
def compute_benchmark_metrics(portfolio_returns, benchmark_returns):
    """
    Compute beta and correlation to benchmark.
    Formulas:
      - Beta = Cov(R_p, R_m) / Var(R_m)
      - Corr = Pearson correlation between R_p and R_m
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var_bm = np.var(aligned.iloc[:, 1])
    beta = cov / var_bm
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return beta, corr

# -------------------------
# ⚖️ Portfolio Optimization
# -------------------------
def optimize_portfolio(returns, method='MV', objective='Sharpe'):
    """
    Optimize portfolio with various risk models:
      - MV: Mean-Variance
      - CVaR: Conditional VaR
      - DaR: Drawdown at Risk
      - EV: Equal Volatility Contribution
    """
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu='hist', method_cov='hist')
    w = port.optimization(model="Classic", rm=method, obj=objective, hist=True)
    return w

# -------------------------
# 💬 Suggestions Engine
# -------------------------
def suggest_portfolio_tweaks(weights, returns, max_adjustment=0.05):
    """
    Suggest portfolio tweaks based on Marginal Risk Contribution (MRC),
    risk decomposition, and complementary metrics (volatility, Sharpe ratio, VaR, beta).

    -------------------
    🧮 Formulas Used:
    -------------------
    - MRC_i = ∂σ_p / ∂w_i = (Σw)_i / σ_p
    - PRC_i = w_i × MRC_i         ← % Risk Contribution
    - Volatility = std(returns) × √252
    - Sharpe = mean / std × √252
    - VaR = Percentile(5%) of returns
    - Beta_i = Cov(R_i, R_port) / Var(R_port)
    """

    cov = returns.cov()
    total_risk = np.sqrt(weights @ cov @ weights.T)
    mrc = cov @ weights / total_risk
    prc = weights * mrc
    prc_pct = prc / prc.sum()

    # Volatility, Sharpe, VaR
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    var = returns.apply(lambda r: np.percentile(r.dropna(), 5))

    # Beta calculation (relative to the portfolio)
    betas = {}
    port_ret = returns.dot(weights)
    for col in returns.columns:
        aligned = pd.concat([returns[col], port_ret], axis=1).dropna()
        cov_ = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        var_p = np.var(aligned.iloc[:, 1])
        betas[col] = cov_ / var_p if var_p > 0 else np.nan

    suggestions = []

    # 1. Shift weight from high PRC to low PRC
    sorted_risk = prc_pct.sort_values(ascending=False)
    top_risk = sorted_risk.head(1).index[0]
    low_risk = sorted_risk.tail(1).index[0]
    suggestions.append(f"🔁 Shift 2% from **{top_risk}** (risk ↑ {prc_pct[top_risk]:.1%}) to **{low_risk}** (risk ↓ {prc_pct[low_risk]:.1%}) to balance PRC.")

    # 2. Volatility imbalance
    high_vol = vol.idxmax()
    low_vol = vol.idxmin()
    suggestions.append(f"📉 Reduce allocation to **{high_vol}** (highest volatility: {vol[high_vol]:.2%}); consider reallocating to **{low_vol}**.")

    # 3. Weak Sharpe ratio
    low_sharpe = sharpe.idxmin()
    suggestions.append(f"⚠️ **{low_sharpe}** has the lowest Sharpe ratio ({sharpe[low_sharpe]:.2f}); reassess its role in the portfolio.")

    # 4. Downside risk (VaR)
    worst_var = var.idxmin()
    suggestions.append(f"🚨 **{worst_var}** has the worst downside risk (VaR: {var[worst_var]:.2%}); consider trimming if conviction is low.")

    # 5. High beta to portfolio
    max_beta = max(betas, key=betas.get)
    suggestions.append(f"📈 **{max_beta}** is most sensitive to market movement (β = {betas[max_beta]:.2f}); reduce if you want to lower market correlation.")

    # 6. Optional: High concentration in top-weighted asset
    if isinstance(weights, pd.Series):
        top_weight_ticker = weights.idxmax()
        top_weight_value = weights.max()
    else:
        # fallback for numpy arrays
        top_weight_ticker = returns.columns[np.argmax(weights)]
        top_weight_value = np.max(weights)

    if top_weight_value > 0.3:
        suggestions.append(f"⚠️ High concentration in **{top_weight_ticker}** ({top_weight_value:.1%}); consider capping exposure.")

    return suggestions[:5]

# -------------------------
# 📈 Longitudinal Risk Table (e.g., 3m, 6m, 1y Volatility Evolution)
# -------------------------
def compute_longitudinal_volatility(returns, windows=[63, 126, 252]):
    """
    Compute rolling volatility over various time windows (3m, 6m, 1y).
    Returns a long-form DataFrame with asset, window, and volatility.
    """
    results = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        for win in windows:
            if len(r) >= win:
                rolling_vol = r.rolling(win).std() * np.sqrt(252)
                latest_vol = rolling_vol.dropna().iloc[-1] if not rolling_vol.dropna().empty else np.nan
                results.append({
                    "Ticker": ticker,
                    "Window": f"{win}d",
                    "Volatility": latest_vol
                })
    return pd.DataFrame(results).round(4)

def compute_concentration_risk(weights):
    """
    Computes concentration risk using the Herfindahl-Hirschman Index (HHI).

    -------------------
    🧮 Formulas Used:
    -------------------
    - HHI = ∑(wᵢ²)
        Where wᵢ is the portfolio weight of asset i.

    - Normalized HHI = (HHI - 1/n) / (1 - 1/n)
        Scales HHI to [0, 1] range where:
            0 = perfect diversification (equal weight)
            1 = maximum concentration (single asset)

    Interpretation:
    - HHI near 1/n → well-diversified
    - HHI near 1   → highly concentrated
    """

    w = np.array(weights)
    hhi = np.sum(w ** 2)
    n = len(w)
    hhi_norm = (hhi - 1/n) / (1 - 1/n) if n > 1 else 1.0
    return hhi, hhi_norm
