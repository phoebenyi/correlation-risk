import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp

# -------------------------
# 🔢 Risk Metric Calculations
# -------------------------
def get_risk_metrics(returns, alpha=0.05, rf=0.0):
    """
    Computes VaR, CVaR, and Sharpe Ratio for each asset.
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
        metrics[col] = {"VaR_0.05": var, "CVaR_0.05": cvar, "Sharpe": sharpe}
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
    Compute portfolio standard deviation, VaR, and CVaR.
    """
    cov_matrix = returns.cov()
    portfolio_std = np.sqrt(weights @ cov_matrix @ weights.T)
    port_returns = returns.dot(weights)
    var = np.percentile(port_returns, 5)
    cvar = port_returns[port_returns <= var].mean()
    return portfolio_std, var, cvar

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
    Suggest tweaks based on Marginal Risk Contribution (MRC).
    Formulas:
      - MRC_i = ∂σ_p / ∂w_i = (Σw)_i / σ_p
      - PRC_i = w_i × MRC_i
    """
    cov = returns.cov()
    total_risk = np.sqrt(weights @ cov @ weights.T)
    mrc = cov @ weights / total_risk
    prc = weights * mrc  # Percentage Risk Contribution
    prc_pct = prc / prc.sum()

    suggestions = []
    for i, ticker in enumerate(weights.index):
        if prc_pct[i] > 0.2:
            suggestions.append(f"⬇️ Decrease {ticker} (high risk contribution: {prc_pct[i]:.1%})")
        elif prc_pct[i] < 0.05:
            suggestions.append(f"⬆️ Increase {ticker} (low risk contribution: {prc_pct[i]:.1%})")
        else:
            suggestions.append(f"✅ Keep {ticker} stable (balanced risk: {prc_pct[i]:.1%})")
    return suggestions
