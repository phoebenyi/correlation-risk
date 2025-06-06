# 📈 Financial Correlation & Risk Dashboard

An interactive dashboard built with **Streamlit** to analyze cross-asset correlations, assess risk, and optimize portfolios using historical market data. It is powered by **Supabase** for user authentication and persistent group management, and yfinance for live financial data analysis.

---

## 🌍 Live Demo
[https://correlation-risk.streamlit.app/](https://correlation-risk.streamlit.app/) — *hosted on Streamlit Cloud*
> ⚠️ Data is pulled from Yahoo Finance (yfinance) for live analysis
  
---

## 🚀 Features

### ✅ Core Capabilities

- **User-Specific Private Group** - Each user can save and manage named stock groups (e.g., "Tech Stocks", "China Exposure") tied to their account which will not be accessible to the public
- **Dynamic Ticker Input** – Analyse any stocks (Yahoo Finance-compatible ticker)
- **Flexible Date Range Control** – Analyze as far back as 2010 or narrow to recent periods
- **Customizable Return Calculations** – Compare raw price changes vs percent changes, daily/monthly/yearly
- **Overlap Handling** – Choose between overlapping and non-overlapping return windows for YoY analysis
- **Rolling Correlation** – Analyze how correlation between any stock pair changes over time
- **Correlation Matrix Analysis** – Pearson, Kendall, Spearman options with one-click CSV download
- **Top Correlation Highlights** – Automatically detect top positive, negative, and strongest correlation pairs
- **Risk Metrics** – Compute VaR, CVaR, and Sharpe ratios per asset (with Riskfolio)
- **Portfolio Optimization** – Construct a max Sharpe ratio portfolio and view allocation weights
- **Rebalancing Simulation** – Visualize cumulative return, drawdown, and monthly rebalanced performance
- **Secure Login & Logout Flow** – Seamlessly re-authenticate via session tokens

---

## 🧠 Use Cases

- Easily organize private groups of tickers for repeated analysis
- Construct and analyze diversified, low-correlation investment portfolios
- Identify highly correlated assets to reduce redundancy and improve risk-adjusted returns
- Compare market regimes using different correlation methods (Pearson, Kendall, Spearman)
- Test impact of overlapping vs non-overlapping return windows on statistical accuracy
- Evaluate risk across assets with standardized metrics like VaR, CVaR, and Sharpe
- Optimize asset allocations to maximize Sharpe ratio using historical returns
- Track how relationships between assets evolve over time via rolling correlation
- Assess data sufficiency and reliability for newly listed or delisted tickers
- Benchmark performance against monthly rebalanced portfolio simulations

---

## 📦 Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

If you encounter solver issues with Riskfolio optimization, also install:

```bash
pip install cvxopt
```

---

## ▶️ Running the App

### 🖥 Streamlit Web App

```bash
streamlit run app.py
```

Then visit `http://localhost:8501`
> ⚠️ Make sure .streamlit/secrets.toml includes your Supabase project config
```bash
[supabase]
url = "https://your-project-id.supabase.co"
key = "your-anon-key"
```

---

### 📝 Excel Backtesting
All correlation matrix for `absolute/relative` and `daily/monthly/yearly` calculated on Streamlit app.py **matches** Excel's backtesting.

---

### 📓 Jupyter Notebook

> ⚠️ The Jupyter notebook is not updated as it is only used for experimental purposes. Refer to Streamlit app.py for the most up to date code logic and features.

Open:

```bash
jupyter notebook correlation_analysis.ipynb
```

Manually adjust variables for analysis and code for testing.

---

## ⚠️ Copyright

This project is publicly viewable for educational and demonstration purposes.  
**All rights are reserved.** Please do not reproduce, copy, or redistribute without permission from author (Phoebe Neo).
