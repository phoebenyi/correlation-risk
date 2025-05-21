# 📈 Financial Correlation & Risk Dashboard

An interactive dashboard built with **Streamlit** and **Jupyter Notebook** to analyze cross-asset correlations, assess risk, and optimize portfolios using historical market data.

---

## 🌍 Live Demo
[https://correlation-risk.streamlit.app/](https://correlation-risk.streamlit.app/) — *hosted on Streamlit Cloud*
> ⚠️ Demo is public — no authentication, so please use sample or non-sensitive data only.

---

## 🚀 Features

### ✅ Core Capabilities

- **Dynamic Ticker Input** – Instantly add or remove any Yahoo Finance-compatible ticker for live analysis
- **Flexible Date Range Control** – Analyze as far back as 2010 or narrow to recent periods
- **Customizable Return Calculations** – Switch between daily, monthly, or yearly data; view raw price differences or percent returns
- **Overlap Handling** – Choose between overlapping and non-overlapping return windows for YoY analysis
- **Interactive Price Visualization** – View raw prices, normalized growth (rebased to 100), or log-scaled charts
- **Rolling Correlation Viewer** – Analyze how correlations between selected pairs evolve over time
- **Correlation Matrix Analysis** – Exportable heatmaps and dataframes of asset correlations with multiple methods (Pearson, Kendall, Spearman)
- **Automated Correlation Highlights** – Automatically detect and summarize top positive, negative, and strong correlations
- **Risk Metrics Dashboard** – Compute VaR, CVaR, and Sharpe ratios per asset (manually or via Riskfolio)
- **Portfolio Optimization** – Construct a max Sharpe ratio portfolio and view allocation weights
- **Performance Tracking** – Visualize cumulative return, drawdown, and monthly rebalanced performance
- **Ticker Data Integrity Audit** – Identify tickers with missing data due to IPOs, delistings, or Yahoo limitations
- **Export Tools** – One-click download of correlation matrix as CSV for further analysis

---

## 🧠 Use Cases

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

---

### 📓 Jupyter Notebook

Open:

```bash
jupyter notebook correlation_analysis.ipynb
```

And manually adjust variables for analysis.

---

## ⚠️ Notes

- **Partial Data Notice**: Stocks with data outside the selected window (e.g. IPOs) will be flagged with a reason and available date range.
- **Minimum Data Threshold**: If fewer than 30 return points, a warning is shown.

---

## 💡 Example Stocks

Default:
- `ANET` – Arista Networks
- `FN` – Fabrinet
- `ALAB` – Astera Labs (IPO 2024)
- `NVDA` – NVIDIA
Feel free to input your own!

---

## 📜 License

MIT License
