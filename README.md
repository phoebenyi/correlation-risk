# 📈 Financial Correlation & Risk Dashboard

An interactive dashboard built with **Streamlit** and **Jupyter Notebook** to analyze cross-asset correlations, assess risk, and optimize portfolios using historical market data.

---

## 🚀 Features

### ✅ Core Capabilities

- **Dynamic Ticker Input** – Add or remove any Yahoo Finance-compatible ticker
- **Date Range Flexibility** – Analyze from 2010 to present or custom ranges
- **Correlation matrix export** – CSV download option
- **Return frequency control** – Daily, Monthly, Yearly
- **Return type toggle** – Percent change or absolute difference
- **Overlap logic** – Choose overlapping vs non-overlapping windows
- **Risk metrics** – Value-at-Risk (VaR), Conditional VaR, Sharpe Ratio (manual or via Riskfolio)
- **Rolling Correlation Viewer** – Select any pair and visualize time-varying correlation trends
- **Correlation Highlights** – Quickly identify top positive, negative, and extreme correlations
- **Portfolio Optimization** – Max Sharpe portfolio optimization with Riskfolio
- **Drawdown & Rebalancing** – See performance over time with monthly rebalance simulation
- **Log, Raw, and Normalized Price Views** – For comparing assets of different magnitudes
- **Ticker Data Availability Audit** – Flags IPOs, delistings, or data gaps

---

## 🧠 Use Cases

- Identify diversification opportunities in multi-asset portfolios
- Explore antifragile asset combinations (low correlation)
- Compare Pearson vs Kendall vs Spearman correlation for different regimes
- Examine how overlapping data affects statistical significance
- Analyze global markets with time zone awareness
- Spot redundancy or diversification gaps in your portfolio
- Compare correlation regimes across frequencies and transformations
- Explore optimal asset combinations using quantitative risk-return metrics
- Investigate data quality for newly listed or delisted stocks

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