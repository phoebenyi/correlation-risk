# 📈 Financial Correlation & Risk Dashboard

An interactive dashboard built with **Streamlit** to analyze cross-asset correlations, assess risk, and optimize portfolios using historical market data.

---

## 🌍 Live Demo
[https://correlation-risk.streamlit.app/](https://correlation-risk.streamlit.app/) — *hosted on Streamlit Cloud*
> ⚠️ Data is pulled from Yahoo Finance (yfinance) for live analysis
  
---

## 🚀 Features

### ✅ Core Capabilities

- **Dynamic Ticker Input** – Instantly add or remove any stocks (Yahoo Finance-compatible ticker) for live analysis
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

## ⚙️ Sample Configurations
<p align="center">
  <img width="200" alt="config1" src="https://github.com/user-attachments/assets/9f9f9710-0b83-45fb-8e00-86c9543f57df">
  <img width="200" alt="config2" src="https://github.com/user-attachments/assets/08f3da1c-bf3b-437b-b494-d8b77568eb97">
</p>

## 💹 Sample Workflow
<p align="center">
  <img width="800" alt="price" src="https://github.com/user-attachments/assets/d8491227-b52f-4dce-b946-4f673c4b02c3">
  <img width="800" alt="raw" src="https://github.com/user-attachments/assets/4647fbe9-7aaa-4426-b813-faa7379084a3">
  <img width="800" alt="normalised" src="https://github.com/user-attachments/assets/eb81d558-c2ca-4029-be28-3ff714769ff5">
  <img width="800" alt="log" src="https://github.com/user-attachments/assets/29ef151f-b390-49ae-aa70-1f2881afbe90">
  <img width="500" alt="logchart" src="https://github.com/user-attachments/assets/39004c4b-d410-4705-810b-53d6cd249218">
  <img width="800" alt="matrix" src="https://github.com/user-attachments/assets/9b42fd9b-0bdc-4a18-bd0c-7f35bcf98f6b">
  <img width="800" alt="heatmap" src="https://github.com/user-attachments/assets/eee5ca49-3260-4feb-b7f6-289e64ecc908">
  <img width="800" alt="top5" src="https://github.com/user-attachments/assets/3594b598-9bdb-4be8-b97c-1578f3c2c1e1">
  <img width="800" alt="pairs" src="https://github.com/user-attachments/assets/e8fda63d-82c2-40d1-bcf4-ba21cfbc0b87">
  <img width="800" alt="rolling" src="https://github.com/user-attachments/assets/e5d96a16-798d-4aae-88ab-ea7c95a94388">
  <img width="800" alt="metrics" src="https://github.com/user-attachments/assets/1fab0553-8b49-4ebb-9eeb-33e2612d1a24">
  <img width="800" alt="MaxSharpe" src="https://github.com/user-attachments/assets/98e96abc-cca0-467b-9572-7a7dc795952a">
  <img width="800" alt="cumulative" src="https://github.com/user-attachments/assets/53df4fb4-ec5f-4e65-afa9-8c866505acef">
  <img width="800" alt="drawdown" src="https://github.com/user-attachments/assets/6c54c448-46c4-47aa-8a78-f277f8be151f">
  <img width="800" alt="rebalanced" src="https://github.com/user-attachments/assets/67fda1e6-beb0-49e8-913f-510a58facf23">
  <img width="800" alt="complete" src="https://github.com/user-attachments/assets/79f245b6-54c0-4c45-a03e-44ed9ba788e9">
</p>

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

## 💡 Example Stocks

Default:
- `ANET` – Arista Networks
- `FN` – Fabrinet
- `ALAB` – Astera Labs
- `NVDA` – NVIDIA

Feel free to input your own!

---

## 📜 License

MIT License
