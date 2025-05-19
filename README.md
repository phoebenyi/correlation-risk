# 📈 Financial Correlation & Risk Dashboard

This project provides a powerful, interactive dashboard — both as a **Streamlit web app** and a **Jupyter notebook** — to analyze historical stock correlations, optimize portfolios, and compute key risk metrics.

---

## 🚀 Features

### ✅ Core Capabilities

- **Dynamic stock selection** – Add/remove tickers on the fly
- **Flexible date range** – Analyze 3M to 20Y of price data
- **Return frequency control** – Daily, Monthly, Yearly
- **Return type toggle** – Percent change or absolute difference
- **Overlap logic** – Choose overlapping vs non-overlapping windows
- **Risk metrics** – Value-at-Risk (VaR), Conditional VaR, Sharpe Ratio
- **Rolling correlations** – Time-varying correlation trends
- **Portfolio optimization** – Max Sharpe portfolio using Riskfolio
- **Drawdown analysis** – Visualize peak-to-trough losses
- **Rebalancing simulation** – Monthly rebalanced return tracking
- **Correlation matrix export** – CSV download option

---

## 🧠 Use Cases

- Identify diversification opportunities in multi-asset portfolios
- Explore antifragile asset combinations (low correlation)
- Compare Pearson vs Kendall vs Spearman correlation for different regimes
- Examine how overlapping data affects statistical significance
- Analyze global markets with time zone awareness

---

## 📦 Requirements

Install the following dependencies:

```bash
pip install -r requirements.txt
```

> Note: You may also need `cvxopt` for full optimization support:
> ```bash
> pip install cvxopt
> ```

---


## 🖥️ Running the App

### ▶️ Streamlit

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

### 📓 Jupyter Notebook

```bash
jupyter notebook correlation_analysis.ipynb
```

Edit variables like `tickers`, `frequency`, `overlap` manually inside the notebook.

---

## ⚠️ Notes

- October returns are automatically excluded due to known options distortion.
- Fewer than 30 data points will trigger a reliability warning.
- `Adj Close` is used when available; falls back to `Close` with adjusted prices.

---

## 🧩 Example Tickers

These are supported by default:
- `ANET` – Arista Networks
- `FN` – Fabrinet
- `ALAB` – Astera Labs
- `NVDA` – NVIDIA

Feel free to input your own!

---

## 🛡️ License

MIT License