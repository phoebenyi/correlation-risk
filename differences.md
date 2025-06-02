
# 📊 Correlation Matrix Comparison: Streamlit vs Excel

This document explains **why correlation values from Streamlit do not match Excel**, even when using the same data, return frequencies, and time periods.

---

## ✅ Executive Summary

- Streamlit and Excel use different underlying assumptions for correlation calculation.
- **Streamlit handles data alignment per pair with precision and safety**.
- **Excel assumes all rows are aligned**, which silently introduces mismatches when returns are missing or date ranges differ.

👉 For accurate, audit-ready results, **use Streamlit** as the source of truth.  
Excel can only approximate it with heavy custom logic or manual alignment.

---

## 📌 The Core Difference: `.dropna()` Behavior in Streamlit

### 🔍 In Streamlit (Python/pandas):
```python
x = returns[i].dropna()
y = returns[j].dropna()
overlap = x.index.intersection(y.index)
corr_val = x[overlap].corr(y[overlap])
```

- Filters only the dates where **both tickers have valid return values**
- Ensures each pair is **aligned by actual overlapping data**
- Robust against missing values due to IPOs, delistings, or market holidays

### ⚠️ In Excel:
```excel
=CORREL(B2:B100, C2:C100)
```

- Assumes row 2 aligns with row 2, row 3 with row 3, etc.
- Ignores actual dates or missing values unless manually filtered
- Includes misaligned or padded rows, which skews correlation results

---

## 🔬 Visual Example

| Row | Date       | ANET Return | FN Return | What `.dropna()` does | What Excel `=CORREL()` does |
|------|------------|-------------|-----------|------------------------|------------------------------|
| 1    | 2020-01-31 | 0.02        | 0.03      | ✅ Keep                | ✅ Use                       |
| 2    | 2020-02-29 | `NaN`       | 0.01      | ❌ Drop                | ⚠️ Included, misaligned      |
| 3    | 2020-03-31 | 0.01        | `NaN`     | ❌ Drop                | ⚠️ Included, misaligned      |
| 4    | 2020-04-30 | 0.015       | 0.02      | ✅ Keep                | ✅ Use                       |

---

## 📊 Why Monthly/Yearly Correlations Are More Affected

- **More gaps** (e.g., IPO in 2023 means no returns from 2010–2022)
- **Weekend/holiday offsets** mess with last-day logic
- Fewer data points = greater impact from misalignment

---

## ✅ Summary Table: Streamlit vs Excel

| Feature                         | Excel         | Streamlit (pandas)       |
|---------------------------------|---------------|---------------------------|
| Date-aware correlation          | ❌ No         | ✅ Yes (per pair)         |
| Handles missing data            | ❌ No         | ✅ `.dropna()` logic      |
| Rolling window logic            | ✅ Supported  | ✅ Supported              |
| Monthly/yearly end alignment    | Approximate   | Exact (resample with ffill) |
| Default correlation method      | Pearson only  | Pearson, Kendall, Spearman |
| Audit trace (helper columns)    | ❌ Manual      | ✅ Built into logic       |

---

## 🛠️ Can Excel Ever Match?

Not with standard formulas. You would need:

- Helper columns per pair
- Date filtering with `=FILTER(...)` or PowerQuery
- Custom `=CORREL(...)` with `MATCH/INDEX` logic
- Or just use Streamlit to export final values

---

## ✅ Recommended

- Use **Streamlit's correlation output as ground truth**
- If Excel output is needed for presentation:
  - Use values exported from Streamlit
  - Or ask for an Excel version with **per-pair aligned helper sheets** (we can generate this)
