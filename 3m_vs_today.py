# This is NOT in app.py because it is not working yet.
# st.subheader("🕰️ Portfolio Comparison: Today vs Previous Period")

#             # Get weights: from uploaded CSV or fallback to editable UI weights
#             weights_source = portfolio_weights

#             # fallback to UI-edited weights if available
#             if weights_source is None and "editable_weights" in st.session_state:
#                 df_ui = st.session_state["editable_weights"]
#                 if isinstance(df_ui, pd.DataFrame) and {"Ticker", "Weight"}.issubset(df_ui.columns):
#                     weights_source = pd.Series(df_ui["Weight"].values, index=df_ui["Ticker"]).dropna()

#             # fallback to equal weight if still None and tickers exist
#             if weights_source is None and tickers and len(tickers) >= 2:
#                 equal_w = pd.Series([1/len(tickers)] * len(tickers), index=tickers)
#                 weights_source = equal_w
#                 if "editable_weights" not in st.session_state:
#                     st.session_state["editable_weights"] = pd.DataFrame({
#                         "Ticker": equal_w.index,
#                         "Weight": equal_w.values
#                     })

#             if weights_source is not None:
#                 def get_portfolio_stats(ret_data, weights, min_days):
#                     ret_data = ret_data[weights.index]
#                     port_ret = ret_data.dot(weights)
#                     if port_ret.shape[0] < min_days:
#                         return None
#                     port_ret = port_ret.dropna()
#                     cumret = (1 + port_ret).cumprod()
#                     return {
#                         "Cumulative Return": cumret.iloc[-1] if not cumret.empty else np.nan,
#                         "Std Dev": port_ret.std() * np.sqrt(252),
#                         "VaR (95%)": np.percentile(port_ret, 5),
#                         "CVaR (95%)": port_ret[port_ret <= np.percentile(port_ret, 5)].mean(),
#                         "Sharpe": port_ret.mean() / port_ret.std() * np.sqrt(252),
#                         "Median Return": np.median(port_ret)
#                     }

#                 def get_portfolio_series(ret_data, weights, min_days):
#                     ret_data = ret_data[weights.index]
#                     st.write("🔍 get_portfolio_series ret_data shape:", ret_data.shape)
#                     st.write("🔍 get_portfolio_series weights index:", weights.index.tolist())

#                     port_ret = ret_data.dot(weights)
#                     if port_ret.dropna().shape[0] < min_days:
#                         return None
#                     cumret = (1 + port_ret).cumprod()
#                     return cumret

#                 try:
#                     comparison_window = st.selectbox("Comparison Window", ["1 Month", "3 Months", "6 Months", "1 Year"])
#                     window_map = {
#                         "1 Month": 21,
#                         "3 Months": 63,
#                         "6 Months": 126,
#                         "1 Year": 252
#                     }
#                     lookback_days = window_map[comparison_window]
#                     cutoff_date = pd.to_datetime(end) - pd.DateOffset(days=lookback_days)
#                     min_required_days = st.slider("Minimum days of return data required", 1, 30, 5)
#                     returns.index = pd.to_datetime(returns.index)
#                     common_tickers = list(set(weights_source.index).intersection(returns.columns))
#                     aligned_weights = weights_source.loc[common_tickers]
#                     if aligned_weights.sum() == 0:
#                         st.warning("⚠️ Sum of weights is zero. Skipping comparison.")
#                         return
#                     aligned_weights = aligned_weights / aligned_weights.sum()
#                     returns = returns[common_tickers]

#                     if len(aligned_weights) < 1:
#                         st.warning("⚠️ No overlapping tickers between weights and return data.")
#                     else:
#                         recent_returns = returns[returns.index >= cutoff_date]
#                         old_returns = returns[returns.index < cutoff_date]

#                         if recent_returns.shape[0] < min_required_days:
#                             st.warning("📉 Not enough recent return data after cutoff.")
#                         if old_returns.shape[0] < min_required_days:
#                             st.warning("📉 Not enough historical return data before cutoff.")
#                         if recent_returns.shape[0] < min_required_days or old_returns.shape[0] < min_required_days:
#                             return

#                         st.write("📆 returns.index min/max:", returns.index.min(), returns.index.max())
#                         st.write("📆 cutoff_date:", cutoff_date)
#                         st.write("🧮 aligned_weights index:", aligned_weights.index.tolist())
#                         st.write("🧮 returns columns:", returns.columns.tolist())
#                         st.write("📊 recent_returns shape:", recent_returns.shape)
#                         st.write("📊 old_returns shape:", old_returns.shape)

#                         today_stats = get_portfolio_stats(recent_returns, aligned_weights, min_required_days)
#                         past_stats = get_portfolio_stats(old_returns, aligned_weights, min_required_days)
#                         recent_cum = get_portfolio_series(recent_returns, aligned_weights, min_required_days)
#                         old_cum = get_portfolio_series(old_returns, aligned_weights, min_required_days)

#                         if today_stats and past_stats:
#                             stats_df = pd.DataFrame([past_stats, today_stats], index=[f"{comparison_window} Ago", "Today"]).T
#                             st.dataframe(stats_df.round(4))

#                             if recent_cum is not None and old_cum is not None:

#                                 if recent_cum.index.tz is not None:
#                                     recent_cum.index = recent_cum.index.tz_localize(None)
#                                 if old_cum.index.tz is not None:
#                                     old_cum.index = old_cum.index.tz_localize(None)

#                                 recent_cum.index = pd.to_datetime(recent_cum.index).normalize()
#                                 old_cum.index = pd.to_datetime(old_cum.index).normalize()

#                                 common_idx = recent_cum.index.intersection(old_cum.index).sort_values()

#                                 st.write("📅 recent_cum index sample:", recent_cum.index[:3])
#                                 st.write("📅 old_cum index sample:", old_cum.index[:3])
#                                 st.write("📅 Common Index Length:", len(common_idx))
#                                 st.write("📅 Common Dates:", common_idx[:5])

#                                 # Reindex both series to that superset and forward-fill
#                                 aligned_recent = recent_cum.reindex(common_idx).ffill()
#                                 aligned_old = old_cum.reindex(common_idx).ffill()

#                                 st.write("🔍 aligned_recent shape:", aligned_recent.shape)
#                                 st.write("🔍 aligned_old shape:", aligned_old.shape)
#                                 st.write("🔍 aligned_recent index sample:", aligned_recent.index[:3])
#                                 st.write("🔍 aligned_old index sample:", aligned_old.index[:3])

#                                 # Drop rows where either one is still missing
#                                 chart_df = pd.DataFrame({
#                                     "Today": aligned_recent,
#                                     f"{comparison_window} Ago": aligned_old
#                                 }).dropna()

#                                 if chart_df.shape[0] >= 2:
#                                     chart_df = chart_df / chart_df.iloc[0] * 100
#                                     st.line_chart(chart_df)
#                                 else:
#                                     st.warning("📉 Not enough overlapping data after reindexing.")
#                                     st.dataframe(chart_df)
#                         else:
#                             st.warning("📉 Not enough return data to build comparison.")
#                 except Exception as e:
#                     st.error(f"❌ Failed to compute portfolio comparison: {e}")
#             else:
#                 st.info("ℹ️ Upload a portfolio or edit weights above to see historical comparison.")