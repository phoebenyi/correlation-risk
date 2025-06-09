import pandas as pd


def compute_returns(df, freq="Daily", return_type="% Change (Relative)", overlap_window="No"):
    """
    Computes return matrix based on frequency and return type.
    """
    if freq == "Daily":
        returns = df.pct_change() if return_type == "% Change (Relative)" else df.diff()
        returns = returns.dropna(how="all")

    elif freq == "Monthly":
        monthly_prices = df.ffill().resample("M").last()
        if return_type == "% Change (Relative)":
            returns = monthly_prices.pct_change()
        else:
            returns = monthly_prices.diff()

    elif freq == "Yearly":
        if overlap_window == "Yes":
            if return_type == "% Change (Relative)":
                returns = df.pct_change(252)
            else:
                returns = df.diff(252)
        else:
            yearly_prices = df.ffill().resample("YE").last()
            if return_type == "% Change (Relative)":
                returns = yearly_prices.pct_change()
            else:
                returns = yearly_prices.diff()

    returns.columns = df.columns
    return returns


def normalize_prices(df):
    """
    Normalize price series to start at 100.
    """
    return df.apply(lambda x: x / x.dropna().iloc[0] * 100 if x.dropna().shape[0] > 0 else x)