import pandas as pd

def apply_margins(prices: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()

    if "Product Category" not in df.columns:
        df["Product Category"] = ""

    if margins is None or margins.empty:
        df["Sell Price"] = df["Price"].astype(float)
        return df

    cat = margins[margins["scope_type"] == "category"].set_index("scope_value")["margin_per_t"]
    prod = margins[margins["scope_type"] == "product"].set_index("scope_value")["margin_per_t"]

    df["_margin"] = df["Product Category"].map(cat).fillna(0.0)

    prod_m = df["Product"].map(prod)
    df.loc[prod_m.notna(), "_margin"] = prod_m[prod_m.notna()]

    df["Sell Price"] = df["Price"].astype(float) + df["_margin"].astype(float)
    df = df.drop(columns=["_margin"], errors="ignore")
    return df
