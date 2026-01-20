import pandas as pd

def apply_margins(prices: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()

    # Accept either column name
    if "Price" not in df.columns and "Base Price" in df.columns:
        df["Price"] = df["Base Price"]

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
    return df.drop(columns=["_margin"], errors="ignore")

    # Build lookup series
    cat = margins[margins["scope_type"] == "category"].set_index("scope_value")["margin_per_t"]
    prod = margins[margins["scope_type"] == "product"].set_index("scope_value")["margin_per_t"]

    # Default margin from category
    df["_margin"] = df["Product Category"].map(cat).fillna(0.0)

    # Product margin overrides category
    prod_m = df["Product"].map(prod)
    df.loc[prod_m.notna(), "_margin"] = prod_m[prod_m.notna()]

    # Always recompute Sell Price from base + margin
    df["Sell Price"] = df[base_col].astype(float) + df["_margin"].astype(float)

    df = df.drop(columns=["_margin"], errors="ignore")
    return df

