import pandas as pd

def apply_margins(prices: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()

    # Accept either "Price" or "Base Price" as the base price column
    if "Price" in df.columns:
        base_col = "Price"
    elif "Base Price" in df.columns:
        base_col = "Base Price"
        df["Price"] = df["Base Price"]  # normalise for downstream code
    else:
        raise KeyError("apply_margins() requires a 'Price' or 'Base Price' column.")

    if "Product Category" not in df.columns:
        df["Product Category"] = ""

    if margins is None or margins.empty:
        df["Sell Price"] = pd.to_numeric(df[base_col], errors="coerce").astype(float)
        return df

    cat = margins[margins["scope_type"] == "category"].set_index("scope_value")["margin_per_t"]
    prod = margins[margins["scope_type"] == "product"].set_index("scope_value")["margin_per_t"]

    df["_margin"] = df["Product Category"].map(cat).fillna(0.0)

    prod_m = df["Product"].map(prod)
    df.loc[prod_m.notna(), "_margin"] = prod_m[prod_m.notna()]

    df["Sell Price"] = pd.to_numeric(df[base_col], errors="coerce").astype(float) + df["_margin"].astype(float)
    df = df.drop(columns=["_margin"], errors="ignore")
    return df

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

