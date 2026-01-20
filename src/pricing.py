import pandas as pd

def apply_margins(prices: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()

    # Ensure required cols
    if "Product Category" not in df.columns:
        df["Product Category"] = ""

    # Normalise incoming keys
    df["Product Category"] = df["Product Category"].fillna("").astype(str).str.strip().str.lower()
    df["Product"] = df["Product"].fillna("").astype(str).str.strip().str.lower()

    # If no margins, Sell = Price
    if margins is None or margins.empty:
        df["Sell Price"] = pd.to_numeric(df["Price"], errors="coerce")
        return df

    m = margins.copy()
    m["scope_type"] = m["scope_type"].astype(str).str.strip().str.lower()
    m["scope_value"] = m["scope_value"].astype(str).str.strip().str.lower()
    m["margin_per_t"] = pd.to_numeric(m["margin_per_t"], errors="coerce").fillna(0.0)

    cat_map  = m[m["scope_type"] == "category"].set_index("scope_value")["margin_per_t"]
    prod_map = m[m["scope_type"] == "product"].set_index("scope_value")["margin_per_t"]

    df["_margin"] = df["Product Category"].map(cat_map).fillna(0.0)
    prod_m = df["Product"].map(prod_map)
    df.loc[prod_m.notna(), "_margin"] = prod_m[prod_m.notna()]

    df["Sell Price"] = pd.to_numeric(df["Price"], errors="coerce") + pd.to_numeric(df["_margin"], errors="coerce")
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

