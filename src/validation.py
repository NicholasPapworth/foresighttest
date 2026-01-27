import io
import pandas as pd

REQUIRED = ["Supplier", "Product", "Location", "Delivery Window", "Price", "Unit"]
UNIQUE_KEY = ["Supplier", "Product", "Location", "Delivery Window"]

SUPPLIER_SHEET = "SUPPLIER_PRICES"
SEED_SHEET = "SEED_PRICES"


def _load_sheet(content: bytes, sheet_name: str) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(content))
    if sheet_name not in xls.sheet_names:
        raise ValueError(f"Workbook must contain a sheet named '{sheet_name}'. Found: {xls.sheet_names}")

    df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Optional columns
    if "Product Category" not in df.columns:
        df["Product Category"] = ""

    def _clean_str(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.strip()

    for c in ["Supplier", "Product", "Location", "Delivery Window", "Unit", "Product Category", "Notes", "Cost/kg N"]:
        df[c] = _clean_str(df[c])

    # Optional columns (NEW)
    if "Notes" not in df.columns:
        df["Notes"] = ""
    if "Cost/kg N" not in df.columns:
        df["Cost/kg N"] = ""

    # Convert Price to numeric; blanks/non-numeric become NaN (we will drop them)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    
    # Optional Sell Price: if provided, coerce to numeric; if missing, default to Price
    if "Sell Price" in df.columns:
        df["Sell Price"] = pd.to_numeric(df["Sell Price"], errors="coerce")
    else:
        df["Sell Price"] = df["Price"]
    
    # Drop rows where Price is missing / blank / non-numeric
    df = df.dropna(subset=["Price"])
    
    # If Sell Price is NaN for any row, default it to Price (and then drop if still invalid)
    df["Sell Price"] = df["Sell Price"].fillna(df["Price"])
    df = df.dropna(subset=["Sell Price"])


    # Drop blank required fields
    df = df[
        (df["Supplier"] != "") &
        (df["Product"] != "") &
        (df["Location"] != "") &
        (df["Delivery Window"] != "") &
        (df["Unit"] != "")
    ]

    # Duplicate key check
    dup = df.duplicated(subset=UNIQUE_KEY, keep=False)
    if dup.any():
        bad = df.loc[dup, UNIQUE_KEY]
        raise ValueError(
            "Duplicate rows found for key (Supplier+Product+Location+Delivery Window). Fix:\n"
            f"{bad.head(50)}"
        )

    return df[[
        "Supplier", "Product Category", "Product", "Location", "Delivery Window",
        "Price", "Sell Price", "Unit",
        "Notes", "Cost/kg N"
    ]]


def load_supplier_sheet(content: bytes) -> pd.DataFrame:
    return _load_sheet(content, SUPPLIER_SHEET)


def load_seed_sheet(content: bytes) -> pd.DataFrame:
    return _load_sheet(content, SEED_SHEET)


