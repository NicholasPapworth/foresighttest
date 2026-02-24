import time
import streamlit as st
import pandas as pd
import altair as alt
import uuid
import base64
import streamlit.components.v1 as components
from pathlib import Path
from datetime import datetime, timezone, timedelta
from src.db import list_online_users

from src.db import (
    get_settings, set_setting,
    get_small_lot_tiers, save_small_lot_tiers,

    latest_supplier_snapshot, list_supplier_snapshots,
    load_supplier_prices, publish_supplier_snapshot,

    latest_seed_snapshot, list_seed_snapshots,
    load_seed_prices, publish_seed_snapshot, list_seed_treatments, save_seed_treatments,

    list_fert_delivery_options, save_fert_delivery_options,

    add_margin, list_margins, deactivate_margin, get_effective_margins,
    create_order_from_allocation, list_orders_for_user, list_orders_admin,
    get_order_header, get_order_lines, get_order_actions,
    trader_cancel_order, trader_accept_counter,
    admin_counter_order, admin_confirm_order, admin_reject_order, admin_mark_filled,
    admin_blotter_lines,
    admin_margin_report,

    list_active_offers, list_offers, create_offer, deactivate_offer,
)

from src.db import (
    list_distinct_presence_users,
    list_distinct_presence_pages,
    list_presence_events,
    presence_daily_logins,
    presence_daily_page_transitions,
    presence_session_summary,
)

from src.db import (
    list_stock_stores, save_stock_stores,
    list_stock_store_products, save_stock_store_products,
    get_haulage_settings, set_haulage_settings,
    list_haulage_bands, save_haulage_bands,
)

from src import routing

from src.validation import load_supplier_sheet, load_seed_sheet
from src.optimizer import optimise_basket
from src.pricing import apply_margins

LOGO_PATH = "assets/logo.svg"
STOCK_SUPPLIER_NAME = "STOCK"

def show_boot_splash(video_path: str | None = None, seconds: float = 4.8):
    """
    Full-screen splash ONCE per session.
    Autoplay, no controls, NOT looping, full screen for `seconds`, then disappears.
    """
    if st.session_state.get("booted", False):
        return
    if st.session_state.get("_booting", False):
        st.stop()

    st.session_state["_booting"] = True

    if not video_path:
        st.session_state["booted"] = True
        st.session_state["_booting"] = False
        return

    p = Path(video_path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / video_path
    if not p.exists():
        st.session_state["booted"] = True
        st.session_state["_booting"] = False
        st.error(f"Splash video not found: {p}")
        return

    # Cache base64 once (prevents blank delay on reruns)
    if "boot_b64" not in st.session_state:
        st.session_state["boot_b64"] = base64.b64encode(p.read_bytes()).decode("utf-8")

    b64 = st.session_state["boot_b64"]

    st.markdown(
        f"""
        <style>
          header, footer {{ visibility: hidden; }}
          [data-testid="stSidebar"] {{ display: none; }}
          .block-container {{ padding-top: 0rem; }}

          #boot-splash {{
            position: fixed;
            inset: 0;
            z-index: 2147483647;
            background: #000;
            overflow: hidden;
          }}
          #boot-splash video {{
            position: absolute;
            inset: 0;
            width: 100vw;
            height: 100vh;
            object-fit: cover;
          }}
        </style>

        <div id="boot-splash">
          <video id="bootVid"
                 autoplay
                 muted
                 playsinline
                 preload="auto">
            <source src="data:video/mp4;base64,{b64}" type="video/mp4" />
          </video>
        </div>

        <script>
          (function() {{
            const v = document.getElementById("bootVid");
            if (!v) return;
            v.controls = false;
            const tryPlay = () => {{
              try {{
                const p = v.play();
                if (p && p.catch) p.catch(() => {{}});
              }} catch(e) {{}}
            }};
            v.addEventListener("canplay", tryPlay);
            v.addEventListener("loadeddata", tryPlay);
            tryPlay();
          }})();
        </script>
        """,
        unsafe_allow_html=True
    )

    # Hold the server run so the DOM isn't rebuilt (prevents "looping" via reruns)
    time.sleep(seconds)

    st.session_state["booted"] = True
    st.session_state["_booting"] = False
    st.session_state.pop("boot_b64", None)
    st.rerun()

# ---------------------------
# Product books (Fertiliser vs Seed)
# ---------------------------

BOOKS = {
    "Fertiliser": {
        "code": "fert",
        "latest_snapshot": latest_supplier_snapshot,
        "list_snapshots": list_supplier_snapshots,
        "load_prices": load_supplier_prices,
        "publish_snapshot": publish_supplier_snapshot,
        "loader": load_supplier_sheet,
        "upload_label": "Upload fertiliser prices (SUPPLIER_PRICES)",
        "publish_button": "Publish fertiliser snapshot",
    },
    "Seed": {
        "code": "seed",
        "latest_snapshot": latest_seed_snapshot,
        "list_snapshots": list_seed_snapshots,
        "load_prices": load_seed_prices,
        "publish_snapshot": publish_seed_snapshot,
        "loader": load_seed_sheet,
        "upload_label": "Upload seed prices (SEED_PRICES)",
        "publish_button": "Publish seed snapshot",
    },
}

BOOKS_BY_CODE = {v["code"]: v for v in BOOKS.values()}

def _pick_book(page_key: str, default: str = "fert") -> str:
    """
    Returns 'fert' or 'seed' based on a single UI selector.
    This avoids st.tabs() executing BOTH branches and corrupting presence_context.
    """
    labels = ["Fertiliser", "Seed"]
    default_idx = 0 if default == "fert" else 1

    choice = st.selectbox(
        "Book",
        options=labels,
        index=default_idx,
        key=f"{page_key}__book",
    )
    return "fert" if choice == "Fertiliser" else "seed"


def _ss_key(book_code: str, name: str) -> str:
    """Namespaced session_state key so Fert and Seed don't clash."""
    return f"{book_code}__{name}"

def _ensure_upload_has_sell_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures upload DF contains a numeric Sell Price column.
    Expects upload DF shape from validation loaders:
      Supplier, Product Category, Product, Location, Delivery Window, Price, Unit
    """
    work = df.copy()

    if "Price" not in work.columns:
        raise ValueError("Upload sheet must contain a 'Price' column.")

    work["Price"] = pd.to_numeric(work["Price"], errors="coerce")
    work = work.dropna(subset=["Price"])

    if "Sell Price" not in work.columns:
        work["Sell Price"] = work["Price"]

    work["Sell Price"] = pd.to_numeric(work["Sell Price"], errors="coerce")
    work = work.dropna(subset=["Sell Price"])

    return work


def _get_latest_prices_df_for(book_code: str):
    snap = BOOKS_BY_CODE[book_code]["latest_snapshot"]()
    if not snap:
        return None, None
    sid, ts, by = snap
    df = BOOKS_BY_CODE[book_code]["load_prices"](sid)
    return sid, df


def _ensure_basket_for(book_code: str):
    bkey = _ss_key(book_code, "basket")
    tkey = _ss_key(book_code, "basket_created_at")
    if bkey not in st.session_state:
        st.session_state[bkey] = []
        st.session_state[tkey] = time.time()

def render_header():
    left, mid, right = st.columns([2, 5, 3], vertical_alignment="center")
    with left:
        try:
            st.image(LOGO_PATH, width=170)
        except Exception:
            st.caption("Logo missing.")
    with mid:
        st.markdown("## Foresight Pricing")
    with right:
        snap_f = latest_supplier_snapshot()
        snap_s = latest_seed_snapshot()
        
        lines = []
        if snap_f:
            _, ts, by = snap_f
            lines.append(f"Fertiliser: {ts} UTC (by {by})")
        else:
            lines.append("Fertiliser: none")
        
        if snap_s:
            _, ts, by = snap_s
            lines.append(f"Seed: {ts} UTC (by {by})")
        else:
            lines.append("Seed: none")
        
        st.markdown("**Latest snapshots:**  \n" + "  \n".join(lines))

    st.divider()

def _inject_offer_css():
    st.markdown(
        """<style>
.offer-red { color: #d11 !important; font-weight: 700 !important; }
</style>""",
        unsafe_allow_html=True,
    )

def _norm_postcode(pc: str) -> str:
    return str(pc or "").strip().upper().replace(" ", "")

def _norm_product(p: str) -> str:
    # normalise product keys for matching across snapshots + stock config
    return str(p or "").strip().lower()

def _best_prices_board(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns best (lowest) SELL price per:
      Product Category + Product + Location + Delivery Window

    We ignore supplier in the grouping (we pick the cheapest supplier row per group).
    Expects df to already include Sell Price (from apply_margins).
    """
    work = df.copy()

    required = ["Product Category", "Product", "Location", "Delivery Window", "Supplier", "Sell Price", "Unit"]
    for col in required:
        if col not in work.columns:
            raise ValueError(f"Missing column '{col}' needed for best price board.")

    work["Product Category"] = work["Product Category"].fillna("").astype(str)
    work["Product"] = work["Product"].fillna("").astype(str)
    work["Location"] = work["Location"].fillna("").astype(str)
    work["Delivery Window"] = work["Delivery Window"].fillna("").astype(str)
    work["Supplier"] = work["Supplier"].fillna("").astype(str)
    work["Unit"] = work["Unit"].fillna("").astype(str)
    work["Sell Price"] = pd.to_numeric(work["Sell Price"], errors="raise")

    group_cols = ["Product Category", "Product", "Location", "Delivery Window"]
    idx = work.groupby(group_cols)["Sell Price"].idxmin()
    best = work.loc[idx, group_cols + ["Sell Price", "Unit", "Supplier"]].copy()

    best = best.rename(columns={"Sell Price": "Best Price"})
    best = best.sort_values(["Product Category", "Product", "Location", "Delivery Window"]).reset_index(drop=True)
    return best

def _apply_offers_to_prices_df(book_code: str, df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - Effective Sell Price (used for optimisation + board)
      - Offer Active (bool)
      - Offer Ends (iso str)
      - Offer Title
    Expects df has: Supplier, Product, Location, Delivery Window, Sell Price
    """
    offers = list_active_offers(book_code)
    out = df_prices.copy()

    out["Offer Active"] = False
    out["Offer Ends"] = ""
    out["Offer Title"] = ""
    out["Effective Sell Price"] = pd.to_numeric(out["Sell Price"], errors="coerce").fillna(0.0)

    if offers is None or offers.empty:
        return out

    # Normalise keys
    offers = offers.copy()
    offers["product"] = offers["product"].fillna("").astype(str)
    offers["location"] = offers["location"].fillna("").astype(str)
    offers["delivery_window"] = offers["delivery_window"].fillna("").astype(str)
    offers["supplier"] = offers["supplier"].fillna("").astype(str)

    out["Supplier"] = out["Supplier"].fillna("").astype(str)
    out["Product"] = out["Product"].fillna("").astype(str)
    out["Location"] = out["Location"].fillna("").astype(str)
    out["Delivery Window"] = out["Delivery Window"].fillna("").astype(str)

    for i, r in out.iterrows():
        prod = r["Product"]
        loc = r["Location"]
        win = r["Delivery Window"]
        sup = r["Supplier"]

        cand = offers[
            (offers["product"] == prod) &
            (offers["location"] == loc) &
            (offers["delivery_window"] == win) &
            ((offers["supplier"] == "") | (offers["supplier"] == sup))
        ]
        if cand.empty:
            continue

        # If multiple match: soonest expiry wins
        cand = cand.sort_values("ends_at_utc", ascending=True).iloc[0]

        base = float(out.at[i, "Effective Sell Price"])
        mode = str(cand["mode"])
        val = float(cand["value"])

        if mode == "delta":
            eff = max(0.0, base - val)
        else:  # fixed
            eff = max(0.0, val)

        out.at[i, "Effective Sell Price"] = eff
        out.at[i, "Offer Active"] = True
        out.at[i, "Offer Ends"] = str(cand["ends_at_utc"])
        out.at[i, "Offer Title"] = str(cand.get("title", "") or "")

    return out

def _apply_role_margins(df: pd.DataFrame, role_code: str) -> pd.DataFrame:
    """
    Compute Sell Price from base Price using role-specific margins.
    Traders/wholesalers never need to see base Price, but we can still compute from it.
    """
    out = df.copy()

    if "Price" not in out.columns:
        # fallback: if snapshot somehow lacks Price, keep existing Sell Price as-is
        if "Sell Price" not in out.columns:
            out["Sell Price"] = 0.0
        return out

    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out["Sell Price"] = out["Price"]  # always recompute from base for the role

    margins = get_effective_margins(role_code=role_code)
    out = apply_margins(out, margins)
    return out

def _inject_stock_rows(df_prices: pd.DataFrame, active_stock_products: set[str]) -> pd.DataFrame:
    """
    Create ONE pseudo-supplier STOCK row per (Product, Location, Delivery Window)
    for any products configured as active in Admin | Stock.

    Baseline "Price" for STOCK rows is the cheapest available Effective Sell Price
    among real suppliers for that same (Product, Location, Window). Haulage is added later.
    """
    if df_prices is None or df_prices.empty:
        return df_prices
    if not active_stock_products:
        return df_prices

    df = df_prices.copy()

    # Must have these columns to be meaningful
    needed = ["Product", "Location", "Delivery Window", "Supplier"]
    for c in needed:
        if c not in df.columns:
            return df_prices  # fail safe: don't inject stock if schema unexpected

    df["Product"] = df["Product"].fillna("").astype(str)
    df["Location"] = df["Location"].fillna("").astype(str)
    df["Delivery Window"] = df["Delivery Window"].fillna("").astype(str)
    df["Supplier"] = df["Supplier"].fillna("").astype(str)

    # Use Effective Sell Price if present, else Sell Price, else Price
    price_source = None
    for cand in ["Effective Sell Price", "Sell Price", "Price"]:
        if cand in df.columns:
            price_source = cand
            break
    if price_source is None:
        return df_prices

    df[price_source] = pd.to_numeric(df[price_source], errors="coerce").fillna(0.0)

    # match using normalised product names
    df["_norm_product"] = df["Product"].map(_norm_product)
    stock_base = df[df["_norm_product"].isin({ _norm_product(x) for x in active_stock_products })].copy()
    if stock_base.empty:
        return df_prices

    group_cols = ["Product", "Location", "Delivery Window"]

    # Pick the single cheapest row per group as baseline for STOCK
    idx = stock_base.groupby(group_cols)[price_source].idxmin()
    stock_rows = stock_base.loc[idx].copy()

    # Make it a STOCK supplier row
    stock_rows["Supplier"] = STOCK_SUPPLIER_NAME

    # Optimiser uses column "Price"
    stock_rows["Price"] = pd.to_numeric(stock_rows[price_source], errors="coerce").fillna(0.0)

    # Keep these for downstream display / consistency
    if "Sell Price" in stock_rows.columns:
        stock_rows["Sell Price"] = stock_rows["Price"]
    if "Effective Sell Price" in stock_rows.columns:
        stock_rows["Effective Sell Price"] = stock_rows["Price"]

    # Ensure required cols exist
    for c in ["Unit", "Product Category"]:
        if c not in stock_rows.columns:
            stock_rows[c] = ""

    # IMPORTANT: drop any existing STOCK rows first to avoid duplicates on reruns
    df_no_stock = df_prices[df_prices["Supplier"].astype(str) != STOCK_SUPPLIER_NAME].copy()

    out = pd.concat([df_no_stock, stock_rows], ignore_index=True)

    if "_norm_product" in out.columns:
        out = out.drop(columns=["_norm_product"], errors="ignore")

    return out


def _haulage_per_t(miles: float, bands: pd.DataFrame, break_miles: float, per_mile_per_t: float) -> float:
    """
    Two-tier haulage:
      - if miles <= break_miles -> use band charge (£/t) from bands table
      - if miles > break_miles -> band charge at break + (miles-break)*per_mile_per_t
    """
    m = float(miles or 0.0)

    # Normalise bands
    b = bands.copy() if bands is not None else pd.DataFrame()
    if not b.empty:
        for col in ["min_miles", "max_miles", "charge_per_t"]:
            if col not in b.columns:
                b[col] = None
        b["min_miles"] = pd.to_numeric(b["min_miles"], errors="coerce").fillna(0.0)
        b["max_miles"] = pd.to_numeric(b["max_miles"], errors="coerce")
        b["charge_per_t"] = pd.to_numeric(b["charge_per_t"], errors="coerce").fillna(0.0)
        if "active" in b.columns:
            b = b[b["active"].astype(int) == 1]
        b = b.sort_values(["min_miles", "max_miles"], ascending=True)

    brk = float(break_miles or 0.0)
    per = float(per_mile_per_t or 0.0)

    def band_charge(x: float) -> float:
        if b.empty:
            return 0.0
        hit = b[(b["min_miles"] <= x) & ((b["max_miles"].isna()) | (x < b["max_miles"]))].head(1)
        if hit.empty:
            # If no band matches, fall back to last band charge
            return float(b["charge_per_t"].iloc[-1])
        return float(hit["charge_per_t"].iloc[0])

    if m <= brk:
        return band_charge(m)

    # beyond breakpoint: band at breakpoint + linear add
    base = band_charge(brk)
    return base + ((m - brk) * per)


def _apply_stock_pricing_with_routing(sell_prices: pd.DataFrame, delivery_postcode: str) -> tuple[pd.DataFrame, dict]:
    """
    For STOCK rows only:
      - choose nearest store that has the product
      - compute road miles (routing.get_road_miles)
      - compute haulage (£/t)
      - add haulage onto Price (so optimiser can compare vs real suppliers)
    Returns:
      (priced_df, stock_meta)
    stock_meta is keyed by (Product, Location, Delivery Window) -> {store_name, miles, haulage_per_t}
    """
    df = sell_prices.copy()
    df["Supplier"] = df["Supplier"].fillna("").astype(str)
    df["Product"] = df["Product"].fillna("").astype(str)
    df["Location"] = df["Location"].fillna("").astype(str)
    df["Delivery Window"] = df["Delivery Window"].fillna("").astype(str)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)

    stores = list_stock_stores(active_only=True)
    sp = list_stock_store_products(active_only=True)
    settings = get_haulage_settings()
    bands = list_haulage_bands(active_only=True)

    # Nothing to do if no stores/products configured
    if stores is None or stores.empty or sp is None or sp.empty:
        return df[df["Supplier"] != STOCK_SUPPLIER_NAME].copy(), {}

    # Prep lookup: product -> stores list
    sp = sp.copy()
    sp["product"] = sp["product"].fillna("").astype(str).map(_norm_product)
    sp["store_id"] = sp["store_id"].astype(str)

    stores = stores.copy()
    stores["store_id"] = stores["store_id"].astype(str)
    stores["name"] = stores.get("name", "").fillna("").astype(str)
    stores["postcode"] = stores.get("postcode", "").fillna("").astype(str)

    store_by_id = {r["store_id"]: r.to_dict() for _, r in stores.iterrows()}

    product_to_store_ids = {}
    for _, r in sp.iterrows():
        product_to_store_ids.setdefault(r["product"], set()).add(r["store_id"])

    break_miles = float(settings.get("break_miles", 0.0))
    per_mile_per_t = float(settings.get("per_mile_per_t", 0.0))

    stock_meta = {}

    for i, r in df.iterrows():
        if r["Supplier"] != STOCK_SUPPLIER_NAME:
            continue

        prod = _norm_product(r["Product"])
        cand_ids = list(product_to_store_ids.get(prod, []))
        if not cand_ids:
            # no store has this product -> remove STOCK row by setting price huge
            df.at[i, "Price"] = 1e12
            continue

        # Find nearest store by road miles
        best = None
        for sid in cand_ids:
            s = store_by_id.get(sid)
            if s is None:
                continue
            s_post = str(s.get("postcode", "")).strip()
            if not s_post:
                continue

            miles = float(routing.get_road_miles(s_post, delivery_postcode))
            if best is None or miles < best["miles"]:
                best = {"store_id": sid, "store_name": str(s.get("name", "")), "miles": miles}

        if best is None:
            df.at[i, "Price"] = 1e12
            continue

        haul = _haulage_per_t(best["miles"], bands=bands, break_miles=break_miles, per_mile_per_t=per_mile_per_t)

        # Add haulage onto price
        df.at[i, "Price"] = float(df.at[i, "Price"]) + float(haul)

        # Save meta for audit in quote lines + submit
        k = (str(r["Product"]), str(r["Location"]), str(r["Delivery Window"]))
        stock_meta[k] = {
            "store_name": best["store_name"],
            "miles": float(best["miles"]),
            "haulage_per_t": float(haul),
        }

    # Remove "unpriceable" STOCK rows (where we set huge sentinel)
    df = df[df["Price"] < 1e11].copy()
    return df, stock_meta

def _norm_col(c: str) -> str:
    # normalise column names to avoid £ / NBSP / trailing whitespace mismatches
    return (
        str(c)
        .replace("\u00a0", " ")   # NBSP -> space
        .strip()
        .lower()
    )

def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """
    Find a column in df by normalised name.
    candidates should be raw strings like 'Addons £/t', 'addons_per_t', etc.
    """
    if df is None or df.empty:
        return None

    norm_map = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        hit = norm_map.get(_norm_col(cand))
        if hit:
            return hit
    return None

def _build_quote_lines(
    book_code: str,
    alloc_df: pd.DataFrame,
    res: dict,
    basket: list,
    delivery_delta_map: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (quote_lines_df, totals_dict)

    quote_lines_df includes per-line: base sell, lot £/t, delivery £/t, addons £/t, all-in £/t, line totals.
    Lot charge is applied per supplier and joined back to each allocation line.
    """
    if alloc_df is None or alloc_df.empty:
        return pd.DataFrame(), {
            "tonnes": 0.0,
            "base_value": 0.0,
            "lot_value": 0.0,
            "addons_value": 0.0,
            "delivery_value": 0.0,
            "all_in_value": 0.0,
        }

    work = alloc_df.copy()
    work.columns = [str(c).replace("\u00a0"," ").strip() for c in work.columns]


    # ---- Normalise expected columns from optimiser output ----
    # base sell column used by optimiser (typically "Price")
    base_col = "Price" if "Price" in work.columns else None
    if base_col is None:
        # fallback if optimiser returns something else
        for c in ["Sell Price", "sell_price", "sell"]:
            if c in work.columns:
                base_col = c
                break
    if base_col is None:
        raise ValueError(f"Allocation is missing a base sell column. Columns: {list(work.columns)}")

    # supplier and qty
    if "Supplier" not in work.columns:
        raise ValueError(f"Allocation is missing 'Supplier'. Columns: {list(work.columns)}")
    if "Qty" not in work.columns:
        raise ValueError(f"Allocation is missing 'Qty'. Columns: {list(work.columns)}")

    work["Supplier"] = work["Supplier"].astype(str)
    work["Qty"] = pd.to_numeric(work["Qty"], errors="coerce").fillna(0.0)
    work["Base £/t"] = pd.to_numeric(work[base_col], errors="coerce").fillna(0.0)

    # ---- Lot charge per supplier (£/t) from res["lot_charges"] ----
    lot_map = {}
    lc = res.get("lot_charges")
    if lc:
        lc_df = pd.DataFrame(lc).copy()

        # Try common column names
        sup_c = "Supplier" if "Supplier" in lc_df.columns else ("supplier" if "supplier" in lc_df.columns else None)

        charge_c = None
        for cand in ["Charge £/t", "charge_per_t", "charge", "Charge", "charge_per_t_gbp"]:
            if cand in lc_df.columns:
                charge_c = cand
                break

        if sup_c and charge_c:
            lc_df[sup_c] = lc_df[sup_c].astype(str)
            lc_df[charge_c] = pd.to_numeric(lc_df[charge_c], errors="coerce").fillna(0.0)
            lot_map = dict(zip(lc_df[sup_c], lc_df[charge_c]))

    work["Lot £/t"] = work["Supplier"].map(lot_map).fillna(0.0)

     # ---- Delivery delta per line (fert only) ----
    work["Delivery £/t"] = 0.0
    if book_code == "fert":
        # Build a lookup from basket for delivery method by (Product, Location, Delivery Window)
        bdf = pd.DataFrame(basket) if basket else pd.DataFrame()
        dm_lookup = {}
        if not bdf.empty and all(c in bdf.columns for c in ["Product", "Location", "Delivery Window"]):
            if "Delivery Method" in bdf.columns:
                for _, r in bdf.iterrows():
                    key = (str(r["Product"]), str(r["Location"]), str(r["Delivery Window"]))
                    if key not in dm_lookup:  # keep first match only
                        dm_lookup[key] = str(r.get("Delivery Method", "Delivered"))

        def _dm_delta(row):
            key = (str(row.get("Product", "")), str(row.get("Location", "")), str(row.get("Delivery Window", "")))
            dm = dm_lookup.get(key, "Delivered")
            return float(delivery_delta_map.get(dm, 0.0))

        work["Delivery £/t"] = work.apply(_dm_delta, axis=1)

    # ---- Addons per line (seed optimiser output) ----
    # NOTE: this must run for BOTH books (Seed uses it, Fert usually 0)
    addons_col = _find_col(
        work,
        "Addons £/t",
        "Addons £ per t",
        "Addons/t",
        "addons_per_t",
        "addons per t",
        "addons",
        "addon £/t",
        "addon_per_t",
    )

    if addons_col is None:
        work["Addons £/t"] = 0.0
    else:
        work["Addons £/t"] = pd.to_numeric(work[addons_col], errors="coerce").fillna(0.0)


    # ---- All-in per line ----
     # ---- STOCK haulage (fert only; only applies when Supplier == STOCK) ----
    work["Stock Haulage £/t"] = 0.0
    work["Stock Store"] = ""
    work["Stock Miles"] = 0.0

    if book_code == "fert":
        meta = res.get("stock_meta") or {}
        # meta keyed by (Product, Location, Delivery Window)
        if meta and all(c in work.columns for c in ["Product", "Location", "Delivery Window"]):
            for i, r in work.iterrows():
                if str(r["Supplier"]) != STOCK_SUPPLIER_NAME:
                    continue
                k = (str(r["Product"]), str(r["Location"]), str(r["Delivery Window"]))
                m = meta.get(k)
                if not m:
                    continue
                work.at[i, "Stock Haulage £/t"] = float(m.get("haulage_per_t", 0.0))
                work.at[i, "Stock Store"] = str(m.get("store_name", ""))
                work.at[i, "Stock Miles"] = float(m.get("miles", 0.0))

    # IMPORTANT: Optimiser "Price" for STOCK already includes haulage (we add it in _apply_stock_pricing_with_routing).
    # For quote breakdown we want Base £/t EX haulage, so subtract it back out for STOCK lines.
    if book_code == "fert" and "Supplier" in work.columns and "Base £/t" in work.columns:
        is_stock = work["Supplier"].astype(str) == STOCK_SUPPLIER_NAME
        if is_stock.any():
            work.loc[is_stock, "Base £/t"] = (
                work.loc[is_stock, "Base £/t"] - work.loc[is_stock, "Stock Haulage £/t"]
            )
            work.loc[is_stock, "Base £/t"] = work.loc[is_stock, "Base £/t"].clip(lower=0.0)

    # ---- All-in per line ----
    work["All-in £/t"] = (
        work["Base £/t"] +
        work["Lot £/t"] +
        work["Delivery £/t"] +
        work["Addons £/t"] +
        work["Stock Haulage £/t"]
    )

    work["Stock haulage value"] = work["Stock Haulage £/t"] * work["Qty"]

    # ---- Values ----
    work["Base value"] = work["Base £/t"] * work["Qty"]
    work["Lot value"] = work["Lot £/t"] * work["Qty"]
    work["Delivery value"] = work["Delivery £/t"] * work["Qty"]
    work["Addons value"] = work["Addons £/t"] * work["Qty"]
    work["Line total"] = work["All-in £/t"] * work["Qty"]

    tonnes = float(work["Qty"].sum())
    totals = {
        "tonnes": tonnes,
        "base_value": float(work["Base value"].sum()),
        "lot_value": float(work["Lot value"].sum()),
        "delivery_value": float(work["Delivery value"].sum()),
        "addons_value": float(work["Addons value"].sum()),
        "stock_haulage_value": float(work["Stock haulage value"].sum()),
        "all_in_value": float(work["Line total"].sum()),
    }
    return work, totals


def page_trader_pricing():
    st.subheader("Trader | Pricing")

    book_code = _pick_book("trader_pricing", default="fert")
    st.session_state["presence_context"] = book_code

    _page_pricing_impl(book_code=book_code, role_code="trader")

def _page_pricing_impl(book_code: str, role_code: str):
    sid, df = _get_latest_prices_df_for(book_code)
    if df is None or df.empty:
        st.warning("No supplier snapshot available. Admin must publish one.")
        return

    # role-based Sell Price
    df = _apply_role_margins(df, role_code=role_code)

    _inject_offer_css()
    df_off = _apply_offers_to_prices_df(book_code, df)

    # --- Inject STOCK pseudo-supplier rows for fertiliser ---
    active_stock_products = set()
    if book_code == "fert":
        sp = list_stock_store_products(active_only=True)
        if sp is not None and not sp.empty:
            active_stock_products = set(sp["product"].astype(str).tolist())
            df_off = _inject_stock_rows(df_off, active_stock_products)
            st.caption(f"DEBUG: rows={len(df_off)} | stock_rows={(df_off['Supplier'].astype(str) == STOCK_SUPPLIER_NAME).sum()}")

    settings = get_settings()
    timeout_min = int(settings.get("basket_timeout_minutes", "20"))
    tiers = get_small_lot_tiers()

    # ---- Fertiliser delivery options ----
    delivery_options = ["Delivered"]
    delivery_delta_map = {"Delivered": 0.0}

    if book_code == "fert":
        ddf = list_fert_delivery_options(active_only=True)
        if ddf is not None and not ddf.empty:
            delivery_options = ddf["name"].astype(str).tolist()
            delivery_delta_map = {
                str(r["name"]): float(r["delta_per_t"])
                for _, r in ddf.iterrows()
                if pd.notna(r["name"]) and pd.notna(r["delta_per_t"])
            }
        # Ensure default exists
        if "Delivered" not in delivery_delta_map:
            delivery_options = ["Delivered"] + [x for x in delivery_options if x != "Delivered"]
            delivery_delta_map["Delivered"] = 0.0

    delivery_postcode = ""
    if book_code == "fert":
        delivery_postcode = st.text_input(
            "Delivery postcode (required to price STOCK lines)",
            key=_ss_key(book_code, "delivery_postcode"),
            placeholder="e.g. LN1 1AB",
        )

    # ---- Seed add-ons (treatments) ----
    # Only required for Seed book. Provides:
    # - addons_options: list of treatment names for the multiselect
    # - addons_catalog: dict name -> charge_per_t for optimise_basket()
    addons_options = []
    addons_catalog = {}

    if book_code == "seed":
        tdf = list_seed_treatments(active_only=True)
        if tdf is not None and not tdf.empty:
            addons_options = tdf["name"].astype(str).tolist()
            addons_catalog = {
                str(r["name"]): float(r["charge_per_t"])
                for _, r in tdf.iterrows()
                if pd.notna(r["name"]) and pd.notna(r["charge_per_t"])
            }

    # Namespaced session keys
    basket_key = _ss_key(book_code, "basket")
    basket_created_key = _ss_key(book_code, "basket_created_at")
    last_optim_key = _ss_key(book_code, "last_optim_result")
    last_optim_snap_key = _ss_key(book_code, "last_optim_snapshot")

    # Basket state
    _ensure_basket_for(book_code)

    # Expiry
    age_sec = time.time() - st.session_state[basket_created_key]
    if age_sec > timeout_min * 60:
        st.session_state[basket_key] = []
        st.session_state[basket_created_key] = time.time()
        st.info("Basket expired and has been cleared.")

    st.caption(f"Using supplier snapshot: {sid[:8]} | Basket timeout: {timeout_min} min")

    c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 3])
    
    with c1:
        product = st.selectbox(
            "Product",
            sorted(df["Product"].dropna().unique().tolist()),
            key=_ss_key(book_code, "pricing_product")
        )
    
    with c2:
        loc_opts = sorted(
            df.loc[df["Product"] == product, "Location"].dropna().unique().tolist()
        )
        location = st.selectbox(
            "Location",
            loc_opts,
            key=_ss_key(book_code, "pricing_location")
        )
    
    with c3:
        win_opts = sorted(
            df.loc[(df["Product"] == product) & (df["Location"] == location), "Delivery Window"]
              .dropna().unique().tolist()
        )
        window = st.selectbox(
            "Delivery Window",
            win_opts,
            key=_ss_key(book_code, "pricing_window")
        )
    
    # Notes / Cost/kg N (safe now)
    meta_cols = [c for c in ["Notes", "Cost/kg N"] if c in df.columns]
    if meta_cols:
        sel = df[
            (df["Product"] == product) &
            (df["Location"] == location) &
            (df["Delivery Window"] == window)
        ][["Supplier"] + meta_cols].copy()
    
        st.markdown("#### Notes / Cost of N (for selection)")
        if sel.empty:
            st.caption("No supplier rows found for this selection.")
        else:
            st.dataframe(sel, use_container_width=True, hide_index=True)

    with c4:
        qty = st.number_input(
            "Qty (t)",
            min_value=0.0,
            value=10.0,
            step=1.0,
            key=_ss_key(book_code, "pricing_qty")
        )
    
    # Right-most selector depends on book
    delivery_method = "Delivered"
    addons_selected = []
    
    with c5:
        if book_code == "fert":
            delivery_method = st.selectbox(
                "Delivery method",
                options=delivery_options,
                index=delivery_options.index("Delivered") if "Delivered" in delivery_options else 0,
                key=_ss_key(book_code, "pricing_delivery_method"),
            )
        else:
            addons_selected = st.multiselect(
                "Treatments (0–6)",
                options=addons_options,
                default=[],
                key=_ss_key(book_code, "pricing_addons"),
            )
            if len(addons_selected) > 6:
                st.error("Max 6 treatments per line.")

    if st.button("Add to basket", use_container_width=True, key=_ss_key(book_code, "btn_add_to_basket")):
        if book_code == "seed" and len(addons_selected) > 6:
            st.error("Max 6 treatments per line.")
            return

        item = {
            "line_id": str(uuid.uuid4()),
            "Product": product,
            "Location": location,
            "Delivery Window": window,
            "Qty": float(qty),
            "Addons": addons_selected if book_code == "seed" else [],
        }
        if book_code == "fert":
            item["Delivery Method"] = delivery_method

        st.session_state[basket_key].append(item)
        st.rerun()

    st.divider()

    # Basket view
    if not st.session_state[basket_key]:
        st.info("Basket is empty.")
        return

    bdf = pd.DataFrame(st.session_state[basket_key])
    st.markdown("### Basket")
    st.dataframe(bdf, use_container_width=True, hide_index=True)

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Clear basket", use_container_width=True, key=_ss_key(book_code, "btn_clear_basket")):
            st.session_state[basket_key] = []
            st.session_state[basket_created_key] = time.time()
            st.rerun()

    with colB:
        if st.button("Optimise", type="primary", use_container_width=True, key=_ss_key(book_code, "btn_optimise")):
            # Use offer-adjusted sell prices (Effective Sell Price)
            sell_prices = df_off[["Supplier", "Product", "Location", "Delivery Window", "Effective Sell Price"]].copy()
            sell_prices = sell_prices.rename(columns={"Effective Sell Price": "Price"})
            sell_prices["Supplier"] = sell_prices["Supplier"].astype(str)

            stock_meta = {}

            if book_code == "fert":
                pc = (delivery_postcode or "").strip()

                # If no postcode, STOCK must not be eligible in the optimiser
                if not pc:
                    sell_prices = sell_prices[sell_prices["Supplier"] != STOCK_SUPPLIER_NAME].copy()
                else:
                    # Price STOCK rows with nearest-store haulage and keep meta for audit
                    sell_prices, stock_meta = _apply_stock_pricing_with_routing(sell_prices, pc)

            if book_code == "fert":
                res = optimise_basket(
                    supplier_prices=sell_prices,
                    basket=st.session_state[basket_key],
                    tiers=tiers,
                    addons_catalog=None
                )
            else:
                res = optimise_basket(
                    supplier_prices=sell_prices,
                    basket=st.session_state[basket_key],
                    tiers=None,
                    addons_catalog=addons_catalog
                )
             # attach stock meta (used later for quote-line audit)
            if stock_meta:
                res["stock_meta"] = stock_meta
                res["delivery_postcode"] = (delivery_postcode or "").strip()

            if not res.get("ok"):
                st.error(res.get("error", "Unknown error"))
                return

            st.session_state[last_optim_key] = res
            st.session_state[last_optim_snap_key] = sid
            st.success("Optimisation complete. Review below.")
            st.rerun()

    # Show optimisation result if available
    res = st.session_state.get(last_optim_key)
    if (not res) or (st.session_state.get(last_optim_snap_key) != sid):
        st.info("Optimise to generate an allocation before checkout.")
        return

    st.markdown("### Optimal Allocations")

    alloc_df = pd.DataFrame(res["allocation"])
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    if res.get("lot_charges"):
        st.markdown("### Small-Lot Charges")
        st.dataframe(pd.DataFrame(res["lot_charges"]), use_container_width=True, hide_index=True)

    st.markdown("### Totals")

    quote_lines_df, totals = _build_quote_lines(
        book_code=book_code,
        alloc_df=pd.DataFrame(res["allocation"]),
        res=res,
        basket=st.session_state[basket_key],
        delivery_delta_map=delivery_delta_map,
    )
    
    st.markdown("### Your ADM Quote(s)")
    show_cols = []
    for c in ["Product","Location","Delivery Window","Qty","Supplier","Base £/t","Lot £/t","Delivery £/t","Addons £/t","Stock Haulage £/t","Stock Store","Stock Miles","All-in £/t","Line total"]:
        if c in quote_lines_df.columns:
            show_cols.append(c)
    
    quote_view = quote_lines_df[show_cols].copy()

    # ----- Offer detection for quote lines -----
    offers_now = list_active_offers(book_code)
    offer_keys_any = set()
    offer_keys_sup = set()

    if offers_now is not None and not offers_now.empty:
        for _, rr in offers_now.iterrows():
            # (Product, Location, Window)
            offer_keys_any.add((str(rr["product"]), str(rr["location"]), str(rr["delivery_window"])))
            # (Product, Location, Window, Supplier) if supplier specified
            sup = str(rr.get("supplier", "") or "")
            if sup.strip():
                offer_keys_sup.add((str(rr["product"]), str(rr["location"]), str(rr["delivery_window"]), sup))

    def _is_offer_row(row) -> bool:
        p = str(row.get("Product", ""))
        l = str(row.get("Location", ""))
        w = str(row.get("Delivery Window", ""))
        s = str(row.get("Supplier", ""))
        return ((p, l, w) in offer_keys_any) or ((p, l, w, s) in offer_keys_sup)

    quote_view["Offer"] = quote_view.apply(_is_offer_row, axis=1)

    # Force rerun so blink actually updates (1s)
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=1000, key=_ss_key(book_code, "quote_blink_refresh"))
    except Exception:
        pass

    # Blink toggle (works reliably): changes every second
    blink_on = (int(time.time()) % 2) == 0


    # Optional: format money columns nicely
    money_cols = ["Base £/t", "Lot £/t", "Delivery £/t", "Addons £/t", "All-in £/t", "Line total"]
    fmt = {}
    for c in money_cols:
        if c in quote_view.columns:
            fmt[c] = "£{:,.2f}".format
    if "Qty" in quote_view.columns:
        fmt["Qty"] = "{:,.2f}".format

    styler = quote_view.style.format(fmt)

    # Red + "blink" (by alternating style every second)
    if "Offer" in quote_view.columns:
        if blink_on:
            styler = styler.apply(
                lambda r: ["color:#d11; font-weight:700;" if bool(r.get("Offer")) else "" for _ in r],
                axis=1
            )
        else:
            # alternate state: still red but slightly lighter
            styler = styler.apply(
                lambda r: ["color:#d11; font-weight:700; opacity:0.60;" if bool(r.get("Offer")) else "" for _ in r],
                axis=1
            )

    # Bold the All-in column
    if "All-in £/t" in quote_view.columns:
        styler = styler.set_properties(subset=["All-in £/t"], **{"font-weight": "700"})

    st.dataframe(styler, use_container_width=True, hide_index=True)

    st.divider()

    # --- Lot charge map: Supplier -> charge_per_t (fert only) ---
    lot_charge_per_t_map = {}
    if book_code == "fert" and res.get("lot_charges"):
        # res["lot_charges"] rows look like: Supplier, Tonnes, Charge £/t, Lot Charge
        lcd = pd.DataFrame(res["lot_charges"])
        # Defensive: handle either "Charge £/t" or "charge_per_t" naming
        if "Charge £/t" in lcd.columns:
            lot_charge_per_t_map = dict(zip(lcd["Supplier"], lcd["Charge £/t"]))
        elif "charge_per_t" in lcd.columns:
            lot_charge_per_t_map = dict(zip(lcd["Supplier"], lcd["charge_per_t"]))

    st.markdown("### Checkout")

    if book_code == "seed":
        st.info(
            "Seed ordering is not enabled yet. You can view and optimise seed pricing, "
            "but submitting seed orders is currently disabled."
        )
        return

    trader_note = st.text_area(
        "Order note (optional)",
        placeholder="Customer/account, terms, anything relevant.",
        key=_ss_key(book_code, "trader_note")
    )

    if st.button("Submit order to Admin", type="primary", use_container_width=True, key=_ss_key(book_code, "btn_submit")):
        alloc_lines = []
        for r in res["allocation"]:
            prod = r["Product"]
            loc = r["Location"]
            win = r["Delivery Window"]
            sup = r["Supplier"]
            qty = float(r["Qty"])

            # Use offer-adjusted data
            match = df_off[
                (df_off["Product"] == prod) &
                (df_off["Location"] == loc) &
                (df_off["Delivery Window"] == win) &
                (df_off["Supplier"] == sup)
            ].copy()
            
            if match.empty:
                st.error(f"Internal error: could not find price row for {prod}/{loc}/{win}/{sup}")
                return
            
            sell_price_base = float(match.iloc[0]["Effective Sell Price"])

            base_col = "Price" if "Price" in match.columns else ("Base Price" if "Base Price" in match.columns else None)
            if base_col is None:
                st.error("Internal error: missing base price column (expected 'Price' or 'Base Price').")
                return
            base_price = float(match.iloc[0][base_col])

            # Apply delivery delta (fert only)
            delivery_method_line = ""
            delivery_delta = 0.0
            if book_code == "fert":
                # Find basket line matching Product/Location/Window to get its delivery method
                bdf2 = pd.DataFrame(st.session_state[basket_key])
                dm = "Delivered"
                if not bdf2.empty and "Delivery Method" in bdf2.columns:
                    hit = bdf2[
                        (bdf2["Product"] == prod) &
                        (bdf2["Location"] == loc) &
                        (bdf2["Delivery Window"] == win)
                    ].iloc[:1]
                    if not hit.empty:
                        dm = str(hit.iloc[0].get("Delivery Method", "Delivered"))
                delivery_method_line = dm
                delivery_delta = float(delivery_delta_map.get(dm, 0.0))

            # --- Small-lot charge (fert only) applied per supplier ---
            lot_charge_per_t = 0.0
            lot_charge_value = 0.0
            if book_code == "fert":
                lot_charge_per_t = float(lot_charge_per_t_map.get(sup, 0.0))
                lot_charge_value = lot_charge_per_t * qty

            # All-in sell price per tonne for this line
            # --- Addons per t (seed optimiser output) ---
            # For fert this will be 0.0; for seed it should come from optimiser allocation.
            addons_per_t = 0.0
            for k in ["Addons £/t", "Addons £ per t", "addons_per_t", "Addons/t", "addons"]:
                if k in r and r[k] is not None:
                    try:
                        addons_per_t = float(r[k])
                        break
                    except Exception:
                        pass

            # All-in sell price per tonne for this line
            # --- STOCK haulage per t (fert only; only if Supplier == STOCK) ---
            stock_haul_per_t = 0.0
            stock_store = ""
            stock_miles = 0.0
            
            if book_code == "fert" and sup == STOCK_SUPPLIER_NAME:
                meta = (res.get("stock_meta") or {})
                k = (str(prod), str(loc), str(win))
                m = meta.get(k)
                if m:
                    stock_haul_per_t = float(m.get("haulage_per_t", 0.0))
                    stock_store = str(m.get("store_name", ""))
                    stock_miles = float(m.get("miles", 0.0))
            sell_price = sell_price_base + delivery_delta + lot_charge_per_t + addons_per_t + stock_haul_per_t
            unit = str(match.iloc[0]["Unit"])
            pcat = str(match.iloc[0].get("Product Category", ""))

            alloc_lines.append({
                "Product Category": pcat,
                "Product": prod,
                "Location": loc,
                "Delivery Window": win,
                "Qty": qty,
                "Unit": unit,
                "Supplier": sup,
                "Base Price": base_price,
                "Sell Price": sell_price,
                "Delivery Method": delivery_method_line,
                "Delivery Delta Per T": delivery_delta,
                "Small Lot Charge Per T": lot_charge_per_t,
                "Small Lot Charge Value": lot_charge_value,
                "Addons Per T": addons_per_t,
                "Addons Value": addons_per_t * qty,
                "Stock Haulage Per T": stock_haul_per_t,
                "Stock Haulage Value": stock_haul_per_t * qty,
                "Stock Store": stock_store,
                "Stock Miles": stock_miles,
                "Delivery Postcode": (res.get("delivery_postcode") or ""),
            })
        try:
            user = st.session_state.get("user", "unknown")
            order_id = create_order_from_allocation(
                created_by=user,
                supplier_snapshot_id=sid,
                allocation_lines=alloc_lines,
                trader_note=trader_note
            )
            st.session_state[basket_key] = []
            st.session_state[basket_created_key] = time.time()
            st.session_state[last_optim_key] = None
            st.session_state[last_optim_snap_key] = None

            st.success(f"Order submitted: {order_id[:8]}")
        except Exception as e:
            st.error(str(e))

def page_trader_orders():
    st.subheader("Trader | Orders")

    df = list_orders_for_user(user = st.session_state.get("user", "unknown"))
    if df.empty:
        st.info("No orders yet.")
        return

    status = st.selectbox("Filter status", ["ALL", "PENDING", "COUNTERED", "CONFIRMED", "FILLED", "REJECTED", "CANCELLED"])
    work = df.copy()
    if status != "ALL":
        work = work[work["status"] == status]

    work["label"] = work["created_at_utc"] + " | " + work["status"] + " | " + work["order_id"].str[:8]
    sel = st.selectbox("Select order", work["label"].tolist())

    order_id = work.loc[work["label"] == sel, "order_id"].iloc[0]
    header = get_order_header(order_id)
    lines = get_order_lines(order_id)
    actions = get_order_actions(order_id)

    st.markdown("### Order summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Order", header["order_id"][:8])
    c2.metric("Status", header["status"])
    c3.metric("Snapshot", header["supplier_snapshot_id"][:8] if header.get("supplier_snapshot_id") else "")
    c4.metric("Created (UTC)", str(header["created_at_utc"])[:19])

    if header.get("trader_note"):
        st.caption(f"Trader note: {header['trader_note']}")
    if header.get("admin_note"):
        st.caption(f"Admin note: {header['admin_note']}")

    with st.expander("Show technical details", expanded=False):
        st.json(header)

    st.markdown("### Lines")
    st.dataframe(lines, use_container_width=True, hide_index=True)

    sell_total = float((lines["Sell Price"] * lines["Qty"]).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Sell value", f"£{sell_total:,.2f}")
    c2.metric("Lines", int(len(lines)))
    c3.metric("Total tonnes", f"{float(lines['Qty'].sum()):,.2f} t")

    st.markdown("### Timeline")
    st.dataframe(actions[["action_type", "action_at_utc", "action_by"]], use_container_width=True, hide_index=True)

    st.divider()

    if header["status"] in ("PENDING", "COUNTERED"):
        if st.button("Cancel order", use_container_width=True):
            try:
                trader_cancel_order(order_id, user = st.session_state.get("user", "unknown"), expected_version=header["version"])
                st.success("Order cancelled.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if header["status"] == "COUNTERED":
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Accept counter", type="primary", use_container_width=True):
                try:
                    trader_accept_counter(order_id, user = st.session_state.get("user", "unknown"), expected_version=header["version"])
                    st.success("Counter accepted. Order is now CONFIRMED.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with c2:
            st.info("If you disagree, cancel and resubmit.")


def page_admin_pricing():
    if st.session_state.get("role") != "admin":
        st.warning("Admin access required.")
        return

    st.subheader("Admin | Pricing")

    book_code = _pick_book("admin_pricing", default="fert")
    st.session_state["presence_context"] = book_code

    _page_admin_pricing_impl(book_code=book_code)

def _page_admin_pricing_impl(book_code: str):
    settings = get_settings()
    timeout = st.number_input(
        "Basket timeout (minutes)",
        min_value=1,
        value=int(settings.get("basket_timeout_minutes", "20")),
        key=_ss_key(book_code, "admin_timeout")
    )
    if st.button("Save settings", use_container_width=True, key=_ss_key(book_code, "btn_save_settings")):
        set_setting("basket_timeout_minutes", str(timeout))
        st.success("Settings saved.")

    st.divider()

    if book_code == "fert":
        st.markdown("### Small-lot tiers (Fertiliser only)")
        tiers = get_small_lot_tiers()
        if tiers is None or tiers.empty:
            tiers = pd.DataFrame(columns=["min_t", "max_t", "charge_per_t", "active"])
        else:
            tiers = tiers[["min_t", "max_t", "charge_per_t", "active"]].copy()

        edited = st.data_editor(
            tiers,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=_ss_key(book_code, "tiers_editor"),
            column_config={
                "min_t": st.column_config.NumberColumn("Min t", min_value=0.0, step=0.1),
                "max_t": st.column_config.NumberColumn("Max t", min_value=0.0, step=0.1),
                "charge_per_t": st.column_config.NumberColumn("Charge (£/t)", min_value=0.0, step=0.1),
                "active": st.column_config.CheckboxColumn("Active"),
            }
        )

        st.divider()
        st.markdown("### Fertiliser delivery / collection options")

        ddf = list_fert_delivery_options(active_only=False)
        if ddf is None or ddf.empty:
            ddf = pd.DataFrame(columns=["name", "delta_per_t", "active"])
        else:
            ddf = ddf[["name", "delta_per_t", "active"]].copy()

        edited_opts = st.data_editor(
            ddf,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=_ss_key(book_code, "fert_delivery_options_editor"),
            column_config={
                "name": st.column_config.TextColumn("Option name"),
                "delta_per_t": st.column_config.NumberColumn("Price delta (£/t)", step=0.5),
                "active": st.column_config.CheckboxColumn("Active"),
            }
        )

        if st.button("Save delivery options", type="primary", use_container_width=True, key=_ss_key(book_code, "btn_save_fert_delivery_options")):
            try:
                save_fert_delivery_options(edited_opts, st.session_state.get("user", "unknown"))
                st.success("Delivery options saved.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

        if st.button("Save tiers", type="primary", use_container_width=True, key=_ss_key(book_code, "btn_save_tiers")):
            try:
                edited2 = edited.copy()
                if "active" not in edited2.columns:
                    edited2["active"] = 1
                edited2["active"] = edited2["active"].apply(lambda x: 1 if bool(x) else 0)
                if "max_t" not in edited2.columns:
                    edited2["max_t"] = None

                save_small_lot_tiers(edited2[["min_t", "max_t", "charge_per_t", "active"]])
                st.success("Tiers saved.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    else:
        st.markdown("### Seed treatments (add-ons, 0–6 per line)")
        tdf = list_seed_treatments(active_only=False)
        if tdf is None or tdf.empty:
            tdf = pd.DataFrame(columns=["name", "charge_per_t", "active"])
        else:
            tdf = tdf[["name", "charge_per_t", "active"]].copy()

        edited = st.data_editor(
            tdf,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=_ss_key(book_code, "seed_treatments_editor"),
            column_config={
                "name": st.column_config.TextColumn("Treatment name"),
                "charge_per_t": st.column_config.NumberColumn("Charge (£/t)", min_value=0.0, step=0.5),
                "active": st.column_config.CheckboxColumn("Active"),
            }
        )

        if st.button("Save seed treatments", type="primary", use_container_width=True, key=_ss_key(book_code, "btn_save_seed_treatments")):
            try:
                save_seed_treatments(edited, st.session_state.get("user", "unknown"))
                st.success("Seed treatments saved.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()

    st.markdown("### Admin margins")

    margin_role = st.selectbox(
        "Margins apply to role",
        options=["trader", "wholesaler"],
        index=0,
        key=_ss_key(book_code, "admin_margin_role"),
    )
    mdf = list_margins(active_only=True, role_code=margin_role)
    if mdf.empty:
        st.info("No active margins set.")
    else:
        show = mdf[["margin_id", "scope_type", "scope_value", "margin_per_t", "created_at_utc", "created_by"]].copy()
        show = show.rename(columns={"margin_per_t": "Margin (£/t)"})
        st.dataframe(show, use_container_width=True, hide_index=True)

        mid = st.number_input("Deactivate margin_id", min_value=0, value=0, step=1, key=_ss_key(book_code, "deact_mid"))
        if st.button("Deactivate selected margin", use_container_width=True, key=_ss_key(book_code, "btn_deact")):
            if mid <= 0:
                st.error("Enter a valid margin_id.")
            else:
                deactivate_margin(int(mid))
                st.success(f"Deactivated margin_id={int(mid)}")
                st.rerun()

    st.markdown("#### Add new margin")
    scope_type = st.selectbox("Scope", ["category", "product"], key=_ss_key(book_code, "margin_scope_type"))
    scope_value = st.text_input("Category/Product name (exact match)", key=_ss_key(book_code, "margin_scope_value"))
    margin_per_t = st.number_input("Margin (£/t)", value=0.0, step=0.5, key=_ss_key(book_code, "margin_per_t"))
    if st.button("Add margin", type="primary", use_container_width=True, key=_ss_key(book_code, "btn_add_margin")):
        try:
            add_margin(scope_type, scope_value, float(margin_per_t), st.session_state.get("user", "unknown"), role_code=margin_role)
            st.success("Margin added.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()
    
    st.markdown("### Reprice (no upload)")

    if st.button(
        "Reprice latest snapshot using current margins and publish",
        type="primary",
        use_container_width=True,
        key=_ss_key(book_code, "btn_reprice_latest")
    ):
        try:
            # 1) Load latest snapshot for this book
            sid_latest, df_latest = _get_latest_prices_df_for(book_code)
            if df_latest is None or df_latest.empty:
                st.error("No latest snapshot found to reprice.")
                return

            # 2) Apply margins to compute new Sell Price
            margins = get_effective_margins(role_code=margin_role)
            tmp = df_latest.copy()

            # Ensure Sell Price exists and starts from base
            tmp["Price"] = pd.to_numeric(tmp["Price"], errors="coerce")
            tmp["Sell Price"] = pd.to_numeric(tmp.get("Sell Price", tmp["Price"]), errors="coerce")
            tmp["Sell Price"] = tmp["Price"]  # always recompute from base

            tmp = apply_margins(tmp, margins)

            # 3) Publish as a new snapshot
            new_sid = BOOKS_BY_CODE[book_code]["publish_snapshot"](
                tmp,
                st.session_state.get("user", "unknown"),
                b""  # no source file
            )

            st.success(f"Repriced and published new snapshot: {new_sid[:8]}")
            st.rerun()

        except ValueError as e:
            # this will catch your duplicate source_hash message
            msg = str(e)
            if "duplicate source_hash" in msg.lower() or "already been published" in msg.lower():
                st.warning(
                    "No changes detected versus an existing published snapshot. "
                    "Either margins didn’t change the Sell Prices, or the result matches a prior snapshot."
                )
            else:
                st.error(msg)
        except Exception as e:
            st.error(str(e))

    st.markdown(f"### {BOOKS_BY_CODE[book_code]['upload_label']}")

    # Namespaced session keys for preview workflow
    preview_df_key = _ss_key(book_code, "admin_preview_df")
    preview_bytes_key = _ss_key(book_code, "admin_preview_bytes")
    preview_name_key = _ss_key(book_code, "admin_preview_filename")
    
    # Namespaced uploader reset token
    tok_key = _ss_key(book_code, "upload_token")
    if tok_key not in st.session_state:
        st.session_state[tok_key] = 0
    
    upload_key = _ss_key(book_code, f"upload_excel_{st.session_state[tok_key]}")
    up = st.file_uploader("Upload Excel", type=["xlsx"], key=upload_key)
        
    # If a new file is uploaded, validate and store preview in session_state (do NOT publish yet)
    if up is not None:
        content = up.read()
        try:
            df_raw = BOOKS_BY_CODE[book_code]["loader"](content)
            df_raw = _ensure_upload_has_sell_price(df_raw)
    
            st.session_state[preview_df_key] = df_raw
            st.session_state[preview_bytes_key] = content
            st.session_state[preview_name_key] = getattr(up, "name", "")
            st.success("Validated. Preview loaded. Review / apply margins / edit Sell Price, then publish below.")
        except Exception as e:
            st.error(str(e))
    
    # If we have a preview in session_state, show the preview workflow UI
    if preview_df_key in st.session_state and st.session_state.get(preview_df_key) is not None:
        dfp = st.session_state[preview_df_key].copy()
    
        st.caption(f"Preview file: {st.session_state.get(preview_name_key, '')}")
        st.markdown("#### Preview (Base vs Sell)")
    
        # Controls
        cA, cB, cC, cD = st.columns([1, 1, 1, 1])
    
        with cA:
            if st.button("Apply current margins to Sell Price", use_container_width=True, key=_ss_key(book_code, "btn_apply_margins_preview")):
                try:
                    margins = get_effective_margins(role_code=margin_role)
            
                    tmp = dfp.copy()
                    # apply_margins expects 'Price' and returns/sets 'Sell Price'
                    tmp = apply_margins(tmp, margins)
            
                    dfp["Sell Price"] = pd.to_numeric(tmp["Sell Price"], errors="coerce")
                    st.session_state[preview_df_key] = dfp
                    st.success("Applied margins to Sell Price (preview).")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    
        with cB:
            if st.button("Reset Sell = Base", use_container_width=True, key=_ss_key(book_code, "btn_reset_sell_preview")):
                dfp["Sell Price"] = pd.to_numeric(dfp["Price"], errors="coerce")
                st.session_state[preview_df_key] = dfp
                st.success("Sell Price reset to Base Price (preview).")
                st.rerun()
    
        with cC:
            if st.button("Clear preview", use_container_width=True, key=_ss_key(book_code, "btn_clear_preview")):
                st.session_state[preview_df_key] = None
                st.session_state[preview_bytes_key] = None
                st.session_state[preview_name_key] = ""
                # Attempt to clear uploader value as well
                st.session_state[tok_key] += 1
                st.rerun()
    
        with cD:
            st.write("")  # spacer
    
        # Editable preview: allow editing Sell Price only
        editable_cols = []
        for col in ["Supplier", "Product Category", "Product", "Location", "Delivery Window",
            "Price", "Sell Price", "Unit", "Notes", "Cost/kg N"]:
            if col in dfp.columns:
                editable_cols.append(col)
    
        edited = st.data_editor(
            dfp[editable_cols],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=[c for c in editable_cols if c != "Sell Price"],  # Notes/Cost remain disabled
            column_config={
                "Price": st.column_config.NumberColumn("Base Price", format="£%.2f"),
                "Sell Price": st.column_config.NumberColumn("Sell Price", min_value=0.0, step=0.5, format="£%.2f"),
                "Notes": st.column_config.TextColumn("Notes"),
                "Cost/kg N": st.column_config.TextColumn("Cost/kg N"),
            },
            key=_ss_key(book_code, "preview_editor")
        )
        st.session_state[preview_df_key] = edited.copy()

        to_pub = st.session_state[preview_df_key].copy()
        for c in ["Notes", "Cost/kg N"]:
            if c not in to_pub.columns:
                to_pub[c] = ""
            
        st.divider()
    
        # Publish button (ONLY publish from preview)
        if st.button(
            BOOKS_BY_CODE[book_code]["publish_button"],
            type="primary",
            use_container_width=True,
            key=_ss_key(book_code, "btn_publish_from_preview")
        ):
            try:
                to_pub = st.session_state[preview_df_key].copy()
                for c in ["Notes", "Cost/kg N"]:
                    if c not in to_pub.columns:
                        to_pub[c] = ""
                
                # Ensure numeric + non-null
                to_pub["Price"] = pd.to_numeric(to_pub["Price"], errors="coerce")
                to_pub["Sell Price"] = pd.to_numeric(to_pub["Sell Price"], errors="coerce")
                to_pub = to_pub.dropna(subset=["Price", "Sell Price"])
    
                sid = BOOKS_BY_CODE[book_code]["publish_snapshot"](
                    to_pub,
                    st.session_state.get("user", "unknown"),
                    st.session_state.get(preview_bytes_key, b"")
                )
    
                st.success(f"Published snapshot: {sid}")
    
                # Clear preview after publish
                st.session_state[preview_df_key] = None
                st.session_state[preview_bytes_key] = None
                st.session_state[preview_name_key] = ""
                st.session_state[tok_key] += 1
                st.rerun()
            except Exception as e:
                st.error(str(e))

def page_admin_orders():
    if st.session_state.get("role") != "admin":
        st.warning("Admin access required.")
        return

    st.subheader("Admin | Orders")

    status = st.selectbox("Status filter", ["ALL", "PENDING", "COUNTERED", "CONFIRMED", "FILLED", "REJECTED", "CANCELLED"])
    if status == "ALL":
        odf = list_orders_admin(None)
    else:
        odf = list_orders_admin(status)

    if odf.empty:
        st.info("No orders.")
        return

    odf = odf.copy()
    odf["label"] = odf["created_at_utc"] + " | " + odf["status"] + " | " + odf["created_by"] + " | " + odf["order_id"].str[:8]
    sel = st.selectbox("Select order", odf["label"].tolist())

    order_id = odf.loc[odf["label"] == sel, "order_id"].iloc[0]
    header = get_order_header(order_id)
    lines = get_order_lines(order_id)
    actions = get_order_actions(order_id)

    st.markdown("### Order summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Order", header["order_id"][:8])
    c2.metric("Status", header["status"])
    c3.metric("Trader", header["created_by"])
    c4.metric("Created (UTC)", str(header["created_at_utc"])[:19])

    if header.get("trader_note"):
        st.caption(f"Trader note: {header['trader_note']}")
    if header.get("admin_note"):
        st.caption(f"Admin note: {header['admin_note']}")

    with st.expander("Show technical details", expanded=False):
        st.json(header)

    st.markdown("### Lines")
    st.dataframe(lines, use_container_width=True, hide_index=True)

    gross_margin = float(((lines["Sell Price"] - lines["Base Price"]) * lines["Qty"]).sum())
    sell_value = float((lines["Sell Price"] * lines["Qty"]).sum())
    base_value = float((lines["Base Price"] * lines["Qty"]).sum())
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sell value", f"£{sell_value:,.2f}")
    c2.metric("Base value", f"£{base_value:,.2f}")
    c3.metric("Gross margin", f"£{gross_margin:,.2f}")
    
    gm_pct = (gross_margin / sell_value * 100.0) if sell_value else 0.0
    c4.metric("GM %", f"{gm_pct:.2f}%")

    st.markdown("### Timeline")
    st.dataframe(actions[["action_type", "action_at_utc", "action_by"]], use_container_width=True, hide_index=True)

    st.divider()

    if header["status"] in ("PENDING", "COUNTERED"):
        st.markdown("### Counter / Confirm / Reject")

        admin_note = st.text_area("Admin note (optional)", value=header.get("admin_note", "") or "")

        editable = lines[["line_no", "Product", "Location", "Delivery Window", "Qty", "Supplier", "Sell Price"]].copy()
        
        edited = st.data_editor(
            editable,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["line_no", "Product", "Location", "Delivery Window", "Qty", "Supplier"],
            column_config={
                "Sell Price": st.column_config.NumberColumn(
                    "Sell Price",
                    min_value=0.0,
                    step=0.5,
                    format="£%.2f",
                )
            },
        )


        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("Confirm as-is", type="primary", use_container_width=True):
                try:
                    admin_confirm_order(
                        order_id,
                        admin_user=st.session_state.get("user", "unknown"),
                        admin_note=admin_note,
                        expected_version=header["version"]
                    )
                    st.success("Order CONFIRMED.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        with c2:
            if st.button("Send counter", use_container_width=True):
                try:
                    # Validate at least one Sell Price changed
                    orig = pd.to_numeric(lines.set_index("line_no")["Sell Price"], errors="coerce").fillna(0.0)
                    new = pd.to_numeric(edited.set_index("line_no")["Sell Price"], errors="coerce").fillna(0.0)

                    if (orig == new).all():
                        st.warning("No Sell Price changes detected. Change at least one line to send a counter.")
                        return
            
                    admin_counter_order(
                        order_id,
                        admin_user=st.session_state.get("user", "unknown"),
                        edited_lines=edited,
                        admin_note=admin_note,
                        expected_version=header["version"]
                    )
                    st.success("Counter sent. Status = COUNTERED.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        with c3:
            if st.button("Reject", use_container_width=True):
                try:
                    admin_reject_order(order_id, admin_user=st.session_state.get("user", "unknown"), admin_note=admin_note, expected_version=header["version"])
                    st.success("Order REJECTED.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if header["status"] == "CONFIRMED":
        st.markdown("### Fill")
        if st.button("Mark FILLED", type="primary", use_container_width=True):
            try:
                admin_mark_filled(order_id, admin_user=st.session_state.get("user", "unknown"), expected_version=header["version"])
                st.success("Order marked FILLED.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown("### Filled order margin report")
    rep = admin_margin_report()
    if rep.empty:
        st.info("No filled orders yet.")
    else:
        st.dataframe(rep, use_container_width=True, hide_index=True)

@st.cache_data(show_spinner=False)
def _build_history_timeseries(book_code: str, role_code: str, max_snaps: int = 80) -> pd.DataFrame:
    """
    Builds a long history table for charting:
    one row per snapshot x price row.
    Uses Sell Price only (never exposes base Price to traders).
    """
    snaps = BOOKS_BY_CODE[book_code]["list_snapshots"]()
    if snaps is None or snaps.empty:
        return pd.DataFrame()

    snaps = snaps.copy()
    snaps["published_at_utc_dt"] = pd.to_datetime(snaps["published_at_utc"], errors="coerce", utc=True)
    snaps = snaps.dropna(subset=["published_at_utc_dt"]).sort_values("published_at_utc_dt", ascending=True)

    snaps = snaps.tail(int(max_snaps))

    rows = []
    for _, s in snaps.iterrows():
        sid = s["snapshot_id"]
        ts = s["published_at_utc_dt"]

        df = BOOKS_BY_CODE[book_code]["load_prices"](sid)
        if df is None or df.empty:
            continue

        df = df.copy()

        # Require base Price
        if "Price" not in df.columns:
            continue
        
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=["Price"])
        
        # Recompute Sell Price for THIS role from base + effective margins
        df["Sell Price"] = df["Price"]
        margins = get_effective_margins(role_code=role_code)

        df["Sell Price"] = pd.to_numeric(df["Sell Price"], errors="coerce")
        df = apply_margins(df, margins)
        df = df.dropna(subset=["Sell Price"])

        keep_cols = [c for c in ["Product", "Location", "Delivery Window", "Supplier", "Unit"] if c in df.columns]
        df = df[keep_cols + ["Sell Price"]].copy()

        df["snapshot_id"] = sid
        df["published_at_utc"] = ts

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)

@st.cache_data(show_spinner=False)
def _build_history_timeseries_admin_base(book_code: str, max_snaps: int = 180) -> pd.DataFrame:
    """
    Admin-only history table for analysis:
    one row per snapshot x price row.
    Uses BASE Price ("Price") for competitiveness analytics.
    """
    snaps = BOOKS_BY_CODE[book_code]["list_snapshots"]()
    if snaps is None or snaps.empty:
        return pd.DataFrame()

    snaps = snaps.copy()
    snaps["published_at_utc_dt"] = pd.to_datetime(snaps["published_at_utc"], errors="coerce", utc=True)
    snaps = snaps.dropna(subset=["published_at_utc_dt"]).sort_values("published_at_utc_dt", ascending=True)
    snaps = snaps.tail(int(max_snaps))

    rows = []
    for _, s in snaps.iterrows():
        sid = s["snapshot_id"]
        ts = s["published_at_utc_dt"]

        df = BOOKS_BY_CODE[book_code]["load_prices"](sid)
        if df is None or df.empty:
            continue

        df = df.copy()

        # Enforce BASE Price availability
        if "Price" not in df.columns:
            continue

        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=["Price"])

        keep_cols = [c for c in ["Product", "Location", "Delivery Window", "Supplier", "Unit"] if c in df.columns]
        df = df[keep_cols + ["Price"]].copy()

        df["snapshot_id"] = sid
        df["published_at_utc"] = ts

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def _history_aggregate_for_chart(
    hist: pd.DataFrame,
    product: str,
    location: str | None,
    mode: str,
) -> pd.DataFrame:
    """
    Returns a chart-ready table with:
      published_at_utc, Series, line_price, band_min, band_max

    mode:
      - "Cheapest (any window)" -> Series="Cheapest" with band across all windows/suppliers
      - "Cheapest per window"   -> Series=Delivery Window with band across suppliers per window
    """
    df = hist.copy()

    if "Product" not in df.columns:
        return pd.DataFrame()

    df = df[df["Product"].astype(str) == str(product)]

    if location and location != "ALL" and "Location" in df.columns:
        df = df[df["Location"].astype(str) == str(location)]

    if df.empty:
        return pd.DataFrame()

    if "published_at_utc" not in df.columns:
        return pd.DataFrame()

    if mode == "Cheapest (any window)" or "Delivery Window" not in df.columns:
        # Band across ALL windows + suppliers for that snapshot
        agg = (
            df.groupby(["published_at_utc"], dropna=False)
              .agg(
                  band_min=("Sell Price", "min"),
                  band_max=("Sell Price", "max"),
                  line_price=("Sell Price", "min"),  # cheapest as the line
              )
              .reset_index()
        )
        agg["Series"] = "Cheapest"
        return agg[["published_at_utc", "Series", "line_price", "band_min", "band_max"]]

    # Cheapest per window, band across suppliers (per snapshot, per window)
    agg = (
        df.groupby(["published_at_utc", "Delivery Window"], dropna=False)
          .agg(
              band_min=("Sell Price", "min"),
              band_max=("Sell Price", "max"),
              line_price=("Sell Price", "min"),
          )
          .reset_index()
          .rename(columns={"Delivery Window": "Series"})
    )
    return agg[["published_at_utc", "Series", "line_price", "band_min", "band_max"]]


def _apply_time_window_rolling_mean(ts: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    Adds rolling_mean column using a true TIME window (e.g., 7D, 31D) per Series.
    Works with irregular snapshot timestamps.
    """
    if ts is None or ts.empty:
        return ts

    out = ts.copy()
    out = out.dropna(subset=["published_at_utc", "Series", "line_price"])
    out = out.sort_values(["Series", "published_at_utc"])

    # compute rolling mean per series using time-based window
    res = []
    for series, g in out.groupby("Series", dropna=False):
        gg = g.copy().set_index("published_at_utc").sort_index()
        gg["rolling_mean"] = gg["line_price"].rolling(f"{int(window_days)}D", min_periods=1).mean()
        gg = gg.reset_index()
        res.append(gg)

    return pd.concat(res, ignore_index=True) if res else out


def _render_pretty_price_chart(
    ts: pd.DataFrame,
    *,
    title: str,
    show_bands: bool,
    show_rolling: bool,
    window_days: int,
):
    if ts is None or ts.empty:
        st.info("No history points found for that selection.")
        return

    # Add rolling mean if requested
    if show_rolling:
        ts = _apply_time_window_rolling_mean(ts, window_days=window_days)

    # Base encodings
    base = alt.Chart(ts).encode(
        x=alt.X("published_at_utc:T", title="Snapshot time (UTC)")
    )

    layers = []

    # Banding area (min-max)
    if show_bands and ("band_min" in ts.columns) and ("band_max" in ts.columns):
        band = base.mark_area(opacity=0.18).encode(
            y=alt.Y("band_min:Q", title="Sell price (£/t)"),
            y2="band_max:Q",
            color=alt.Color("Series:N", legend=alt.Legend(title="Series")),
            tooltip=[
                alt.Tooltip("published_at_utc:T", title="Time (UTC)"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("band_min:Q", title="Min (£/t)", format=",.2f"),
                alt.Tooltip("band_max:Q", title="Max (£/t)", format=",.2f"),
            ],
        )
        layers.append(band)

    # Main line (cheapest)
    line = base.mark_line(point=True).encode(
        y=alt.Y("line_price:Q", title="Sell price (£/t)"),
        color=alt.Color("Series:N", legend=alt.Legend(title="Series")),
        tooltip=[
            alt.Tooltip("published_at_utc:T", title="Time (UTC)"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("line_price:Q", title="Cheapest (£/t)", format=",.2f"),
        ],
    )
    layers.append(line)

    # Rolling mean line
    if show_rolling and "rolling_mean" in ts.columns:
        ma = base.mark_line(strokeDash=[6, 4], point=False).encode(
            y=alt.Y("rolling_mean:Q", title="Sell price (£/t)"),
            color=alt.Color("Series:N", legend=alt.Legend(title="Series")),
            tooltip=[
                alt.Tooltip("published_at_utc:T", title="Time (UTC)"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("rolling_mean:Q", title=f"Rolling {window_days}D (£/t)", format=",.2f"),
            ],
        )
        layers.append(ma)

    chart = alt.layer(*layers).properties(
        height=380,
        title=title,
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def page_history(role_code: str = "trader"):
    st.subheader(f"{role_code.title()} | Price History")

    # Use different keys so trader/wholesale can remember their own selection
    key = f"{role_code}_price_history"
    book_code = _pick_book(key, default="fert")
    st.session_state["presence_context"] = book_code

    _page_history_impl(book_code=book_code, role_code=role_code)

def _page_history_impl(book_code: str, role_code: str):
    snaps = BOOKS_BY_CODE[book_code]["list_snapshots"]()
    if snaps is None or snaps.empty:
        st.info("No snapshots yet.")
        return

    # ---------------------------
    # Price history chart
    # ---------------------------
    st.markdown("### Price History Analysis")

    cA, cB, cC, cD, cE = st.columns([2, 2, 2, 2, 3])

    with cA:
        max_snaps = st.number_input(
            "Lookback snapshots",
            min_value=10,
            max_value=500,
            value=80,
            step=10,
            key=_ss_key(book_code, "hist_max_snaps"),
        )

    with cB:
        show_bands = st.checkbox(
            "Show min–max band",
            value=True,
            key=_ss_key(book_code, "hist_show_bands"),
        )

    with cC:
        show_rolling = st.checkbox(
            "Show rolling average",
            value=False,
            key=_ss_key(book_code, "hist_show_rolling"),
        )

    with cD:
        window_days = st.number_input(
            "Rolling window (days)",
            min_value=1,
            max_value=365,
            value=31,
            step=1,
            key=_ss_key(book_code, "hist_window_days"),
            disabled=(not show_rolling),
        )

    hist = _build_history_timeseries(book_code=book_code, role_code=role_code, max_snaps=int(max_snaps))
    if hist is None or hist.empty:
        st.info("No historical price points available yet.")
    else:
        products = sorted(hist["Product"].dropna().astype(str).unique().tolist()) if "Product" in hist.columns else []
        if not products:
            st.info("No Product values available to chart.")
        else:
            with cE:
                mode = st.selectbox(
                    "Graph rule",
                    options=["Cheapest (any window)", "Cheapest per window"],
                    key=_ss_key(book_code, "hist_chart_mode"),
                    help=(
                        "Cheapest (any window): one line = min Sell Price per snapshot across all windows/suppliers.\n"
                        "Cheapest per window: separate lines per Delivery Window (still min across suppliers)."
                    ),
                )

            c1, c2 = st.columns([3, 2])
            with c1:
                product = st.selectbox(
                    "Grade / Product",
                    options=products,
                    key=_ss_key(book_code, "hist_chart_product"),
                )

            with c2:
                if "Location" in hist.columns:
                    locs = ["ALL"] + sorted(
                        hist.loc[hist["Product"].astype(str) == str(product), "Location"]
                            .dropna().astype(str).unique().tolist()
                    )
                    location = st.selectbox(
                        "Location",
                        options=locs,
                        key=_ss_key(book_code, "hist_chart_location"),
                    )
                else:
                    location = "ALL"

            ts = _history_aggregate_for_chart(
                hist=hist,
                product=product,
                location=location,
                mode=mode,
            )

            _render_pretty_price_chart(
                ts=ts,
                title=f"{product} | {book_code.upper()} | {mode}" + (f" | {location}" if location and location != "ALL" else ""),
                show_bands=bool(show_bands),
                show_rolling=bool(show_rolling),
                window_days=int(window_days),
            )

    st.divider()

    # ---------------------------
    # Snapshot table (SELL ONLY for traders)
    # ---------------------------
    st.markdown("### Snapshot history table")

    snaps = snaps.copy()
    snaps["label"] = snaps["published_at_utc"] + " | " + snaps["published_by"] + " | " + snaps["snapshot_id"].str[:8]
    label = st.selectbox("Select snapshot", snaps["label"].tolist(), key=_ss_key(book_code, "hist_select"))
    sid = snaps.loc[snaps["label"] == label, "snapshot_id"].iloc[0]

    df = BOOKS_BY_CODE[book_code]["load_prices"](sid)
    if df is None or df.empty:
        st.info("Snapshot is empty.")
        return

    # enforce Sell Price column exists
    df = df.copy()
    df = _apply_role_margins(df, role_code=role_code)
    if "Sell Price" not in df.columns:
        if "Price" in df.columns:
            df["Sell Price"] = df["Price"]
        else:
            st.error("Snapshot has no Sell Price (and no Price fallback).")
            return

    # HARD RULE: traders never see base Price and can't even search it
    is_admin = (st.session_state.get("role") == "admin")

    base_cols = [c for c in ["Product Category", "Product", "Location", "Delivery Window", "Supplier", "Unit"] if c in df.columns]
    show_cols = base_cols + ["Sell Price"]

    if is_admin:
        show_base = st.checkbox(
            "Admin: show base Price column",
            value=False,
            key=_ss_key(book_code, "hist_admin_show_base"),
        )
        if show_base and "Price" in df.columns:
            show_cols = base_cols + ["Price", "Sell Price"]

    view = df[show_cols].copy()

    q = st.text_input("Search", key=_ss_key(book_code, "hist_search"))
    if q:
        ql = q.lower()
        view = view[view.apply(lambda r: any(ql in str(v).lower() for v in r.values), axis=1)]

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sell Price": st.column_config.NumberColumn("Sell Price (£/t)", format="£%.2f"),
            "Price": st.column_config.NumberColumn("Base Price (£/t)", format="£%.2f"),
        },
    )

def page_trader_best_prices():
    st.subheader("Trader | FULL LOAD PRICES")

    book_code = _pick_book("trader_full_load", default="fert")
    st.session_state["presence_context"] = book_code

    _page_best_prices_impl(book_code=book_code, role_code="trader")

def _page_best_prices_impl(book_code: str, role_code: str):
    sid, df = _get_latest_prices_df_for(book_code)
    if df is None or df.empty:
        st.warning("No supplier snapshot available. Admin must publish one.")
        return
    
    df = _apply_role_margins(df, role_code=role_code)
    
    _inject_offer_css()
    df_off = _apply_offers_to_prices_df(book_code, df)

    # build board off the offer-adjusted sell price
    tmp = df_off.copy()
    tmp["Sell Price"] = tmp["Effective Sell Price"]

    board = _best_prices_board(tmp)

    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])

    with f1:
        cats = ["ALL"] + sorted(board["Product Category"].unique().tolist())
        cat = st.selectbox("Product Category", cats, key=_ss_key(book_code, "bp_cat"))

    with f2:
        prods = ["ALL"] + sorted(board["Product"].unique().tolist())
        prod = st.selectbox("Product", prods, key=_ss_key(book_code, "bp_prod"))

    with f3:
        locs = ["ALL"] + sorted(board["Location"].unique().tolist())
        loc = st.selectbox("Location", locs, key=_ss_key(book_code, "bp_loc"))

    with f4:
        wins = ["ALL"] + sorted(board["Delivery Window"].unique().tolist())
        win = st.selectbox("Delivery Window", wins, key=_ss_key(book_code, "bp_win"))

    view = board.copy()
    if cat != "ALL":
        view = view[view["Product Category"] == cat]
    if prod != "ALL":
        view = view[view["Product"] == prod]
    if loc != "ALL":
        view = view[view["Location"] == loc]
    if win != "ALL":
        view = view[view["Delivery Window"] == win]

    st.divider()
    st.caption(f"Supplier snapshot: {sid[:8]} | Rows: {len(view)}")

    # Basket expiry for this book
    _ensure_basket_for(book_code)
    settings = get_settings()
    timeout_min = int(settings.get("basket_timeout_minutes", "20"))

    basket_key = _ss_key(book_code, "basket")
    basket_created_key = _ss_key(book_code, "basket_created_at")

    age_sec = time.time() - st.session_state[basket_created_key]
    if age_sec > timeout_min * 60:
        st.session_state[basket_key] = []
        st.session_state[basket_created_key] = time.time()
        st.info("Basket expired and has been cleared.")

    st.markdown("### Add to basket from board")
    qty = st.number_input("Qty (t) for selected lines", min_value=0.0, value=10.0, step=1.0, key=_ss_key(book_code, "bp_qty"))

    editable = view.rename(columns={
        "Product Category": "Category",
        "Delivery Window": "Window",
    }).copy()
    editable.insert(0, "Add", False)

    edited = st.data_editor(
        editable[["Add", "Category", "Product", "Location", "Window", "Best Price", "Unit", "Supplier"]],
        use_container_width=True,
        hide_index=True,
        disabled=["Category", "Product", "Location", "Window", "Best Price", "Unit", "Supplier"],
        column_config={
            "Add": st.column_config.CheckboxColumn("Add"),
            "Best Price": st.column_config.NumberColumn("Best Price", format="£%.2f"),
        },
        key=_ss_key(book_code, "best_prices_board_editor"),
    )

    if st.button("Add selected lines to basket", type="primary", use_container_width=True, key=_ss_key(book_code, "bp_add_selected")):
        selected = edited[edited["Add"] == True].copy()
        if selected.empty:
            st.warning("Tick at least one line.")
        else:
            for _, r in selected.iterrows():
                item = {
                    "Product": r["Product"],
                    "Location": r["Location"],
                    "Delivery Window": r["Window"],
                    "Qty": float(qty),
                }
                if book_code == "fert":
                    item["Delivery Method"] = "Delivered"
                st.session_state[basket_key].append(item)
            st.success(f"Added {len(selected)} line(s) to basket.")
            st.info("Go to Trader | Pricing to optimise and submit the order.")
            st.rerun()

    st.divider()

    with st.expander("Show full table", expanded=False):
        st.dataframe(
            view.rename(columns={"Product Category": "Category", "Delivery Window": "Window"})[
                ["Category", "Product", "Location", "Window", "Best Price", "Unit", "Supplier"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={"Best Price": st.column_config.NumberColumn("Best Price", format="£%.2f")},
        )

def page_todays_offers(role_code: str = "trader"):
    st.subheader("Today's Offers")
    _inject_offer_css()

    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=10_000, key=f"offers_autorefresh_{role_code}")
    except Exception:
        pass

    book_code = _pick_book(f"{role_code}_offers", default="fert")
    st.session_state["presence_context"] = book_code

    _page_todays_offers_impl(book_code, role_code=role_code)


def _page_todays_offers_impl(book_code: str, role_code: str):
    sid, df = _get_latest_prices_df_for(book_code)
    if df is None or df.empty:
        st.warning("No snapshot available.")
        return

    df = _apply_role_margins(df, role_code=role_code)

    df_off = _apply_offers_to_prices_df(book_code, df)
    live = df_off[df_off["Offer Active"] == True].copy()

    if live.empty:
        st.info("No live offers right now.")
    else:
        now = datetime.now(timezone.utc)

        def _remain(iso):
            try:
                end = datetime.fromisoformat(str(iso))
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
        
                sec = int((end - now).total_seconds())
                if sec <= 0:
                    return "expired"
                h = sec // 3600
                m = (sec % 3600) // 60
                return f"{h}h {m}m"
            except Exception:
                return ""

        live["Ends in"] = live["Offer Ends"].apply(_remain)
        live["Was £/t"] = pd.to_numeric(live["Sell Price"], errors="coerce").fillna(0.0)
        live["Now £/t"] = pd.to_numeric(live["Effective Sell Price"], errors="coerce").fillna(0.0)
        live["Save £/t"] = live["Was £/t"] - live["Now £/t"]

        show = live[[
            "Product", "Location", "Delivery Window", "Supplier",
            "Offer Title", "Was £/t", "Now £/t", "Save £/t", "Ends in"
        ]].copy()

        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=1000, key=_ss_key(book_code, "quote_blink_refresh"))
        except Exception:
            pass

        blink_on = (int(time.time()) % 2) == 0

        # Force currency formatting to 2dp
        sty = show.style.format({
            "Was £/t": "£{:.2f}",
            "Now £/t": "£{:.2f}",
            "Save £/t": "£{:.2f}",
        })
        
        # Keep your blinking emphasis
        if blink_on:
            sty = sty.set_properties(
                subset=["Now £/t", "Save £/t", "Ends in"],
                **{"color": "#d11", "font-weight": "700"}
            )
        else:
            sty = sty.set_properties(
                subset=["Now £/t", "Save £/t", "Ends in"],
                **{"color": "#d11", "font-weight": "700", "opacity": "0.60"}
            )
        
        st.dataframe(sty, use_container_width=True, hide_index=True)

    # ---------------------------
    # Daily market comment (per book)
    # ---------------------------
    settings = get_settings()
    comment_key = f"daily_market_comment_{book_code}"
    current_comment = (settings.get(comment_key, "") or "").strip()

    st.markdown("### Daily market comment")

    # Everyone can read it
    if current_comment:
        st.markdown(current_comment)
    else:
        st.caption("No market comment published yet.")

    # Only admins can edit + save
    if st.session_state.get("role") == "admin":
        new_comment = st.text_area(
            "Admin edit (visible to all users)",
            value=current_comment,
            height=120,
            key=_ss_key(book_code, "daily_comment_editor"),
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Save market comment", type="primary", use_container_width=True, key=_ss_key(book_code, "btn_save_daily_comment")):
                set_setting(comment_key, new_comment.strip())
                st.success("Market comment saved.")
                st.rerun()
        with c2:
            st.caption("Saved per book (fert/seed) and shown directly under today’s offers.")

    # Admin controls (simple MVP)
    if st.session_state.get("role") == "admin":
        st.divider()
        st.markdown("### Admin: Create offer")

        c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
        with c1:
            product = st.selectbox("Product", sorted(df["Product"].dropna().unique().tolist()), key=_ss_key(book_code, "offer_prod"))
        with c2:
            locs = sorted(df.loc[df["Product"] == product, "Location"].dropna().unique().tolist())
            location = st.selectbox("Location", locs, key=_ss_key(book_code, "offer_loc"))
        with c3:
            wins = sorted(df.loc[(df["Product"] == product) & (df["Location"] == location), "Delivery Window"].dropna().unique().tolist())
            window = st.selectbox("Delivery Window", wins, key=_ss_key(book_code, "offer_win"))
        with c4:
            sups = sorted(df.loc[
                (df["Product"] == product) &
                (df["Location"] == location) &
                (df["Delivery Window"] == window),
                "Supplier"
            ].dropna().unique().tolist())
            supplier = st.selectbox("Supplier (optional)", ["(Any)"] + sups, key=_ss_key(book_code, "offer_sup"))

        c5, c6, c7 = st.columns([2, 2, 4])
        with c5:
            mode = st.selectbox("Mode", ["delta", "fixed"], key=_ss_key(book_code, "offer_mode"))
        with c6:
            value = st.number_input("Value (£/t)", min_value=0.0, value=5.0, step=0.5, key=_ss_key(book_code, "offer_val"))
        with c7:
            title = st.text_input("Title", value="Today's offer", key=_ss_key(book_code, "offer_title"))

        hours = st.number_input("Expires in (hours)", min_value=1, value=12, step=1, key=_ss_key(book_code, "offer_hours"))

        if st.button("Create offer", type="primary", use_container_width=True, key=_ss_key(book_code, "offer_create_btn")):
            now_utc = datetime.utcnow()  # naive UTC, matches db utc_now_iso format
            start = now_utc.strftime("%Y-%m-%d %H:%M:%S")
            end = (now_utc + timedelta(hours=int(hours))).strftime("%Y-%m-%d %H:%M:%S")

            create_offer(
                book_code=book_code,
                product=product,
                location=location,
                delivery_window=window,
                supplier=None if supplier == "(Any)" else supplier,
                mode=mode,
                value=float(value),
                title=title,
                starts_at_utc=start,
                ends_at_utc=end,
                created_by=st.session_state.get("user", "unknown")
            )
            st.success("Offer created.")
            st.rerun()

        st.divider()
        st.markdown("### Admin: Manage offers")
        all_off = list_offers(book_code=book_code, active_only=True)
        if all_off is None or all_off.empty:
            st.info("No offers created yet.")
        else:
            st.dataframe(all_off, use_container_width=True, hide_index=True)
            # pick an offer_id safely
            ids = all_off["offer_id"].astype(int).tolist() if "offer_id" in all_off.columns else []
            if ids:
                off_id = st.selectbox("Select offer_id to deactivate", ids, key=_ss_key(book_code, "offer_deact_id"))
                if st.button("Deactivate", use_container_width=True, key=_ss_key(book_code, "offer_deact_btn")):
                    deactivate_offer(int(off_id))
                    st.success(f"Offer {int(off_id)} deactivated.")
                    st.rerun()
            else:
                st.info("No offer_id column found in offers table.")

def page_wholesale_pricing():
    st.subheader("Wholesale | Pricing")

    book_code = _pick_book("wholesale_pricing", default="fert")
    st.session_state["presence_context"] = book_code

    _page_pricing_impl(book_code=book_code, role_code="wholesaler")


def page_wholesale_best_prices():
    st.subheader("Wholesale | FULL LOAD PRICES")

    book_code = _pick_book("wholesale_full_load", default="fert")
    st.session_state["presence_context"] = book_code

    _page_best_prices_impl(book_code=book_code, role_code="wholesaler")


def page_wholesale_orders():
    # Do NOT call page_trader_orders() because it hardcodes the Trader header
    st.subheader("Wholesale | Orders")

    df = list_orders_for_user(user=st.session_state.get("user", "unknown"))
    if df.empty:
        st.info("No orders yet.")
        return

    status = st.selectbox(
        "Filter status",
        ["ALL", "PENDING", "COUNTERED", "CONFIRMED", "FILLED", "REJECTED", "CANCELLED"],
        key="wh_orders_status"
    )
    work = df.copy()
    if status != "ALL":
        work = work[work["status"] == status]

    work["label"] = work["created_at_utc"] + " | " + work["status"] + " | " + work["order_id"].str[:8]
    sel = st.selectbox("Select order", work["label"].tolist(), key="wh_orders_sel")

    order_id = work.loc[work["label"] == sel, "order_id"].iloc[0]
    header = get_order_header(order_id)
    lines = get_order_lines(order_id)
    actions = get_order_actions(order_id)

    st.markdown("### Order summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Order", header["order_id"][:8])
    c2.metric("Status", header["status"])
    c3.metric("Snapshot", header["supplier_snapshot_id"][:8] if header.get("supplier_snapshot_id") else "")
    c4.metric("Created (UTC)", str(header["created_at_utc"])[:19])

    if header.get("trader_note"):
        st.caption(f"Trader note: {header['trader_note']}")
    if header.get("admin_note"):
        st.caption(f"Admin note: {header['admin_note']}")

    st.markdown("### Lines")
    st.dataframe(lines, use_container_width=True, hide_index=True)

    sell_total = float((lines["Sell Price"] * lines["Qty"]).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Sell value", f"£{sell_total:,.2f}")
    c2.metric("Lines", int(len(lines)))
    c3.metric("Total tonnes", f"{float(lines['Qty'].sum()):,.2f} t")

    st.markdown("### Timeline")
    st.dataframe(actions[["action_type", "action_at_utc", "action_by"]], use_container_width=True, hide_index=True)

    st.divider()

    if header["status"] in ("PENDING", "COUNTERED"):
        if st.button("Cancel order", use_container_width=True, key="wh_orders_cancel"):
            try:
                trader_cancel_order(order_id, user=st.session_state.get("user", "unknown"), expected_version=header["version"])
                st.success("Order cancelled.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if header["status"] == "COUNTERED":
        if st.button("Accept counter", type="primary", use_container_width=True, key="wh_orders_accept"):
            try:
                trader_accept_counter(order_id, user=st.session_state.get("user", "unknown"), expected_version=header["version"])
                st.success("Counter accepted. Order is now CONFIRMED.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
def page_wholesale_offers():
    st.subheader("Wholesale | Offers")
    _inject_offer_css()

    book_code = _pick_book("wholesale_offers", default="fert")
    st.session_state["presence_context"] = book_code

    _page_todays_offers_impl(book_code, role_code="wholesaler")

# ---------------------------
# Presence (sidebar widget)
# ---------------------------


def _utc_parse(ts: str) -> datetime:
    """
    Parse timestamps that may be stored as:
      - "YYYY-MM-DD HH:MM:SS" (naive)
      - "YYYY-MM-DDTHH:MM:SS+00:00" (aware)
    Always return an aware UTC datetime.
    """
    if ts is None:
        return datetime.now(timezone.utc)

    s = str(ts).strip()
    if not s:
        return datetime.now(timezone.utc)

    dt = datetime.fromisoformat(s)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


def render_presence_panel(current_page_name: str, *, refresh_ms: int = 10_000):
    """
    Sidebar presence panel only:
    - shows online users
    - toasts when users come/go (diff vs last refresh)

    NOTE: heartbeat is handled centrally in app.py to avoid double writes.
    """
    user = st.session_state.get("user", "") or ""
    role = st.session_state.get("role", "") or ""

    # Optional auto-refresh (install streamlit-autorefresh for true "live" feel)
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_ms, key=f"presence_autorefresh_{current_page_name}")
    except Exception:
        pass

    df = list_online_users(online_within_seconds=45)

    prev = set(st.session_state.get("presence_prev_online", []))
    now = set(df["user"].tolist()) if (df is not None and not df.empty and "user" in df.columns) else set()

    # Avoid toasting on first ever render
    if "presence_prev_online" in st.session_state:
        for u in sorted(now - prev):
            st.toast(f"{u} is now online")
        for u in sorted(prev - now):
            st.toast(f"{u} went offline")

    st.session_state["presence_prev_online"] = list(now)

    st.markdown("### Online now")

    if df is None or df.empty:
        st.caption("No active users detected.")
        return

    now_dt = datetime.now(timezone.utc)

    rows = []
    for _, r in df.iterrows():
        last_seen = _utc_parse(r["last_seen_utc"])
        sec_ago = int((now_dt - last_seen).total_seconds())
        rows.append({
            "User": r.get("user", ""),
            "Role": r.get("role", "") or "",
            "Context": r.get("context", "") or "",
            "Page": r.get("page", "") or "",
            "Last seen": f"{sec_ago}s ago",
        })

    out = pd.DataFrame(rows)

    st.dataframe(
        out,
        use_container_width=True,
        hide_index=True,
        column_config={
            "User": st.column_config.TextColumn("User"),
            "Role": st.column_config.TextColumn("Role"),
            "Context": st.column_config.TextColumn("Context"),
            "Page": st.column_config.TextColumn("Page"),
            "Last seen": st.column_config.TextColumn("Last seen"),
        }
    )

def page_admin_supplier_analysis():
    if st.session_state.get("role") != "admin":
        st.warning("Admin access required.")
        return

    st.subheader("Admin | Supplier Analysis")

    book_code = _pick_book("admin_supplier_analysis", default="fert")
    st.session_state["presence_context"] = book_code

    _page_admin_supplier_analysis_impl(book_code=book_code)

def page_admin_stock():
    st.subheader("Admin | Stock")
    st.session_state["presence_context"] = "fert"

    user = st.session_state.get("user") or st.session_state.get("username") or ""

    stores = list_stock_stores(active_only=False)
    store_products = list_stock_store_products(active_only=False)
    bands = list_haulage_bands(active_only=False)
    settings = get_haulage_settings()

    # Build product universe from latest fert snapshot (best UX)
    _, fert_df = _get_latest_prices_df_for("fert")
    product_universe = sorted(fert_df["Product"].dropna().astype(str).unique().tolist()) if fert_df is not None and not fert_df.empty else []

    tab1, tab2, tab3 = st.tabs(["Stores", "Store Products", "Haulage Tariff"])

    with tab1:
        st.markdown("### Stores")
        edited = st.data_editor(
            stores,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "active": st.column_config.CheckboxColumn("active"),
            },
        )
        if st.button("Save stores", type="primary"):
            save_stock_stores(edited, user=user or "admin")
            st.success("Saved stores.")
            st.rerun()

    with tab2:
        st.markdown("### Store → Products")
        # Build store_id choices
        store_id_opts = stores["store_id"].astype(str).tolist() if stores is not None and not stores.empty else []

        edited2 = st.data_editor(
            store_products,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "store_id": st.column_config.SelectboxColumn("store_id", options=store_id_opts),
                "product": st.column_config.SelectboxColumn("product", options=product_universe) if product_universe else None,
                "active": st.column_config.CheckboxColumn("active"),
            },
        )
        if st.button("Save store products", type="primary"):
            save_stock_store_products(edited2, user=user or "admin")
            st.success("Saved store products.")
            st.rerun()

    with tab3:
        st.markdown("### Haulage tariff")
        c1, c2 = st.columns(2)
        with c1:
            break_miles = st.number_input("Band breakpoint miles", min_value=0.0, value=float(settings["break_miles"]), step=1.0)
        with c2:
            per_mile = st.number_input("£/mile/t beyond breakpoint", min_value=0.0, value=float(settings["per_mile_per_t"]), step=0.01, format="%.2f")

        if st.button("Save haulage settings", type="primary"):
            set_haulage_settings(break_miles, per_mile, user=user or "admin")
            st.success("Saved haulage settings.")
            st.rerun()

        st.markdown("#### Bands (charge £/t)")
        edited3 = st.data_editor(
            bands,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "active": st.column_config.CheckboxColumn("active"),
            },
        )
        if st.button("Save haulage bands", type="primary", key="save_haulage_bands"):
            save_haulage_bands(edited3, user=user or "admin")
            st.success("Saved bands.")
            st.rerun()

def _page_admin_supplier_analysis_impl(book_code: str):
    st.markdown("### Supplier Analysis")

    c1, c2 = st.columns([2, 2])
    with c1:
        max_snaps = st.number_input(
            "Lookback snapshots",
            min_value=10, max_value=500, value=180, step=10,
            key=_ss_key(book_code, "supana_max_snaps"),
        )
    with c2:
        min_obs = st.number_input(
            "Min observations per supplier",
            min_value=1, max_value=9999, value=25, step=1,
            key=_ss_key(book_code, "supana_min_obs"),
        )

    hist = _build_history_timeseries_admin_base(book_code=book_code, max_snaps=int(max_snaps))
    if hist is None or hist.empty:
        st.info("No historical price points available yet.")
        return

    # Dropdown filters (built from available history)
    prod_opts = ["(All)"] + sorted(hist["Product"].dropna().astype(str).unique().tolist()) if "Product" in hist.columns else ["(All)"]
    loc_opts = ["(All)"]
    if "Location" in hist.columns:
        loc_opts += sorted(hist["Location"].dropna().astype(str).unique().tolist())
    
    c3, c4 = st.columns([3, 3])
    with c3:
        product = st.selectbox(
            "Product (optional)",
            options=prod_opts,
            index=0,
            key=_ss_key(book_code, "supana_product_filter"),
        )
    with c4:
        location = st.selectbox(
            "Location (optional)",
            options=loc_opts,
            index=0,
            key=_ss_key(book_code, "supana_location_filter"),
        )
    
    product = "" if product == "(All)" else str(product).strip()
    location = "" if location == "(All)" else str(location).strip()

    # Required columns
    required = ["published_at_utc", "Supplier", "Product", "Price"]
    missing = [c for c in required if c not in hist.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    df = hist.copy()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price", "published_at_utc"])

    if product.strip():
        df = df[df["Product"].astype(str) == product.strip()]
    if location.strip() and "Location" in df.columns:
        df = df[df["Location"].astype(str) == location.strip()]

    if df.empty:
        st.info("No rows match your filter.")
        return

    # Best price per snapshot (within current filters)
    best = (
        df.groupby("published_at_utc", dropna=False)["Price"]
          .min()
          .rename("best_price")
          .reset_index()
    )

    df = df.merge(best, on="published_at_utc", how="left")
    df["premium_to_best"] = df["Price"] - df["best_price"]
    df["is_cheapest"] = df["premium_to_best"] <= 1e-9

    sup = (
        df.groupby("Supplier", dropna=False)
          .agg(
              obs=("Price", "count"),
              cheapest_wins=("is_cheapest", "sum"),
              avg_sell=("Price", "mean"),
              med_sell=("Price", "median"),
              avg_premium=("premium_to_best", "mean"),
              med_premium=("premium_to_best", "median"),
              p90_premium=("premium_to_best", lambda x: x.quantile(0.90) if len(x) else 0.0),
              vol_std=("Price", "std"),
          )
          .reset_index()
    )
    sup["vol_std"] = sup["vol_std"].fillna(0.0)
    sup = sup[sup["obs"] >= int(min_obs)].copy()
    if sup.empty:
        st.info("No suppliers meet the minimum observation threshold.")
        return

    sup["cheapest_share_pct"] = (sup["cheapest_wins"] / sup["obs"]) * 100.0
    sup = sup.sort_values(["cheapest_share_pct", "avg_premium"], ascending=[False, True])

    # KPI
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Suppliers", int(len(sup)))
    k2.metric("Top supplier", str(sup.iloc[0]["Supplier"]))
    k3.metric("Cheapest share", f"{float(sup.iloc[0]['cheapest_share_pct']):.2f}%")
    k4.metric("Avg premium", f"£{float(sup.iloc[0]['avg_premium']):.2f}/t")
    k5.metric("Obs (top)", int(sup.iloc[0]["obs"]))

    st.divider()

    st.markdown("### Supplier Leaderboard")
    st.dataframe(
        sup[["Supplier","obs","cheapest_share_pct","avg_premium","med_premium","p90_premium","avg_sell","med_sell","vol_std"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "cheapest_share_pct": st.column_config.NumberColumn("Cheapest share (%)", format="%.2f"),
            "avg_premium": st.column_config.NumberColumn("Avg premium (£/t)", format="£%.2f"),
            "med_premium": st.column_config.NumberColumn("Median premium (£/t)", format="£%.2f"),
            "p90_premium": st.column_config.NumberColumn("P90 premium (£/t)", format="£%.2f"),
            "avg_sell": st.column_config.NumberColumn("Avg base (£/t)", format="£%.2f"),
            "med_sell": st.column_config.NumberColumn("Median base (£/t)", format="£%.2f"),
            "vol_std": st.column_config.NumberColumn("Volatility (std £/t)", format="£%.2f"),
        }
    )

    st.divider()

    st.markdown("### Trend: Premium to best over time")
    top_n = st.slider("Top N suppliers to plot", 3, 15, 8, key=_ss_key(book_code, "supana_topn"))
    top_suppliers = sup["Supplier"].head(int(top_n)).astype(str).tolist()

    chart_df = df[df["Supplier"].astype(str).isin(top_suppliers)].copy()
    chart_df = (
        chart_df.groupby(["published_at_utc", "Supplier"], dropna=False)
                .agg(premium=("premium_to_best", "min"))
                .reset_index()
                .sort_values(["Supplier","published_at_utc"])
    )

    roll_days = st.number_input("Rolling mean window (days)", 1, 365, 31, 1, key=_ss_key(book_code, "supana_roll"))
    out = []
    for sname, g in chart_df.groupby("Supplier", dropna=False):
        gg = g.set_index("published_at_utc").sort_index()
        gg["roll"] = gg["premium"].rolling(f"{int(roll_days)}D", min_periods=1).mean()
        gg = gg.reset_index()
        gg["Supplier"] = sname
        out.append(gg)
    chart_df = pd.concat(out, ignore_index=True) if out else chart_df

    base = alt.Chart(chart_df).encode(
        x=alt.X("published_at_utc:T", title="Snapshot time (UTC)"),
        color=alt.Color("Supplier:N", legend=alt.Legend(title="Supplier"))
    )

    line = base.mark_line(point=True).encode(
        y=alt.Y("premium:Q", title="Premium to best (£/t)"),
        tooltip=[
            alt.Tooltip("published_at_utc:T", title="Time (UTC)"),
            alt.Tooltip("Supplier:N", title="Supplier"),
            alt.Tooltip("premium:Q", title="Premium (£/t)", format=",.2f"),
        ]
    )

    roll = base.mark_line(strokeDash=[6,4]).encode(
        y=alt.Y("roll:Q", title="Premium to best (£/t)"),
        tooltip=[
            alt.Tooltip("roll:Q", title=f"Rolling {int(roll_days)}D", format=",.2f"),
        ]
    )

    st.altair_chart(
        alt.layer(line, roll).properties(height=360).interactive(),
        use_container_width=True
    )

    with st.expander("Drilldown table", expanded=False):
        drill = df[["published_at_utc","Supplier","Product","Price","best_price","premium_to_best"]].copy()
        drill = drill.sort_values(["published_at_utc","premium_to_best"], ascending=[False, True])
        st.dataframe(
            drill,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn("Sell (£/t)", format="£%.2f"),
                "best_price": st.column_config.NumberColumn("Best (£/t)", format="£%.2f"),
                "premium_to_best": st.column_config.NumberColumn("Premium (£/t)", format="£%.2f"),
            }
        )

def page_admin_user_data():
    if st.session_state.get("role") != "admin":
        st.warning("Admin access required.")
        return

    st.subheader("Admin | User Data")
    st.caption("DEBUG: Admin | User Data v2 loaded")


    # -------------------------
    # Controls
    # -------------------------
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        lookback_days = st.number_input("Lookback (days)", min_value=1, max_value=365, value=30, step=1, key="ud_lookback")
    with c2:
        # user filter
        users = ["(All)"] + (list_distinct_presence_users(days=int(lookback_days)) or [])
        sel_user = st.selectbox("User", users, index=0, key="ud_user")
        user_filter = None if sel_user == "(All)" else sel_user
    with c3:
        # page filter
        pages = ["(All)"] + (list_distinct_presence_pages(days=int(lookback_days)) or [])
        sel_page = st.selectbox("Page", pages, index=0, key="ud_page")
        page_filter = None if sel_page == "(All)" else sel_page

    with c4:
        ctx_label = st.selectbox(
            "Book",
            ["(All)", "Fertiliser", "Seed"],
            index=0,
            key="ud_ctx",
        )
        if ctx_label == "(All)":
            ctx_filter = None
        elif ctx_label == "Fertiliser":
            ctx_filter = "fert"
        else:  # "Seed"
            ctx_filter = "seed"

    # Build UTC range
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(lookback_days))
    start_utc = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_utc = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------
    # Pull data
    # -------------------------
    # Daily logins (sessions/day based on first event per session)
    logins = presence_daily_logins(start_utc=start_utc, end_utc=end_utc, user=user_filter, context=ctx_filter)
    transitions = presence_daily_page_transitions(start_utc=start_utc, end_utc=end_utc, user=user_filter, context=ctx_filter)
    sessions = presence_session_summary(start_utc=start_utc, end_utc=end_utc, user=user_filter, context=ctx_filter)
    
    events = list_presence_events(
        user=user_filter,
        page=page_filter,
        context=ctx_filter,
        start_utc=start_utc,
        end_utc=end_utc,
        limit=20000,
    )

    # -------------------------
    # Top KPIs
    # -------------------------
    # Unique users (from events in-window, respects filters)
    unique_users = int(events["user"].nunique()) if (events is not None and not events.empty and "user" in events.columns) else 0
    total_events = int(len(events)) if (events is not None and not events.empty) else 0
    total_sessions = int(len(sessions)) if (sessions is not None and not sessions.empty) else 0

    total_navs = 0
    if transitions is not None and not transitions.empty and "navigations" in transitions.columns:
        total_navs = int(pd.to_numeric(transitions["navigations"], errors="coerce").fillna(0).sum())

    avg_dur = 0
    if sessions is not None and not sessions.empty and "duration_seconds" in sessions.columns:
        avg_dur = int(pd.to_numeric(sessions["duration_seconds"], errors="coerce").fillna(0).mean())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Unique users", unique_users)
    k2.metric("Sessions", total_sessions)
    k3.metric("Page transitions", total_navs)
    k4.metric("Avg session (mins)", round(avg_dur / 60.0, 1) if avg_dur else 0)

    st.divider()

    # -------------------------
    # Charts
    # -------------------------
    # Chart 1: sessions (logins) per day
    if logins is not None and not logins.empty:
        dfc = logins.copy()
        dfc["day"] = pd.to_datetime(dfc["day"])
        dfc["logins"] = pd.to_numeric(dfc["logins"], errors="coerce").fillna(0)

        ch = alt.Chart(dfc).mark_line(point=True).encode(
            x=alt.X("day:T", title="Day"),
            y=alt.Y("logins:Q", title="Sessions (logins)"),
            tooltip=[alt.Tooltip("day:T", title="Day"), alt.Tooltip("logins:Q", title="Sessions")]
        ).properties(height=260).interactive()

        st.markdown("### Sessions per day")
        st.altair_chart(ch, use_container_width=True)
    else:
        st.info("No login/session data in this window.")

    # Chart 2: top pages by transitions (optionally respecting user filter)
    if transitions is not None and not transitions.empty:
        top = transitions.copy()
        top["navigations"] = pd.to_numeric(top["navigations"], errors="coerce").fillna(0)
        if page_filter:
            top = top[top["page"] == page_filter]

        # Aggregate across days for the bar chart
        agg = top.groupby("page", as_index=False)["navigations"].sum().sort_values("navigations", ascending=False).head(20)

        if not agg.empty:
            bar = alt.Chart(agg).mark_bar().encode(
                x=alt.X("navigations:Q", title="Transitions"),
                y=alt.Y("page:N", sort="-x", title="Page"),
                tooltip=[alt.Tooltip("page:N", title="Page"), alt.Tooltip("navigations:Q", title="Transitions")]
            ).properties(height=420)

            st.markdown("### Top pages by navigation events")
            st.altair_chart(bar, use_container_width=True)

        # Optional: day-by-day heat style table (simple)
        with st.expander("Transitions (day x page)", expanded=False):
            td = top.copy()
            td["day"] = pd.to_datetime(td["day"])
            st.dataframe(td.sort_values(["day", "navigations"], ascending=[False, False]), use_container_width=True, hide_index=True)
    else:
        st.info("No page transition data in this window.")

    st.divider()

    # -------------------------
    # Drilldowns
    # -------------------------
    st.markdown("### Session summary")
    if sessions is None or sessions.empty:
        st.caption("No sessions in this window.")
    else:
        s = sessions.copy()
        # light formatting
        for col in ["duration_seconds", "distinct_pages", "transitions"]:
            if col in s.columns:
                s[col] = pd.to_numeric(s[col], errors="coerce").fillna(0).astype(int)

        st.dataframe(
            s,
            use_container_width=True,
            hide_index=True,
            column_config={
                "duration_seconds": st.column_config.NumberColumn("Duration (s)"),
                "distinct_pages": st.column_config.NumberColumn("Distinct pages"),
                "transitions": st.column_config.NumberColumn("Transitions"),
            }
        )

    st.markdown("### Raw events")
    if events is None or events.empty:
        st.caption("No events in this window (or filters exclude everything).")
    else:
        e = events.copy()
        st.dataframe(e, use_container_width=True, hide_index=True)

def page_admin_blotter():
    # --- Guard ---
    if st.session_state.get("role") != "admin":
        st.warning("Admin access required.")
        return

    st.subheader("Admin | Blotter")

    rep = admin_blotter_lines()
    if rep is None or rep.empty:
        st.info("No filled orders yet (or report is empty).")
        return

    df = rep.copy()

    # --- NORMALISE COLUMN NAMES ---
    # Map whatever the DB returns into canonical names used by this page.
    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    qty_col  = pick("Qty", "qty", "total_tonnes", "tonnes", "Tonnes")
    sell_col = pick("Sell Price", "sell_price", "sell_price_per_t", "SellPrice")
    base_col = pick("Base Price", "base_price", "base_price_per_t", "BasePrice")

    created_by_col = pick("created_by", "Created By", "trader", "Trader")
    prod_cat_col   = pick("Product Category", "product_category", "Category", "category", "product_group")
    product_col    = pick("Product", "product")
    location_col   = pick("Location", "location", "Region", "region")
    window_col     = pick("Delivery Window", "delivery_window", "Window", "window")
    supplier_col   = pick("Supplier", "supplier")

    rename_map = {}
    if qty_col and qty_col != "Qty": rename_map[qty_col] = "Qty"
    if sell_col and sell_col != "Sell Price": rename_map[sell_col] = "Sell Price"
    if base_col and base_col != "Base Price": rename_map[base_col] = "Base Price"

    if created_by_col and created_by_col != "created_by": rename_map[created_by_col] = "created_by"
    if prod_cat_col and prod_cat_col != "Product Category": rename_map[prod_cat_col] = "Product Category"
    if product_col and product_col != "Product": rename_map[product_col] = "Product"
    if location_col and location_col != "Location": rename_map[location_col] = "Location"
    if window_col and window_col != "Delivery Window": rename_map[window_col] = "Delivery Window"
    if supplier_col and supplier_col != "Supplier": rename_map[supplier_col] = "Supplier"

    if rename_map:
        df = df.rename(columns=rename_map)

    # --- Hard stop if core columns missing ---
    required = ["Qty", "Sell Price", "Base Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "Blotter cannot run because admin_blotter_lines() does not return the required columns: "
            + ", ".join(missing)
            + f"\n\nAvailable columns: {list(df.columns)}"
        )
        return

    # ---- Type fixes ----
    if "created_at_utc" in df.columns:
        df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], errors="coerce", utc=True)

    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0.0)
    df["Sell Price"] = pd.to_numeric(df["Sell Price"], errors="coerce").fillna(0.0)
    df["Base Price"] = pd.to_numeric(df["Base Price"], errors="coerce").fillna(0.0)

    # ---- Derived metrics ----
    df["sell_value"] = df["Sell Price"] * df["Qty"]
    df["base_value"] = df["Base Price"] * df["Qty"]
    df["gross_margin"] = df["sell_value"] - df["base_value"]
    df["gm_pct"] = (df["gross_margin"] / df["sell_value"]) * 100.0
    df["gm_pct"] = df["gm_pct"].where(df["sell_value"] != 0, 0.0)

    # ---- Filters ----
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])

    trader_col = "created_by" if "created_by" in df.columns else None
    cat_col = "Product Category" if "Product Category" in df.columns else None

    with f1:
        traders = ["ALL"] + (sorted(df[trader_col].dropna().unique().tolist()) if trader_col else [])
        trader = st.selectbox("Trader", traders) if trader_col else "ALL"

    with f2:
        cats = ["ALL"] + (sorted(df[cat_col].dropna().unique().tolist()) if cat_col else [])
        cat = st.selectbox("Product group", cats) if cat_col else "ALL"

    with f3:
        products = ["ALL"] + (sorted(df["Product"].dropna().unique().tolist()) if "Product" in df.columns else [])
        prod = st.selectbox("Product", products) if "Product" in df.columns else "ALL"

    with f4:
        locs = ["ALL"] + (sorted(df["Location"].dropna().unique().tolist()) if "Location" in df.columns else [])
        loc = st.selectbox("Location", locs) if "Location" in df.columns else "ALL"

    view = df.copy()
    if trader_col and trader != "ALL":
        view = view[view[trader_col] == trader]
    if cat_col and cat != "ALL":
        view = view[view[cat_col] == cat]
    if "Product" in view.columns and prod != "ALL":
        view = view[view["Product"] == prod]
    if "Location" in view.columns and loc != "ALL":
        view = view[view["Location"] == loc]

    st.divider()

    # ---- KPI strip ----
    sell_value = float(view["sell_value"].sum())
    base_value = float(view["base_value"].sum())
    gm = float(view["gross_margin"].sum())
    tonnes = float(view["Qty"].sum())
    gm_pct = (gm / sell_value * 100.0) if sell_value else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sell value", f"£{sell_value:,.0f}")
    k2.metric("Base value", f"£{base_value:,.0f}")
    k3.metric("Gross margin", f"£{gm:,.0f}")
    k4.metric("GM %", f"{gm_pct:.2f}%")
    k5.metric("Tonnes", f"{tonnes:,.1f} t")

    # ---- Grouping ----
    st.markdown("### Grouped summary")
    group_options = []
    for c in ["created_by", "Product Category", "Product", "Location", "Delivery Window", "Supplier"]:
        if c in view.columns:
            group_options.append(c)

    default_groups = [c for c in ["created_by", "Location"] if c in group_options]
    group_by = st.multiselect("Group by", options=group_options, default=default_groups)

    if not group_by:
        st.info("Select at least one field in 'Group by'.")
        return

    agg = (
        view.groupby(group_by, dropna=False)
        .agg(
            Qty=("Qty", "sum"),
            sell_value=("sell_value", "sum"),
            base_value=("base_value", "sum"),
            gross_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )
    agg["gm_pct"] = (agg["gross_margin"] / agg["sell_value"]) * 100.0
    agg["gm_pct"] = agg["gm_pct"].where(agg["sell_value"] != 0, 0.0)

    agg = agg.sort_values("gross_margin", ascending=False)

    st.dataframe(
        agg,
        use_container_width=True,
        hide_index=True,
        column_config={
            "sell_value": st.column_config.NumberColumn("Sell value", format="£%.0f"),
            "base_value": st.column_config.NumberColumn("Base value", format="£%.0f"),
            "gross_margin": st.column_config.NumberColumn("Gross margin", format="£%.0f"),
            "gm_pct": st.column_config.NumberColumn("GM %", format="%.2f"),
        },
    )

    st.markdown("### Detail")
    st.dataframe(view, use_container_width=True, hide_index=True)
