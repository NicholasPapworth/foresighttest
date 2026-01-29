import time
import streamlit as st
import pandas as pd
import uuid
import base64
import streamlit.components.v1 as components
from pathlib import Path
from datetime import datetime, timezone
from src.db import presence_heartbeat, list_online_users

from src.db import (
    get_settings, set_setting,
    get_small_lot_tiers, save_small_lot_tiers,

    # Fertiliser snapshot functions (existing)
    latest_supplier_snapshot, list_supplier_snapshots,
    load_supplier_prices, publish_supplier_snapshot,

    # Seed snapshot functions
    latest_seed_snapshot, list_seed_snapshots,
    load_seed_prices, publish_seed_snapshot, list_seed_treatments, save_seed_treatments,

    list_fert_delivery_options, save_fert_delivery_options,

    add_margin, list_margins, deactivate_margin, get_effective_margins,
    create_order_from_allocation, list_orders_for_user, list_orders_admin,
    get_order_header, get_order_lines, get_order_actions,
    trader_cancel_order, trader_accept_counter,
    admin_counter_order, admin_confirm_order, admin_reject_order, admin_mark_filled,
    admin_blotter_lines,
    admin_margin_report
)

from src.validation import load_supplier_sheet, load_seed_sheet
from src.optimizer import optimise_basket
from src.pricing import apply_margins

LOGO_PATH = "assets/logo.svg"

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
                    dm_lookup[key] = str(r.get("Delivery Method", "Delivered"))

        def _dm_delta(row):
            key = (str(row.get("Product", "")), str(row.get("Location", "")), str(row.get("Delivery Window", "")))
            dm = dm_lookup.get(key, "Delivered")
            return float(delivery_delta_map.get(dm, 0.0))

        work["Delivery £/t"] = work.apply(_dm_delta, axis=1)

    # ---- Addons per line (seed only) ----
    work["Addons £/t"] = 0.0
    # Optimiser in your screenshots uses "Addons £/t"
    if "Addons £/t" in work.columns:
        work["Addons £/t"] = pd.to_numeric(work["Addons £/t"], errors="coerce").fillna(0.0)
    else:
        # fallback if optimiser output differs
        for cand in ["addons_per_t", "Addons_per_t", "Addons/t", "Addons £ per t"]:
            if cand in work.columns:
                work["Addons £/t"] = pd.to_numeric(work[cand], errors="coerce").fillna(0.0)
                break

    # ---- All-in per line ----
    work["All-in £/t"] = work["Base £/t"] + work["Lot £/t"] + work["Delivery £/t"] + work["Addons £/t"]

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
        "all_in_value": float(work["Line total"].sum()),
    }
    return work, totals


def page_trader_pricing():
    st.subheader("Trader | Pricing")

    tab_f, tab_s = st.tabs(["Fertiliser", "Seed"])

    with tab_f:
        _page_trader_pricing_impl(book_code="fert")

    with tab_s:
        _page_trader_pricing_impl(book_code="seed")

def _page_trader_pricing_impl(book_code: str):
    sid, df = _get_latest_prices_df_for(book_code)
    if df is None:
        st.warning("No supplier snapshot available. Admin must publish one.")
        return

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
            price_col = "Sell Price" if "Sell Price" in df.columns else "Price"
            sell_prices = df[["Supplier", "Product", "Location", "Delivery Window", price_col]].rename(
                columns={price_col: "Price"}
            )

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

            if not res.get("ok"):
                st.error(res.get("error", "Unknown error"))
                return

            st.session_state[last_optim_key] = res
            st.session_state[last_optim_snap_key] = sid
            st.success("Optimisation complete. Review below.")
            st.rerun()

    # Show optimisation result if available
    if last_optim_key not in st.session_state or st.session_state.get(last_optim_snap_key) != sid:
        st.info("Optimise to generate an allocation before checkout.")
        return

    res = st.session_state[last_optim_key]

    st.markdown("### Optimal Allocation's")

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
    for c in ["Product","Location","Delivery Window","Qty","Supplier","Base £/t","Lot £/t","Delivery £/t","Addons £/t","All-in £/t","Line total"]:
        if c in quote_lines_df.columns:
            show_cols.append(c)
    
    quote_view = quote_lines_df[show_cols].copy()

    # Optional: format money columns nicely
    money_cols = ["Base £/t", "Lot £/t", "Delivery £/t", "Addons £/t", "All-in £/t", "Line total"]
    fmt = {}
    for c in money_cols:
        if c in quote_view.columns:
            fmt[c] = "£{:,.2f}".format
    if "Qty" in quote_view.columns:
        fmt["Qty"] = "{:,.2f}".format

    styler = quote_view.style.format(fmt)

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

            match = df[
                (df["Product"] == prod) &
                (df["Location"] == loc) &
                (df["Delivery Window"] == win)
            ].copy()

            match = match[match["Supplier"] == sup]
            if match.empty:
                st.error(f"Internal error: could not find base row for {prod}/{loc}/{win}/{sup}")
                return

            base_col = "Price" if "Price" in match.columns else ("Base Price" if "Base Price" in match.columns else None)
            if base_col is None:
                st.error("Internal error: missing base price column (expected 'Price' or 'Base Price').")
                return
            base_price = float(match.iloc[0][base_col])

            sell_price_base = float(match.iloc[0]["Sell Price"]) if "Sell Price" in match.columns else float(match.iloc[0]["Price"])

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
                    ]
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
            sell_price = sell_price_base + delivery_delta + lot_charge_per_t
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

    tab_f, tab_s = st.tabs(["Fertiliser", "Seed"])
    with tab_f:
        _page_admin_pricing_impl(book_code="fert")
    with tab_s:
        _page_admin_pricing_impl(book_code="seed")

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
    mdf = list_margins(active_only=True)
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
            add_margin(scope_type, scope_value, float(margin_per_t), st.session_state.get("user", "unknown"))
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
            margins = get_effective_margins()
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
                    margins = get_effective_margins()
            
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


def page_history():
    st.subheader("History")

    tab_f, tab_s = st.tabs(["Fertiliser", "Seed"])
    with tab_f:
        _page_history_impl(book_code="fert")
    with tab_s:
        _page_history_impl(book_code="seed")

def _page_history_impl(book_code: str):
    snaps = BOOKS_BY_CODE[book_code]["list_snapshots"]()
    if snaps.empty:
        st.info("No snapshots yet.")
        return

    snaps = snaps.copy()
    snaps["label"] = snaps["published_at_utc"] + " | " + snaps["published_by"] + " | " + snaps["snapshot_id"].str[:8]
    label = st.selectbox("Select snapshot", snaps["label"].tolist(), key=_ss_key(book_code, "hist_select"))
    sid = snaps.loc[snaps["label"] == label, "snapshot_id"].iloc[0]

    df = BOOKS_BY_CODE[book_code]["load_prices"](sid)

    # Snapshot already contains Sell Price in DB; show both Base and Sell
    # (No runtime margin application)

    q = st.text_input("Search", key=_ss_key(book_code, "hist_search"))
    if q:
        ql = q.lower()
        df = df[df.apply(lambda r: any(ql in str(v).lower() for v in r.values), axis=1)]

    st.dataframe(df, use_container_width=True, hide_index=True)

def page_trader_best_prices():
    st.subheader("Trader | Best Prices")

    tab_f, tab_s = st.tabs(["Fertiliser", "Seed"])
    with tab_f:
        _page_trader_best_prices_impl(book_code="fert")
    with tab_s:
        _page_trader_best_prices_impl(book_code="seed")

def _page_trader_best_prices_impl(book_code: str):
    sid, df = _get_latest_prices_df_for(book_code)
    if df is None:
        st.warning("No supplier snapshot available. Admin must publish one.")
        return

    board = _best_prices_board(df)

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
# ---------------------------
# Presence (sidebar widget)
# ---------------------------

def _ensure_session_id():
    if "presence_session_id" not in st.session_state:
        st.session_state.presence_session_id = str(uuid.uuid4())


def _utc_parse(ts: str) -> datetime:
    # Supports "2026-01-14T15:12:34+00:00"
    return datetime.fromisoformat(ts)


def render_presence_panel(current_page_name: str, *, refresh_ms: int = 10_000):
    """
    Sidebar presence panel only:
    - heartbeats current user
    - shows online users
    - toasts when users come/go (diff vs last refresh)
    """
    _ensure_session_id()

    user = st.session_state.get("user", "") or ""
    role = st.session_state.get("role", "") or ""
    session_id = st.session_state.presence_session_id

    # Heartbeat every run
    presence_heartbeat(
        user=user,
        role=role,
        page=current_page_name,
        session_id=session_id
    )

    # Optional auto-refresh (install streamlit-autorefresh for true "live" feel)
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_ms, key="presence_autorefresh")
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

    st.session_state.presence_prev_online = list(now)

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
            "Page": st.column_config.TextColumn("Page"),
            "Last seen": st.column_config.TextColumn("Last seen"),
        },
    )

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

