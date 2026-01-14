import time
import streamlit as st
import pandas as pd
import uuid
from datetime import datetime, timezone
from src.db import presence_heartbeat, list_online_users

from src.db import (
    get_settings, set_setting,
    get_small_lot_tiers, save_small_lot_tiers,
    latest_supplier_snapshot, list_supplier_snapshots,
    load_supplier_prices, publish_supplier_snapshot,
    add_margin, list_margins, deactivate_margin, get_effective_margins,
    create_order_from_allocation, list_orders_for_user, list_orders_admin,
    get_order_header, get_order_lines, get_order_actions,
    trader_cancel_order, trader_accept_counter,
    admin_counter_order, admin_confirm_order, admin_reject_order, admin_mark_filled,
    admin_margin_report
)
from src.validation import load_supplier_sheet
from src.optimizer import optimise_basket
from src.pricing import apply_margins

LOGO_PATH = "assets/logo.svg"


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
        snap = latest_supplier_snapshot()
        if snap:
            sid, ts, by = snap
            st.caption(f"Latest supplier snapshot: {ts} UTC\nPublished by: {by}")
        else:
            st.caption("No supplier snapshot published yet.")
    st.divider()


def _get_latest_prices_df():
    snap = latest_supplier_snapshot()
    if not snap:
        return None, None
    sid, ts, by = snap
    df = load_supplier_prices(sid)
    return sid, df

def _ensure_basket():
    if "basket" not in st.session_state:
        st.session_state.basket = []
        st.session_state.basket_created_at = time.time()

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


def page_trader_pricing():
    st.subheader("Trader | Pricing")

    sid, df = _get_latest_prices_df()
    if df is None:
        st.warning("No supplier snapshot available. Admin must publish one.")
        return

    settings = get_settings()
    timeout_min = int(settings.get("basket_timeout_minutes", "20"))
    tiers = get_small_lot_tiers()

    # Apply margins
    margins = get_effective_margins()
    df = apply_margins(df, margins)

    # Basket state
    if "basket" not in st.session_state:
        st.session_state.basket = []
        st.session_state.basket_created_at = time.time()

    # Expiry
    age_sec = time.time() - st.session_state.basket_created_at
    if age_sec > timeout_min * 60:
        st.session_state.basket = []
        st.session_state.basket_created_at = time.time()
        st.info("Basket expired and has been cleared.")

    st.caption(f"Using supplier snapshot: {sid[:8]} | Basket timeout: {timeout_min} min")

    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    with c1:
        product = st.selectbox("Product", sorted(df["Product"].dropna().unique().tolist()))
    with c2:
        location = st.selectbox("Location", sorted(df["Location"].dropna().unique().tolist()))
    with c3:
        window = st.selectbox("Delivery Window", sorted(df["Delivery Window"].dropna().unique().tolist()))
    with c4:
        qty = st.number_input("Qty (t)", min_value=0.0, value=10.0, step=1.0)

    if st.button("Add to basket", use_container_width=True):
        st.session_state.basket.append({
            "Product": product,
            "Location": location,
            "Delivery Window": window,
            "Qty": float(qty),
        })
        st.rerun()

    st.divider()

    # Basket view
    if not st.session_state.basket:
        st.info("Basket is empty.")
        return

    bdf = pd.DataFrame(st.session_state.basket)
    st.markdown("### Basket")
    st.dataframe(bdf, use_container_width=True, hide_index=True)

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Clear basket", use_container_width=True):
            st.session_state.basket = []
            st.session_state.basket_created_at = time.time()
            st.rerun()
    with colB:
        if st.button("Optimise", type="primary", use_container_width=True):
            sell_prices = df[["Supplier", "Product", "Location", "Delivery Window", "Sell Price"]].rename(
                columns={"Sell Price": "Price"}
            )

            res = optimise_basket(
                supplier_prices=sell_prices,
                basket=st.session_state.basket,
                tiers=tiers
            )

            if not res.get("ok"):
                st.error(res.get("error", "Unknown error"))
                return

            st.session_state.last_optim_result = res
            st.session_state.last_optim_snapshot = sid
            st.success("Optimisation complete. Review below.")
            st.rerun()

    # Show optimisation result if available
    if "last_optim_result" not in st.session_state or st.session_state.get("last_optim_snapshot") != sid:
        st.info("Optimise to generate an allocation before checkout.")
        return

    res = st.session_state.last_optim_result

    st.markdown("### Optimal allocation (Sell prices)")
    alloc_df = pd.DataFrame(res["allocation"])
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    if res.get("lot_charges"):
        st.markdown("### Small-lot charges")
        st.dataframe(pd.DataFrame(res["lot_charges"]), use_container_width=True, hide_index=True)

    st.markdown("### Totals")
    c1, c2, c3 = st.columns(3)
    c1.metric("Base cost (sell)", f"£{float(res['base_cost']):,.2f}")
    c2.metric("Small-lot total", f"£{float(res['lot_charge_total']):,.2f}")
    c3.metric("Grand total", f"£{float(res['total']):,.2f}")

    st.divider()
    st.markdown("### Checkout")

    trader_note = st.text_area("Order note (optional)", placeholder="Customer/account, terms, anything relevant.")

    if st.button("Submit order to Admin", type="primary", use_container_width=True):
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

            sell_price = float(match.iloc[0]["Sell Price"])
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
            })

        try:
            order_id = create_order_from_allocation(
                created_by=st.session_state.user,
                supplier_snapshot_id=sid,
                allocation_lines=alloc_lines,
                trader_note=trader_note
            )
            st.session_state.basket = []
            st.session_state.basket_created_at = time.time()
            st.session_state.last_optim_result = None
            st.session_state.last_optim_snapshot = None

            st.success(f"Order submitted: {order_id[:8]}")
        except Exception as e:
            st.error(str(e))


def page_trader_orders():
    st.subheader("Trader | Orders")

    df = list_orders_for_user(st.session_state.user)
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
                trader_cancel_order(order_id, st.session_state.user, expected_version=header["version"])
                st.success("Order cancelled.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if header["status"] == "COUNTERED":
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Accept counter", type="primary", use_container_width=True):
                try:
                    trader_accept_counter(order_id, st.session_state.user, expected_version=header["version"])
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

    settings = get_settings()
    timeout = st.number_input(
        "Basket timeout (minutes)",
        min_value=1,
        value=int(settings.get("basket_timeout_minutes", "20"))
    )
    if st.button("Save settings", use_container_width=True):
        set_setting("basket_timeout_minutes", str(timeout))
        st.success("Settings saved.")

    st.divider()

    st.markdown("### Small-lot tiers")
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
        column_config={
            "min_t": st.column_config.NumberColumn("Min t", min_value=0.0, step=0.1),
            "max_t": st.column_config.NumberColumn("Max t", min_value=0.0, step=0.1),
            "charge_per_t": st.column_config.NumberColumn("Charge (£/t)", min_value=0.0, step=0.1),
            "active": st.column_config.CheckboxColumn("Active"),
        }
    )

    if st.button("Save tiers", type="primary", use_container_width=True):
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

    st.divider()

    st.markdown("### Admin margins")
    mdf = list_margins(active_only=True)
    if mdf.empty:
        st.info("No active margins set.")
    else:
        show = mdf[["margin_id", "scope_type", "scope_value", "margin_per_t", "created_at_utc", "created_by"]].copy()
        show = show.rename(columns={"margin_per_t": "Margin (£/t)"})
        st.dataframe(show, use_container_width=True, hide_index=True)

        mid = st.number_input("Deactivate margin_id", min_value=0, value=0, step=1)
        if st.button("Deactivate selected margin", use_container_width=True):
            if mid <= 0:
                st.error("Enter a valid margin_id.")
            else:
                deactivate_margin(int(mid))
                st.success(f"Deactivated margin_id={int(mid)}")
                st.rerun()

    st.markdown("#### Add new margin")
    scope_type = st.selectbox("Scope", ["category", "product"])
    scope_value = st.text_input("Category/Product name (exact match)")
    margin_per_t = st.number_input("Margin (£/t)", value=0.0, step=0.5)
    if st.button("Add margin", type="primary", use_container_width=True):
        try:
            add_margin(scope_type, scope_value, float(margin_per_t), st.session_state.get("user", "unknown"))
            st.success("Margin added.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()

    st.markdown("### Upload supplier prices (SUPPLIER_PRICES)")
    up = st.file_uploader("Upload Excel", type=["xlsx"])
    if up:
        content = up.read()
        try:
            df = load_supplier_sheet(content)
            st.success("Validated. Preview:")
            st.dataframe(df, use_container_width=True, hide_index=True)
            if st.button("Publish supplier snapshot", type="primary", use_container_width=True):
                sid = publish_supplier_snapshot(df, st.session_state.get("user", "unknown"), content)
                st.success(f"Published supplier snapshot: {sid}")
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
                    admin_confirm_order(order_id, st.session_state.user, expected_version=header["version"])
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
                        st.session_state.user,
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
                    admin_reject_order(order_id, st.session_state.user, admin_note=admin_note, expected_version=header["version"])
                    st.success("Order REJECTED.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if header["status"] == "CONFIRMED":
        st.markdown("### Fill")
        if st.button("Mark FILLED", type="primary", use_container_width=True):
            try:
                admin_mark_filled(order_id, st.session_state.user, expected_version=header["version"])
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

    snaps = list_supplier_snapshots()
    if snaps.empty:
        st.info("No snapshots yet.")
        return

    snaps = snaps.copy()
    snaps["label"] = snaps["published_at_utc"] + " | " + snaps["published_by"] + " | " + snaps["snapshot_id"].str[:8]
    label = st.selectbox("Select snapshot", snaps["label"].tolist())
    sid = snaps.loc[snaps["label"] == label, "snapshot_id"].iloc[0]

    df = load_supplier_prices(sid)

    margins = get_effective_margins()
    df = apply_margins(df, margins)
    df["Price"] = df["Sell Price"]
    df = df.drop(columns=["Sell Price"], errors="ignore")

    q = st.text_input("Search")
    if q:
        ql = q.lower()
        df = df[df.apply(lambda r: any(ql in str(v).lower() for v in r.values), axis=1)]

    st.dataframe(df, use_container_width=True, hide_index=True)

def page_trader_best_prices():
    st.subheader("Trader | Best Prices")

    sid, df = _get_latest_prices_df()
    if df is None:
        st.warning("No supplier snapshot available. Admin must publish one.")
        return

    # Apply margins so Best Price reflects SELL (incl margin)
    margins = get_effective_margins()
    df = apply_margins(df, margins)

    board = _best_prices_board(df)

    # --- Filters ---
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])

    with f1:
        cats = ["ALL"] + sorted(board["Product Category"].unique().tolist())
        cat = st.selectbox("Product Category", cats)

    with f2:
        prods = ["ALL"] + sorted(board["Product"].unique().tolist())
        prod = st.selectbox("Product", prods)

    with f3:
        locs = ["ALL"] + sorted(board["Location"].unique().tolist())
        loc = st.selectbox("Location", locs)

    with f4:
        wins = ["ALL"] + sorted(board["Delivery Window"].unique().tolist())
        win = st.selectbox("Delivery Window", wins)

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

    # --- Basket expiry (same behaviour as pricing page) ---
    _ensure_basket()
    settings = get_settings()
    timeout_min = int(settings.get("basket_timeout_minutes", "20"))
    age_sec = time.time() - st.session_state.basket_created_at
    if age_sec > timeout_min * 60:
        st.session_state.basket = []
        st.session_state.basket_created_at = time.time()
        st.info("Basket expired and has been cleared.")

    # --- One-click add to basket ---
    st.markdown("### Add to basket from board")
    qty = st.number_input("Qty (t) for selected lines", min_value=0.0, value=10.0, step=1.0)

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
        key="best_prices_board_editor",
    )

    if st.button("Add selected lines to basket", type="primary", use_container_width=True):
        selected = edited[edited["Add"] == True].copy()
        if selected.empty:
            st.warning("Tick at least one line.")
        else:
            for _, r in selected.iterrows():
                st.session_state.basket.append({
                    "Product": r["Product"],
                    "Location": r["Location"],
                    "Delivery Window": r["Window"],
                    "Qty": float(qty),
                })
            st.success(f"Added {len(selected)} line(s) to basket.")
            st.info("Go to Trader | Pricing to optimise and submit the order.")
            st.rerun()

    st.divider()

    # Show ONE table only (avoid the duplication you complained about)
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

    rep = admin_margin_report()
    if rep is None or rep.empty:
        st.info("No filled orders yet (or report is empty).")
        return

    df = rep.copy()

        # --- NORMALISE COLUMN NAMES ---
    # admin_margin_report() might return snake_case or other names.
    # We map whatever exists into the canonical names this page expects.

    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    qty_col = pick("Qty", "qty", "qty_t", "tonnes", "Tonnes")
    sell_col = pick("Sell Price", "sell_price", "sell_price_per_t", "sell_price_gbp", "sell_price_per_t_gbp")
    base_col = pick("Base Price", "base_price", "base_price_per_t", "base_price_gbp", "base_price_per_t_gbp")

    # Optional dims (only used for filtering/grouping)
    created_by_col = pick("created_by", "Created By", "trader", "Trader")
    prod_cat_col = pick("Product Category", "product_category", "Category", "category", "Product group", "product_group")
    product_col = pick("Product", "product")
    location_col = pick("Location", "location", "Region", "region")
    window_col = pick("Delivery Window", "delivery_window", "Window", "window")
    supplier_col = pick("Supplier", "supplier")

    # Rename into canonical names used below
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

    # Hard stop if core columns are still missing
    required = ["Qty", "Sell Price", "Base Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "Blotter cannot run because admin_margin_report() does not return the required columns: "
            + ", ".join(missing)
            + f"\n\nAvailable columns: {list(df.columns)}"
        )
        return

    # ---- Type fixes ----
    if "created_at_utc" in df.columns:
        df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], errors="coerce", utc=True)

    for c in ["Qty", "Base Price", "Sell Price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Type fixes ----
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0.0)
    df["Sell Price"] = pd.to_numeric(df["Sell Price"], errors="coerce").fillna(0.0)
    df["Base Price"] = pd.to_numeric(df["Base Price"], errors="coerce").fillna(0.0)

    # ---- Derived metrics ----
    df["sell_value"] = df["Sell Price"] * df["Qty"]
    df["base_value"] = df["Base Price"] * df["Qty"]
    df["gross_margin"] = df["sell_value"] - df["base_value"]
    df["gm_pct"] = (df["gross_margin"] / df["sell_value"]) * 100.0
    df["gm_pct"] = df["gm_pct"].where(df["sell_value"] != 0, 0.0)

    # ---- Filters (top row) ----
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])

    # Column names (adjust here if yours differ)
    trader_col = "created_by" if "created_by" in df.columns else None
    cat_col = "Product Category" if "Product Category" in df.columns else ("Category" if "Category" in df.columns else None)

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
    for c in ["created_by", cat_col, "Product", "Location", "Delivery Window", "Supplier"]:
        if c and c in view.columns:
            group_options.append(c)

    group_by = st.multiselect("Group by", options=group_options, default=[c for c in ["created_by", "Location"] if c in group_options])

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

