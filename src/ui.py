import time
import streamlit as st
import pandas as pd

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


def page_trader_pricing():
    st.subheader("Trader — Pricing")

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
    st.write({
        "Base cost (sell)": res["base_cost"],
        "Small-lot total": res["lot_charge_total"],
        "Grand total": res["total"],
    })

    st.divider()
    st.markdown("### Checkout")

    trader_note = st.text_area("Order note (optional)", placeholder="Customer/account, terms, anything relevant.")

    if st.button("Submit order to Admin", type="primary", use_container_width=True):
        # Build allocation lines for order snapshot
        # Need Base Price and Unit too -> look up from df (which still has base Price and Unit)
        alloc_lines = []
        for r in res["allocation"]:
            prod = r["Product"]
            loc = r["Location"]
            win = r["Delivery Window"]
            sup = r["Supplier"]
            qty = float(r["Qty"])

            # Lookup base price + unit from supplier snapshot df
            match = df[
                (df["Product"] == prod) &
                (df["Location"] == loc) &
                (df["Delivery Window"] == win)
            ].copy()

            # match is across suppliers; find the chosen supplier row
            match = match[match["Supplier"] == sup]
            if match.empty:
                st.error(f"Internal error: could not find base row for {prod}/{loc}/{win}/{sup}")
                return

            base_price = float(match.iloc[0]["Price"])
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
            # clear basket + last optim
            st.session_state.basket = []
            st.session_state.basket_created_at = time.time()
            st.session_state.last_optim_result = None
            st.session_state.last_optim_snapshot = None

            st.success(f"Order submitted: {order_id[:8]}")
        except Exception as e:
            st.error(str(e))


def page_trader_orders():
    st.subheader("Trader — Orders")

    df = list_orders_for_user(st.session_state.user)
    if df.empty:
        st.info("No orders yet.")
        return

    # Basic filters
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

    st.markdown("### Order header")
    st.write(header)

    st.markdown("### Lines")
    st.dataframe(lines, use_container_width=True, hide_index=True)

    # Totals + margin display to trader: show totals only (no margin)
    base = float((lines["Sell Price"] * lines["Qty"]).sum())
    st.write({"Sell value": base})

    st.markdown("### Timeline")
    st.dataframe(actions[["action_type","action_at_utc","action_by"]], use_container_width=True, hide_index=True)

    st.divider()

    # Trader actions
    if header["status"] in ("PENDING", "COUNTERED"):
        if st.button("Cancel order", use_container_width=True):
            try:
                trader_cancel_order(order_id, st.session_state.user)
                st.success("Order cancelled.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if header["status"] == "COUNTERED":
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Accept counter", type="primary", use_container_width=True):
                try:
                    trader_accept_counter(order_id, st.session_state.user)
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

    st.subheader("Admin — Pricing")

    # Settings
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

    # Tiers
    st.markdown("### Small-lot tiers (global)")
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

    # Margins
    st.markdown("### Admin margins (hidden from traders)")
    mdf = list_margins(active_only=True)
    if mdf.empty:
        st.info("No active margins set.")
    else:
        show = mdf[["margin_id","scope_type","scope_value","margin_per_t","created_at_utc","created_by"]].copy()
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
            add_margin(scope_type, scope_value, float(margin_per_t), st.session_state.get("user","unknown"))
            st.success("Margin added.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()

    # Upload supplier prices
    st.markdown("### Upload supplier prices (SUPPLIER_PRICES)")
    up = st.file_uploader("Upload Excel", type=["xlsx"])
    if up:
        content = up.read()
        try:
            df = load_supplier_sheet(content)
            st.success("Validated. Preview:")
            st.dataframe(df, use_container_width=True, hide_index=True)
            if st.button("Publish supplier snapshot", type="primary", use_container_width=True):
                sid = publish_supplier_snapshot(df, st.session_state.get("user","unknown"), content)
                st.success(f"Published supplier snapshot: {sid}")
                st.rerun()
        except Exception as e:
            st.error(str(e))


def page_admin_orders():
    if st.session_state.get("role") != "admin":
        st.warning("Admin access required.")
        return

    st.subheader("Admin — Order blotter")

    status = st.selectbox("Status filter", ["ALL","PENDING","COUNTERED","CONFIRMED","FILLED","REJECTED","CANCELLED"])
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

    st.markdown("### Order header")
    st.write(header)

    st.markdown("### Lines")
    st.dataframe(lines, use_container_width=True, hide_index=True)

    # Admin margin view
    gross_margin = float(((lines["Sell Price"] - lines["Base Price"]) * lines["Qty"]).sum())
    sell_value = float((lines["Sell Price"] * lines["Qty"]).sum())
    base_value = float((lines["Base Price"] * lines["Qty"]).sum())
    st.write({"Sell value": sell_value, "Base value": base_value, "Gross margin": gross_margin})

    st.markdown("### Timeline")
    st.dataframe(actions[["action_type","action_at_utc","action_by"]], use_container_width=True, hide_index=True)

    st.divider()

    # Actions
    if header["status"] in ("PENDING","COUNTERED"):
        st.markdown("### Counter / Confirm / Reject")

        admin_note = st.text_area("Admin note (optional)", value=header.get("admin_note","") or "")

        # Allow editing sell prices (MVP)
        editable = lines[["line_no","Product","Location","Delivery Window","Qty","Supplier","Base Price","Sell Price"]].copy()
        edited = st.data_editor(editable, use_container_width=True, hide_index=True, num_rows="fixed")

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("Confirm as-is", type="primary", use_container_width=True):
                try:
                    admin_confirm_order(order_id, st.session_state.user)
                    st.success("Order CONFIRMED.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        with c2:
            if st.button("Send counter", use_container_width=True):
                try:
                    admin_counter_order(order_id, st.session_state.user, edited_lines=edited, admin_note=admin_note)
                    st.success("Counter sent. Status = COUNTERED.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        with c3:
            if st.button("Reject", use_container_width=True):
                try:
                    admin_reject_order(order_id, st.session_state.user, admin_note=admin_note)
                    st.success("Order REJECTED.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if header["status"] == "CONFIRMED":
        st.markdown("### Fill")
        if st.button("Mark FILLED", type="primary", use_container_width=True):
            try:
                admin_mark_filled(order_id, st.session_state.user)
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

    # Apply margins but hide them
    margins = get_effective_margins()
    df = apply_margins(df, margins)
    df["Price"] = df["Sell Price"]
    df = df.drop(columns=["Sell Price"], errors="ignore")

    q = st.text_input("Search")
    if q:
        ql = q.lower()
        df = df[df.apply(lambda r: any(ql in str(v).lower() for v in r.values), axis=1)]

    st.dataframe(df, use_container_width=True, hide_index=True)



