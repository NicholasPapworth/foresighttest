import time
import streamlit as st
import pandas as pd

from src.db import (
    get_settings, set_setting,
    get_small_lot_tiers, save_small_lot_tiers,
    latest_supplier_snapshot, list_supplier_snapshots,
    load_supplier_prices, publish_supplier_snapshot
)
from src.validation import load_supplier_sheet
from src.optimizer import optimise_basket

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


def page_admin():
    if st.session_state.get("role") != "admin":
        st.warning("Admin access required.")
        return

    st.subheader("Admin")

    # --- Settings (basket timeout) ---
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

    # --- Tier editor ---
    st.markdown("### Small-lot tiers (global)")
    tiers = get_small_lot_tiers()

    # Ensure expected columns exist even if tiers table is empty
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

            # Normalize active checkbox to 0/1
            if "active" not in edited2.columns:
                edited2["active"] = 1
            edited2["active"] = edited2["active"].apply(lambda x: 1 if bool(x) else 0)

            # Ensure expected numeric columns exist
            for col in ["min_t", "charge_per_t"]:
                if col not in edited2.columns:
                    raise ValueError(f"Missing column '{col}'.")

            if "max_t" not in edited2.columns:
                edited2["max_t"] = None

            save_small_lot_tiers(edited2[["min_t", "max_t", "charge_per_t", "active"]])
            st.success("Tiers saved.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()

    # --- Upload supplier prices ---
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

    q = st.text_input("Search")
    if q:
        ql = q.lower()
        df = df[df.apply(lambda r: any(ql in str(v).lower() for v in r.values), axis=1)]

    st.dataframe(df, use_container_width=True, hide_index=True)


def page_trader():
    st.subheader("Trader")

    sid, df = _get_latest_prices_df()
    if df is None:
        st.warning("No supplier snapshot available. Admin must publish one.")
        return

    settings = get_settings()
    timeout_min = int(settings.get("basket_timeout_minutes", "20"))
    tiers = get_small_lot_tiers()
