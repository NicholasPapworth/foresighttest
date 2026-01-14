import streamlit as st
from src.db import init_db
from src.auth import require_login
from src.ui import (
    render_header,
    page_trader_pricing, page_trader_orders, page_trader_best_prices,
    page_admin_pricing, page_admin_orders,
    page_history
)

st.set_page_config(page_title="Foresight Pricing", layout="wide")
init_db()

if not require_login():
    st.stop()

render_header()

pages_trader = {
    "Pricing": page_trader_pricing,
    "Best Prices": page_trader_best_prices,
    "Orders": page_trader_orders,
    "History": page_history,
}

pages_admin = {
    "Admin Pricing": page_admin_pricing,
    "Admin Orders": page_admin_orders,
}

# --- Sidebar: navigation ONLY ---
with st.sidebar:
    st.markdown(f"**User:** {st.session_state.user}")
    st.markdown(f"**Role:** {st.session_state.role}")
    st.divider()

    if st.session_state.role == "admin":
        nav_items = list(pages_trader.keys()) + list(pages_admin.keys())
    else:
        nav_items = list(pages_trader.keys())

    choice = st.radio("Navigation", nav_items)

# --- Main area: render the selected page OUTSIDE the sidebar ---
if choice in pages_trader:
    pages_trader[choice]()
else:
    pages_admin[choice]()



