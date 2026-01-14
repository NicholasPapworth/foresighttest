import streamlit as st
from src.db import init_db
from src.auth import require_login
from src.ui import (
    render_header,
    page_trader_pricing, page_trader_orders,
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
    "Orders": page_trader_orders,
    "History": page_history,
}

pages_admin = {
    "Admin Pricing": page_admin_pricing,
    "Admin Orders": page_admin_orders,
}

with st.sidebar:
    st.markdown(f"**User:** {st.session_state.user}")
    st.markdown(f"**Role:** {st.session_state.role}")
    st.divider()

    if st.session_state.role == "admin":
        choice = st.radio("Navigation", list(pages_trader.keys()) + list(pages_admin.keys()))
        if choice in pages_trader:
            pages_trader[choice]()
        else:
            pages_admin[choice]()
    else:
        choice = st.radio("Navigation", list(pages_trader.keys()))
        pages_trader[choice]()


