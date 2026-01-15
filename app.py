import streamlit as st
from src.db import init_db
from src.auth import require_login
from src.ui import (
    show_boot_splash,
    render_header,
    render_presence_panel,
    page_trader_pricing,
    page_trader_best_prices,
    page_trader_orders,
    page_admin_pricing,
    page_admin_orders,
    page_admin_blotter,
    page_history,
)

st.set_page_config(page_title="Foresight Pricing", layout="wide")
init_db()

if not require_login():
    st.stop()

show_boot_splash(video_path="assets/boot.mp4", seconds=4.8)

render_header()

pages = {
    "Trader | Pricing": page_trader_pricing,
    "Trader | Best Prices": page_trader_best_prices,
    "Trader | Orders": page_trader_orders,
    "History": page_history,
}

if st.session_state.get("role") == "admin":
    pages.update({
        "Admin | Pricing": page_admin_pricing,
        "Admin | Orders": page_admin_orders,
        "Admin | Blotter": page_admin_blotter,
    })

with st.sidebar:
    st.markdown("### Navigation")
    choice = st.radio("", list(pages.keys()), key="nav_choice")

    st.divider()
    render_presence_panel(current_page_name=choice)  # explicit kwarg is clearer

pages[choice]()






