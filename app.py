import time
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

# MUST be first Streamlit call
st.set_page_config(page_title="Foresight Pricing", layout="wide")

# --- Splash handling (must come immediately after set_page_config) ---
SPLASH_SECONDS = 4.8
SPLASH_PATH = "assets/boot.mp4"

if "boot_start" not in st.session_state and not st.session_state.get("booted", False):
    st.session_state["boot_start"] = time.time()

if not st.session_state.get("booted", False):
    elapsed = time.time() - st.session_state.get("boot_start", time.time())
    if elapsed < SPLASH_SECONDS:
        show_boot_splash(SPLASH_PATH, seconds=SPLASH_SECONDS)  # this will st.stop()
    else:
        st.session_state["booted"] = True
        st.session_state["_booting"] = False
        st.session_state.pop("boot_start", None)
        st.rerun()
# --- End splash handling ---

init_db()

if not require_login():
    st.stop()

# DO NOT call show_boot_splash again here

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
    render_presence_panel(current_page_name=choice)

pages[choice]()





