import streamlit as st
import uuid

from src import db
from src.auth import require_login
from src.ui import (
    show_boot_splash,
    render_header,
    render_presence_panel,

    # Trader
    page_todays_offers,
    page_trader_pricing,
    page_trader_best_prices,
    page_trader_orders,

    # Wholesale
    page_wholesale_offers,
    page_wholesale_pricing,
    page_wholesale_best_prices,
    page_wholesale_orders,

    # Shared / Admin
    page_history,
    page_admin_pricing,
    page_admin_orders,
    page_admin_blotter,
    page_admin_supplier_analysis,
    page_admin_user_data,
)

# MUST be first Streamlit call
st.set_page_config(page_title="Foresight Pricing", layout="wide")

# Splash ONCE per session (show_boot_splash handles timing + rerun internally)
show_boot_splash(video_path="assets/boot.mp4", seconds=4.8)

# Stable per-browser-session id (persists across reruns)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Init DB
db.init_db()

# Auth
if not require_login():
    st.stop()

# Header
render_header()

role = st.session_state.get("role")

# -------------------------
# Page registry
# -------------------------
pages: dict[str, callable] = {}

# Trader access (and admin can see trader pages too)
if role in ("trader", "admin", None, ""):
    pages.update({
        "Trader | Offers": page_todays_offers,
        "Trader | Pricing": page_trader_pricing,
        "Trader | Full Load Prices": page_trader_best_prices,
        "Trader | Orders": page_trader_orders,
        "Trader | Price History": (lambda: page_history(role_code="trader")),
    })

# Wholesale access (and admin can see wholesale pages too)
if role in ("wholesaler", "admin"):
    pages.update({
        "Wholesale | Offers": page_wholesale_offers,
        "Wholesale | Pricing": page_wholesale_pricing,
        "Wholesale | Full Load Prices": page_wholesale_best_prices,
        "Wholesale | Orders": page_wholesale_orders,
        "Wholesale | Price History": (lambda: page_history(role_code="wholesaler")),
    })

# Admin-only pages
if role == "admin":
    pages.update({
        "Admin | Pricing": page_admin_pricing,
        "Admin | Stock": page_admin_stock,
        "Admin | Orders": page_admin_orders,
        "Admin | Blotter": page_admin_blotter,
        "Admin | Supplier Analysis": page_admin_supplier_analysis,
        "Admin | User Data": page_admin_user_data,
    })

# Safety: if somehow no pages are available, stop cleanly
if not pages:
    st.error("No pages available for your role.")
    st.stop()

# -------------------------
# Sidebar: Navigation + Presence (read-only)
# -------------------------
with st.sidebar:
    nav_tab, presence_tab = st.tabs(["Navigation", "Presence"])

    with nav_tab:
        st.markdown("### Navigation")
        choice = st.radio("", list(pages.keys()), key="nav_choice")
        if choice not in pages:
            choice = list(pages.keys())[0]
            st.session_state["nav_choice"] = choice

    with presence_tab:
        render_presence_panel(current_page_name=choice)

# -------------------------
# Run selected page FIRST (so ui.py can set presence_context correctly)
# -------------------------
pages[choice]()

# -------------------------
# Presence heartbeat AFTER page render (single write per run)
# -------------------------
user = st.session_state.get("user") or st.session_state.get("username") or ""
role_code = st.session_state.get("role") or ""
context = st.session_state.get("presence_context", "") or ""

# Only allow expected contexts (avoid polluting analytics)
if context not in ("fert", "seed", ""):
    context = ""

# Guard: never write empty users
if user:
    db.presence_heartbeat(
        user=user,
        role=role_code,
        page=choice,
        session_id=st.session_state.session_id,
        context=context,
    )


