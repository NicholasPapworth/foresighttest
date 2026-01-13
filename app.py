import streamlit as st
from src.db import init_db
from src.auth import require_login
from src.ui import render_header, page_trader, page_history, page_admin

st.set_page_config(page_title="Foresight Pricing", layout="wide")
init_db()

if not require_login():
    st.stop()

render_header()

pages = {
    "Trader": page_trader,
    "History": page_history,
    "Admin": page_admin,
}

with st.sidebar:
    choice = st.radio("Navigation", list(pages.keys()))
pages[choice]()


