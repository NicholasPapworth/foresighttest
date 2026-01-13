import streamlit as st

def require_login() -> bool:
    if "user" not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None

    if st.session_state.user:
        return True

    st.markdown("### Sign in")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    users = st.secrets.get("users", {})

    if st.button("Sign in", use_container_width=True):
        u = users.get(username)
        if not u or u.get("password") != password:
            st.error("Invalid credentials.")
        else:
            st.session_state.user = username
            st.session_state.role = u.get("role", "trader")
            st.rerun()

    st.info("Admins can publish. Traders are read-only.")
    return False
