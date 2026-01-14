import time
import bcrypt
import streamlit as st


# --- Config (seconds) ---
DEFAULT_SESSION_TIMEOUT_SEC = 30 * 60  # 30 minutes inactivity


def _now() -> float:
    return time.time()


def _get_timeout_sec() -> int:
    # Optional: allow override via secrets
    try:
        return int(st.secrets.get("session_timeout_seconds", DEFAULT_SESSION_TIMEOUT_SEC))
    except Exception:
        return DEFAULT_SESSION_TIMEOUT_SEC


def logout():
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.last_seen = None
    st.rerun()


def _expire_if_inactive():
    timeout_sec = _get_timeout_sec()
    last_seen = st.session_state.get("last_seen")
    if last_seen is None:
        return
    if (_now() - float(last_seen)) > timeout_sec:
        # Inactive session expired
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.last_seen = None


def require_login() -> bool:
    # Ensure keys exist
    if "user" not in st.session_state:
        st.session_state.user = None
    if "role" not in st.session_state:
        st.session_state.role = None
    if "last_seen" not in st.session_state:
        st.session_state.last_seen = None

    # Expire if inactive
    _expire_if_inactive()

    # If logged in, refresh last seen and show logout in sidebar
    if st.session_state.user:
        st.session_state.last_seen = _now()
        with st.sidebar:
            st.markdown("---")
            st.caption(f"User: **{st.session_state.user}**")
            st.caption(f"Role: **{st.session_state.role}**")
            if st.button("Logout", use_container_width=True):
                logout()
        return True

    # Not logged in: show login form
    st.markdown("### Sign in")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    users = st.secrets.get("users", {})

    if st.button("Sign in", use_container_width=True):
        u = users.get(username)
        if not u:
            st.error("Invalid credentials.")
            return False

        pw_hash = u.get("password_hash")
        if not pw_hash:
            st.error("User is missing password_hash in secrets.")
            return False

        ok = False
        try:
            ok = bcrypt.checkpw(password.encode("utf-8"), pw_hash.encode("utf-8"))
        except Exception:
            ok = False

        if not ok:
            st.error("Invalid credentials.")
            return False

        st.session_state.user = username
        st.session_state.role = u.get("role", "trader")
        st.session_state.last_seen = _now()
        st.rerun()

    st.info("Admins can publish/manage. Traders are read-only for admin pages.")
    return False
