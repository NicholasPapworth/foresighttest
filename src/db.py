import sqlite3
import pandas as pd
from datetime import datetime, timezone, timedelta
import hashlib
import uuid
import json

DB_PATH = "foresight.db"


def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA foreign_keys=ON;")
    return c


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db():
    c = conn()
    cur = c.cursor()

    # --- Supplier snapshots ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS supplier_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        published_at_utc TEXT NOT NULL,
        published_by TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        row_count INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS ux_supplier_snapshots_source_hash
    ON supplier_snapshots (source_hash);
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS supplier_prices (
        snapshot_id TEXT NOT NULL,
        supplier TEXT NOT NULL,
        product_category TEXT,
        product TEXT NOT NULL,
        location TEXT NOT NULL,
        delivery_window TEXT NOT NULL,
        price REAL NOT NULL,
        sell_price REAL NOT NULL,
        unit TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, supplier, product, location, delivery_window),
        FOREIGN KEY (snapshot_id) REFERENCES supplier_snapshots(snapshot_id)
    );
    """)

    # Helpful indexes
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_supplier_prices_lookup
    ON supplier_prices (snapshot_id, product, location, delivery_window);
    """)

    _ensure_sell_price_column(cur, "supplier_prices")

    # --- Seed snapshots (NEW, identical shape to supplier snapshots) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS seed_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        published_at_utc TEXT NOT NULL,
        published_by TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        row_count INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS ux_seed_snapshots_source_hash
    ON seed_snapshots (source_hash);
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS seed_prices (
        snapshot_id TEXT NOT NULL,
        supplier TEXT NOT NULL,
        product_category TEXT,
        product TEXT NOT NULL,
        location TEXT NOT NULL,
        delivery_window TEXT NOT NULL,
        price REAL NOT NULL,
        sell_price REAL NOT NULL,
        unit TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, supplier, product, location, delivery_window),
        FOREIGN KEY (snapshot_id) REFERENCES seed_snapshots(snapshot_id)
    );
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_seed_prices_lookup
    ON seed_prices (snapshot_id, product, location, delivery_window);
    """)

    _ensure_sell_price_column(cur, "seed_prices") 

    # --- Admin margins ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS price_margins (
        margin_id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope_type TEXT NOT NULL CHECK (scope_type IN ('category','product')),
        scope_value TEXT NOT NULL,
        margin_per_t REAL NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        created_at_utc TEXT NOT NULL,
        created_by TEXT NOT NULL
    );
    """)

    # --- Presence (who is online) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_presence (
        user TEXT NOT NULL,
        session_id TEXT NOT NULL,
        role TEXT,
        page TEXT,
        online_since_utc TEXT NOT NULL,
        last_seen_utc TEXT NOT NULL,
        PRIMARY KEY (user, session_id)
    );
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_presence_last_seen
    ON user_presence (last_seen_utc);
    """)


    # --- App settings ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)
    _set_default(cur, "basket_timeout_minutes", "20")

    # --- Tiered small-lot charges (global) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS small_lot_tiers (
        tier_id INTEGER PRIMARY KEY AUTOINCREMENT,
        min_t REAL NOT NULL,
        max_t REAL,
        charge_per_t REAL NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    );
    """)

    # Seed defaults if empty
    cur.execute("SELECT COUNT(*) FROM small_lot_tiers;")
    if cur.fetchone()[0] == 0:
        cur.executemany("""
            INSERT INTO small_lot_tiers (min_t, max_t, charge_per_t, active)
            VALUES (?, ?, ?, 1)
        """, [
            (0.60, 2.39, 130.0),
            (2.40, 4.80, 70.0),
            (4.90, 9.90, 15.0),
            (10.0, 14.9, 8.0),
            (15.0, 24.0, 4.0),
            (24.0, None, 0.0),  # >=24t no charge
        ])

    # --- Orders workflow (NEW) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        created_at_utc TEXT NOT NULL,
        created_by TEXT NOT NULL,
        status TEXT NOT NULL,
        supplier_snapshot_id TEXT NOT NULL,
        last_action_at_utc TEXT NOT NULL,
        last_action_by TEXT NOT NULL,
        trader_note TEXT,
        admin_note TEXT,
        version INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (supplier_snapshot_id) REFERENCES supplier_snapshots(snapshot_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS order_lines (
        order_id TEXT NOT NULL,
        line_no INTEGER NOT NULL,
        product_category TEXT,
        product TEXT NOT NULL,
        location TEXT NOT NULL,
        delivery_window TEXT NOT NULL,
        qty REAL NOT NULL,
        unit TEXT NOT NULL,
        supplier TEXT NOT NULL,
        base_price REAL NOT NULL,
        sell_price REAL NOT NULL,
        PRIMARY KEY (order_id, line_no),
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS order_actions (
        action_id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT NOT NULL,
        action_type TEXT NOT NULL,
        action_at_utc TEXT NOT NULL,
        action_by TEXT NOT NULL,
        payload_json TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_by_user ON orders(created_by, created_at_utc);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status, created_at_utc);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_actions_order ON order_actions(order_id, action_at_utc);")

    c.commit()
    c.close()


def _set_default(cur, key, value):
    cur.execute("SELECT 1 FROM app_settings WHERE key = ?", (key,))
    if not cur.fetchone():
        cur.execute("INSERT INTO app_settings (key, value) VALUES (?, ?)", (key, value))


# ---------------- Settings ----------------

def get_settings() -> dict:
    c = conn()
    df = pd.read_sql_query("SELECT key, value FROM app_settings", c)
    c.close()
    return {r["key"]: r["value"] for _, r in df.iterrows()}


def set_setting(key: str, value: str):
    c = conn()
    cur = c.cursor()
    cur.execute("""
        INSERT INTO app_settings (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """, (key, value))
    c.commit()
    c.close()


# ---------------- Supplier snapshots ----------------

def list_supplier_snapshots(limit=200) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query(f"""
        SELECT snapshot_id, published_at_utc, published_by, row_count
        FROM supplier_snapshots
        ORDER BY published_at_utc DESC
        LIMIT {int(limit)}
    """, c)
    c.close()
    return df


def latest_supplier_snapshot():
    c = conn()
    cur = c.cursor()
    cur.execute("""
        SELECT snapshot_id, published_at_utc, published_by
        FROM supplier_snapshots
        ORDER BY published_at_utc DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    c.close()
    return row


def load_supplier_prices(snapshot_id: str) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query("""
        SELECT
          supplier AS "Supplier",
          product_category AS "Product Category",
          product AS "Product",
          location AS "Location",
          delivery_window AS "Delivery Window",
          price AS "Price",
          COALESCE(sell_price, price) AS "Sell Price",
          unit AS "Unit"
        FROM supplier_prices
        WHERE snapshot_id = ?
        ORDER BY supplier, product, location, delivery_window
    """, c, params=(snapshot_id,))
    c.close()
    return df


def publish_supplier_snapshot(df: pd.DataFrame, published_by: str, source_bytes: bytes) -> str:
    work = df.copy()
    work["Price"] = pd.to_numeric(work["Price"], errors="coerce")

    if "Sell Price" not in work.columns:
        work["Sell Price"] = work["Price"]
    work["Sell Price"] = pd.to_numeric(work["Sell Price"], errors="coerce")

    work = work.dropna(subset=["Price", "Sell Price"])
    if work.empty:
        raise ValueError("No valid rows to publish (all rows had blank/invalid Price).")

    snapshot_id = str(uuid.uuid4())
    published_at = utc_now_iso()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    row_count = int(len(work))

    c = conn()
    cur = c.cursor()

    # Insert snapshot header
    try:
        cur.execute("""
            INSERT INTO supplier_snapshots (snapshot_id, published_at_utc, published_by, source_hash, row_count)
            VALUES (?, ?, ?, ?, ?)
        """, (snapshot_id, published_at, published_by, source_hash, row_count))
    except sqlite3.IntegrityError as e:
        c.close()
        raise ValueError("This supplier file has already been published (duplicate source_hash).") from e

    # Insert snapshot rows
    rows = []
    for r in work.to_dict("records"):
        rows.append((
            snapshot_id,
            str(r["Supplier"]).strip(),
            str(r.get("Product Category", "")).strip(),
            str(r["Product"]).strip(),
            str(r.get("Location", "")).strip(),
            str(r["Delivery Window"]).strip(),
            float(r["Price"]),
            float(r["Sell Price"]),
            str(r["Unit"]).strip(),
        ))

    cur.executemany("""
        INSERT INTO supplier_prices
        (snapshot_id, supplier, product_category, product, location, delivery_window, price, sell_price, unit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    c.commit()
    c.close()
    return snapshot_id

def _ensure_sell_price_column(cur, table_name: str):
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table_name})").fetchall()]
    if "sell_price" not in cols:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN sell_price REAL;")
    # Backfill any NULLs so old snapshots still work
    cur.execute(f"UPDATE {table_name} SET sell_price = price WHERE sell_price IS NULL;")

# ---------------- Seed snapshots (NEW) ----------------

def list_seed_snapshots(limit=200) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query(f"""
        SELECT snapshot_id, published_at_utc, published_by, row_count
        FROM seed_snapshots
        ORDER BY published_at_utc DESC
        LIMIT {int(limit)}
    """, c)
    c.close()
    return df


def latest_seed_snapshot():
    c = conn()
    cur = c.cursor()
    cur.execute("""
        SELECT snapshot_id, published_at_utc, published_by
        FROM seed_snapshots
        ORDER BY published_at_utc DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    c.close()
    return row


def load_seed_prices(snapshot_id: str) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query("""
        SELECT
          supplier AS "Supplier",
          product_category AS "Product Category",
          product AS "Product",
          location AS "Location",
          delivery_window AS "Delivery Window",
          price AS "Price",
          COALESCE(sell_price, price) AS "Sell Price",
          unit AS "Unit"
        FROM seed_prices
        WHERE snapshot_id = ?
        ORDER BY supplier, product, location, delivery_window
    """, c, params=(snapshot_id,))
    c.close()
    return df

def publish_seed_snapshot(df: pd.DataFrame, published_by: str, source_bytes: bytes) -> str:
    # --- DB safety net: drop rows with missing/invalid Price ---
    work = df.copy()
    work["Price"] = pd.to_numeric(work["Price"], errors="coerce")
    
    if "Sell Price" not in work.columns:
        work["Sell Price"] = work["Price"]
    work["Sell Price"] = pd.to_numeric(work["Sell Price"], errors="coerce")
    
    work = work.dropna(subset=["Price", "Sell Price"])

    if work.empty:
        raise ValueError("No valid rows to publish (all rows had blank/invalid Price).")

    snapshot_id = str(uuid.uuid4())
    published_at = utc_now_iso()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    row_count = int(len(work))

    c = conn()
    cur = c.cursor()

    cur.execute("""
        INSERT INTO seed_snapshots (snapshot_id, published_at_utc, published_by, source_hash, row_count)
        VALUES (?, ?, ?, ?, ?)
    """, (snapshot_id, published_at, published_by, source_hash, row_count))

    rows = []
    for r in work.to_dict("records"):
        rows.append((
            snapshot_id,
            str(r["Supplier"]).strip(),
            str(r.get("Product Category", "")).strip(),
            str(r["Product"]).strip(),
            str(r.get("Location", "")).strip(),
            str(r["Delivery Window"]).strip(),
            float(r["Price"]),
            float(r["Sell Price"]),
            str(r["Unit"]).strip(),
        ))

    cur.executemany("""
        INSERT INTO seed_prices
        (snapshot_id, supplier, product_category, product, location, delivery_window, price, sell_price, unit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    c.commit()
    c.close()
    return snapshot_id

# ---------------- Small-lot tiers ----------------

def get_small_lot_tiers() -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query("""
        SELECT tier_id, min_t, max_t, charge_per_t, active
        FROM small_lot_tiers
        ORDER BY min_t ASC
    """, c)
    c.close()
    return df


def save_small_lot_tiers(df: pd.DataFrame):
    work = df.copy()

    for col in ["min_t", "charge_per_t"]:
        if col not in work.columns:
            raise ValueError(f"Missing column '{col}' in tiers editor.")

    work["min_t"] = pd.to_numeric(work["min_t"], errors="raise")
    work["charge_per_t"] = pd.to_numeric(work["charge_per_t"], errors="raise")

    if "max_t" not in work.columns:
        work["max_t"] = None
    work["max_t"] = work["max_t"].apply(lambda x: None if x == "" or pd.isna(x) else float(x))

    if "active" not in work.columns:
        work["active"] = 1
    work["active"] = work["active"].apply(lambda x: 1 if bool(x) else 0).astype(int)

    tiers = work.sort_values("min_t").reset_index(drop=True)

    # Validate min < max where max exists
    for i in range(len(tiers)):
        mn = float(tiers.loc[i, "min_t"])
        mx = tiers.loc[i, "max_t"]
        if mx is not None and mn >= float(mx):
            raise ValueError(f"Invalid tier: min_t {mn} must be < max_t {mx}")

    # Validate no overlaps among active tiers
    active = tiers[tiers["active"] == 1].copy().sort_values("min_t").reset_index(drop=True)

    def _end(row):
        return float(row["max_t"]) if row["max_t"] is not None else float("inf")

    for i in range(len(active) - 1):
        if _end(active.loc[i]) >= float(active.loc[i + 1, "min_t"]):
            raise ValueError("Overlapping active tiers detected. Adjust min/max so tiers do not overlap.")

    c = conn()
    cur = c.cursor()
    cur.execute("DELETE FROM small_lot_tiers;")

    rows = []
    for r in tiers.to_dict("records"):
        rows.append((
            float(r["min_t"]),
            r["max_t"],
            float(r["charge_per_t"]),
            int(r["active"]),
        ))

    cur.executemany("""
        INSERT INTO small_lot_tiers (min_t, max_t, charge_per_t, active)
        VALUES (?, ?, ?, ?)
    """, rows)

    c.commit()
    c.close()


# ---------------- Margins ----------------

def add_margin(scope_type: str, scope_value: str, margin_per_t: float, user: str):
    scope_type = str(scope_type).strip().lower()
    if scope_type not in ("category", "product"):
        raise ValueError("scope_type must be 'category' or 'product'.")
    scope_value = str(scope_value).strip()
    if not scope_value:
        raise ValueError("scope_value cannot be empty.")

    c = conn()
    cur = c.cursor()
    cur.execute("""
        INSERT INTO price_margins
        (scope_type, scope_value, margin_per_t, active, created_at_utc, created_by)
        VALUES (?, ?, ?, 1, ?, ?)
    """, (scope_type, scope_value, float(margin_per_t), utc_now_iso(), user))
    c.commit()
    c.close()


def list_margins(active_only: bool = True) -> pd.DataFrame:
    c = conn()
    if active_only:
        df = pd.read_sql_query("""
            SELECT margin_id, scope_type, scope_value, margin_per_t, active, created_at_utc, created_by
            FROM price_margins
            WHERE active = 1
            ORDER BY margin_id DESC
        """, c)
    else:
        df = pd.read_sql_query("""
            SELECT margin_id, scope_type, scope_value, margin_per_t, active, created_at_utc, created_by
            FROM price_margins
            ORDER BY margin_id DESC
        """, c)
    c.close()
    return df


def deactivate_margin(margin_id: int):
    c = conn()
    cur = c.cursor()
    cur.execute("UPDATE price_margins SET active = 0 WHERE margin_id = ?", (int(margin_id),))
    c.commit()
    c.close()


def get_effective_margins() -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query("""
        SELECT margin_id, scope_type, scope_value, margin_per_t
        FROM price_margins
        WHERE active = 1
        ORDER BY margin_id DESC
    """, c)
    c.close()

    if df.empty:
        return pd.DataFrame(columns=["scope_type", "scope_value", "margin_per_t"])

    df = df.drop_duplicates(subset=["scope_type", "scope_value"], keep="first")
    return df[["scope_type", "scope_value", "margin_per_t"]]


# ---------------- Orders workflow ----------------

ORDER_STATUSES = {"PENDING", "COUNTERED", "CONFIRMED", "FILLED", "REJECTED", "CANCELLED"}

# Allowed transitions (state machine)
ALLOWED_TRANSITIONS = {
    # current_status: {action_type: new_status}
    "PENDING": {
        "CANCEL": "CANCELLED",
        "COUNTER": "COUNTERED",
        "CONFIRM": "CONFIRMED",
        "REJECT": "REJECTED",
    },
    "COUNTERED": {
        "CANCEL": "CANCELLED",
        "ACCEPT_COUNTER": "CONFIRMED",
        "COUNTER": "COUNTERED",   # allow re-counter
        "CONFIRM": "CONFIRMED",
        "REJECT": "REJECTED",
    },
    "CONFIRMED": {
        "FILL": "FILLED",
        "REJECT": "REJECTED",     # optional, you currently allow reject from confirmed
    },
    "FILLED": {},
    "REJECTED": {},
    "CANCELLED": {},
}

def _add_action(cur, order_id: str, action_type: str, action_by: str, payload: dict | None = None):
    cur.execute("""
        INSERT INTO order_actions (order_id, action_type, action_at_utc, action_by, payload_json)
        VALUES (?, ?, ?, ?, ?)
    """, (
        order_id,
        action_type,
        utc_now_iso(),
        action_by,
        None if payload is None else json.dumps(payload)
    ))

def _transition_order(
    order_id: str,
    action_type: str,
    action_by: str,
    *,
    expected_version: int | None = None,
    admin_note: str | None = None,
    payload: dict | None = None,
    edited_lines: pd.DataFrame | None = None,
):
    """
    Central gatekeeper:
    - Enforces state machine transitions
    - Applies optimistic locking via orders.version
    - Updates orders.status + last_action fields + optional admin_note
    - Writes order_actions audit record
    - Increments orders.version on every successful transition
    - Optionally updates order_lines sell_price for COUNTER
    """
    if action_type not in ("SUBMIT", "CANCEL", "COUNTER", "ACCEPT_COUNTER", "CONFIRM", "REJECT", "FILL"):
        raise ValueError(f"Unknown action_type: {action_type}")

    c = conn()
    cur = c.cursor()

    # Load current status and version
    cur.execute("SELECT status, version, created_by FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    if not row:
        c.close()
        raise ValueError("Order not found.")

    cur_status = row[0]
    cur_version = int(row[1] or 0)

    # Optimistic locking check
    if expected_version is not None and int(expected_version) != cur_version:
        c.close()
        raise ValueError("Order changed since you opened it. Refresh and try again.")

    # Validate transition (SUBMIT is handled during create)
    if action_type != "SUBMIT":
        allowed = ALLOWED_TRANSITIONS.get(cur_status, {})
        if action_type not in allowed:
            c.close()
            raise ValueError(f"Invalid transition: {cur_status} -> ? via {action_type}")
        new_status = allowed[action_type]
    else:
        new_status = cur_status  # not used

    # If COUNTER and edited_lines provided: update sell_price per line_no
    if action_type == "COUNTER" and edited_lines is not None:
        work = edited_lines.copy()
        if "line_no" not in work.columns:
            c.close()
            raise ValueError("edited_lines must include line_no.")
        if "Sell Price" not in work.columns:
            c.close()
            raise ValueError("edited_lines must include 'Sell Price'.")

        work["Sell Price"] = pd.to_numeric(work["Sell Price"], errors="raise")

        # Update each line sell_price (MVP behaviour you already had)
        for _, r in work.iterrows():
            ln = int(r["line_no"])
            sp = float(r["Sell Price"])
            cur.execute("""
                UPDATE order_lines
                SET sell_price = ?
                WHERE order_id = ? AND line_no = ?
            """, (sp, order_id, ln))

    # Update admin note if provided
    if admin_note is not None:
        cur.execute("UPDATE orders SET admin_note = ? WHERE order_id = ?", (admin_note, order_id))

    # Audit action
    _add_action(cur, order_id, action_type, action_by, payload)

    # Apply status change + last action + bump version
    if action_type != "SUBMIT":
        cur.execute("""
            UPDATE orders
            SET status = ?,
                last_action_at_utc = ?,
                last_action_by = ?,
                version = version + 1
            WHERE order_id = ?
        """, (new_status, utc_now_iso(), action_by, order_id))
    else:
        # If you ever call SUBMIT here, still bump version to be consistent
        cur.execute("""
            UPDATE orders
            SET last_action_at_utc = ?,
                last_action_by = ?,
                version = version + 1
            WHERE order_id = ?
        """, (utc_now_iso(), action_by, order_id))

    c.commit()
    c.close()

def create_order_from_allocation(
    created_by: str,
    supplier_snapshot_id: str,
    allocation_lines: list[dict],
    trader_note: str = ""
) -> str:
    """
    allocation_lines: list of dicts containing:
      Product Category, Product, Location, Delivery Window, Qty, Unit, Supplier, Base Price, Sell Price
    """
    if not allocation_lines:
        raise ValueError("No allocation lines to create order.")

    order_id = str(uuid.uuid4())
    now = utc_now_iso()

    c = conn()
    cur = c.cursor()

    cur.execute("""
        INSERT INTO orders
        (order_id, created_at_utc, created_by, status, supplier_snapshot_id, last_action_at_utc, last_action_by, trader_note, admin_note, version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    """, (order_id, now, created_by, "PENDING", supplier_snapshot_id, now, created_by, trader_note, ""))

    rows = []
    for i, ln in enumerate(allocation_lines, start=1):
        rows.append((
            order_id,
            i,
            str(ln.get("Product Category", "")).strip(),
            str(ln["Product"]).strip(),
            str(ln.get("Location", "")).strip(),
            str(ln["Delivery Window"]).strip(),
            float(ln["Qty"]),
            str(ln.get("Unit", "£/t")).strip(),
            str(ln["Supplier"]).strip(),
            float(ln["Base Price"]),
            float(ln["Sell Price"]),
        ))

    cur.executemany("""
        INSERT INTO order_lines
        (order_id, line_no, product_category, product, location, delivery_window, qty, unit, supplier, base_price, sell_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    _add_action(cur, order_id, "SUBMIT", created_by, {"lines": len(rows)})

    cur.execute("""
        UPDATE orders
        SET last_action_at_utc = ?, last_action_by = ?, status = 'PENDING'
        WHERE order_id = ?
    """, (utc_now_iso(), created_by, order_id))

    c.commit()
    c.close()
    return order_id


def list_orders_for_user(user: str) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query("""
        SELECT order_id, created_at_utc, status, supplier_snapshot_id, last_action_at_utc, trader_note
        FROM orders
        WHERE created_by = ?
        ORDER BY created_at_utc DESC
    """, c, params=(user,))
    c.close()
    return df


def list_orders_admin(status_filter: str | None = None) -> pd.DataFrame:
    c = conn()
    if status_filter:
        df = pd.read_sql_query("""
            SELECT order_id, created_at_utc, created_by, status, supplier_snapshot_id, last_action_at_utc, last_action_by
            FROM orders
            WHERE status = ?
            ORDER BY created_at_utc DESC
        """, c, params=(status_filter,))
    else:
        df = pd.read_sql_query("""
            SELECT order_id, created_at_utc, created_by, status, supplier_snapshot_id, last_action_at_utc, last_action_by
            FROM orders
            ORDER BY created_at_utc DESC
        """, c)
    c.close()
    return df


def get_order_lines(order_id: str) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query("""
        SELECT line_no, product_category AS "Product Category", product AS "Product",
               location AS "Location", delivery_window AS "Delivery Window",
               qty AS "Qty", unit AS "Unit", supplier AS "Supplier",
               base_price AS "Base Price", sell_price AS "Sell Price"
        FROM order_lines
        WHERE order_id = ?
        ORDER BY line_no ASC
    """, c, params=(order_id,))
    c.close()
    return df


def get_order_header(order_id: str) -> dict | None:
    c = conn()
    cur = c.cursor()
    cur.execute("""
        SELECT order_id, created_at_utc, created_by, status, supplier_snapshot_id,
               last_action_at_utc, last_action_by, trader_note, admin_note, version
        FROM orders
        WHERE order_id = ?
    """, (order_id,))
    row = cur.fetchone()
    c.close()

    if not row:
        return None

    keys = [
        "order_id","created_at_utc","created_by","status","supplier_snapshot_id",
        "last_action_at_utc","last_action_by","trader_note","admin_note","version"
    ]
    out = dict(zip(keys, row))
    out["version"] = int(out.get("version") or 0)
    return out

def get_order_actions(order_id: str) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query("""
        SELECT action_id, action_type, action_at_utc, action_by, payload_json
        FROM order_actions
        WHERE order_id = ?
        ORDER BY action_at_utc ASC
    """, c, params=(order_id,))
    c.close()
    return df


def trader_cancel_order(order_id: str, user: str, expected_version: int | None = None):
    h = get_order_header(order_id)
    if not h:
        raise ValueError("Order not found.")
    if h["created_by"] != user:
        raise ValueError("Not your order.")

    _transition_order(
        order_id=order_id,
        action_type="CANCEL",
        action_by=user,
        expected_version=expected_version,
    )


def admin_counter_order(
    order_id: str,
    admin_user: str,
    edited_lines: pd.DataFrame,
    admin_note: str = "",
    expected_version: int | None = None
):
    h = get_order_header(order_id)
    if not h:
        raise ValueError("Order not found.")

    before = get_order_lines(order_id)

    work = edited_lines.copy()
    if "line_no" not in work.columns:
        raise ValueError("edited_lines must include line_no.")
    if "Sell Price" not in work.columns:
        raise ValueError("edited_lines must include 'Sell Price'.")

    work["Sell Price"] = pd.to_numeric(work["Sell Price"], errors="raise")

    after_preview = before.copy()
    # Build a payload diff (audit)
    payload = {
        "admin_note": admin_note,
        "diff": {
            "before": before[["line_no", "Sell Price"]].to_dict("records"),
            "after": work[["line_no", "Sell Price"]].to_dict("records"),
        }
    }

    _transition_order(
        order_id=order_id,
        action_type="COUNTER",
        action_by=admin_user,
        expected_version=expected_version,
        admin_note=admin_note,
        payload=payload,
        edited_lines=work,
    )


def trader_accept_counter(order_id: str, user: str, expected_version: int | None = None):
    h = get_order_header(order_id)
    if not h:
        raise ValueError("Order not found.")
    if h["created_by"] != user:
        raise ValueError("Not your order.")

    _transition_order(
        order_id=order_id,
        action_type="ACCEPT_COUNTER",
        action_by=user,
        expected_version=expected_version,
    )


def admin_confirm_order(order_id: str, admin_user: str, expected_version: int | None = None):
    h = get_order_header(order_id)
    if not h:
        raise ValueError("Order not found.")

    _transition_order(
        order_id=order_id,
        action_type="CONFIRM",
        action_by=admin_user,
        expected_version=expected_version,
    )


def admin_reject_order(order_id: str, admin_user: str, admin_note: str = "", expected_version: int | None = None):
    h = get_order_header(order_id)
    if not h:
        raise ValueError("Order not found.")

    _transition_order(
        order_id=order_id,
        action_type="REJECT",
        action_by=admin_user,
        expected_version=expected_version,
        admin_note=admin_note,
        payload={"admin_note": admin_note},
    )


def admin_mark_filled(order_id: str, admin_user: str, expected_version: int | None = None):
    h = get_order_header(order_id)
    if not h:
        raise ValueError("Order not found.")

    _transition_order(
        order_id=order_id,
        action_type="FILL",
        action_by=admin_user,
        expected_version=expected_version,
    )

def presence_heartbeat(user: str, role: str, page: str, session_id: str):
    """
    Upserts a presence heartbeat for this user+session.
    """
    if not user or not session_id:
        return

    now = utc_now_iso()

    c = conn()
    cur = c.cursor()
    cur.execute("""
        INSERT INTO user_presence (user, session_id, role, page, online_since_utc, last_seen_utc)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user, session_id) DO UPDATE SET
            role = excluded.role,
            page = excluded.page,
            last_seen_utc = excluded.last_seen_utc
    """, (user, session_id, role, page, now, now))
    c.commit()
    c.close()


def presence_signoff(user: str, session_id: str):
    """
    Removes this session from presence (call on logout if you have a logout action).
    """
    if not user or not session_id:
        return

    c = conn()
    cur = c.cursor()
    cur.execute("DELETE FROM user_presence WHERE user = ? AND session_id = ?", (user, session_id))
    c.commit()
    c.close()


def list_online_users(online_within_seconds: int = 45) -> pd.DataFrame:
    """
    Returns distinct users considered 'online' if last_seen is within threshold.
    Also performs a small cleanup of stale sessions.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=int(online_within_seconds))).isoformat(timespec="seconds")

    c = conn()
    cur = c.cursor()

    # Cleanup stale sessions (optional but recommended)
    cur.execute("DELETE FROM user_presence WHERE last_seen_utc < ?", (cutoff,))
    c.commit()

    df = pd.read_sql_query("""
        SELECT
            user,
            MAX(role) AS role,
            MAX(page) AS page,
            MIN(online_since_utc) AS online_since_utc,
            MAX(last_seen_utc) AS last_seen_utc
        FROM user_presence
        WHERE last_seen_utc >= ?
        GROUP BY user
        ORDER BY user ASC
    """, c, params=(cutoff,))
    c.close()
    return df

def admin_margin_report() -> pd.DataFrame:
    """
    Simple report over FILLED orders:
      margin = sum((sell_price - base_price) * qty)
    """
    c = conn()
    df = pd.read_sql_query("""
        SELECT
          o.order_id,
          o.created_at_utc,
          o.created_by,
          SUM(ol.qty) AS total_tonnes,
          SUM(ol.sell_price * ol.qty) AS sell_value,
          SUM(ol.base_price * ol.qty) AS base_value,
          SUM((ol.sell_price - ol.base_price) * ol.qty) AS gross_margin
        FROM orders o
        JOIN order_lines ol ON ol.order_id = o.order_id
        WHERE o.status = 'FILLED'
        GROUP BY o.order_id, o.created_at_utc, o.created_by
        ORDER BY o.created_at_utc DESC
    """, c)
    c.close()
    return df

def admin_blotter_lines() -> pd.DataFrame:
    """
    Line-level blotter for FILLED orders.
    Returns one row per order line with dimensions for filtering.
    """
    c = conn()
    q = """
    SELECT
        o.order_id,
        o.created_at_utc,
        o.created_by,
        l.line_no,
        l.product_category  AS product_category,
        l.product           AS product,
        l.location          AS location,
        l.delivery_window   AS delivery_window,
        l.supplier          AS supplier,
        l.qty               AS qty,
        l.base_price        AS base_price,
        l.sell_price        AS sell_price
    FROM orders o
    JOIN order_lines l
      ON l.order_id = o.order_id
    WHERE o.status = 'FILLED'
    ORDER BY o.created_at_utc DESC, o.order_id, l.line_no
    """
    df = pd.read_sql_query(q, c)
    c.close()
    return df
























