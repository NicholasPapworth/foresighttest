import sqlite3
import pandas as pd
from datetime import datetime, timezone
import hashlib
import uuid

DB_PATH = "foresight.db"


def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL;")
    return c


def init_db():
    c = conn()
    cur = c.cursor()

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
    CREATE TABLE IF NOT EXISTS supplier_prices (
        snapshot_id TEXT NOT NULL,
        supplier TEXT NOT NULL,
        product_category TEXT,
        product TEXT NOT NULL,
        location TEXT NOT NULL,
        delivery_window TEXT NOT NULL,
        price REAL NOT NULL,
        unit TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, supplier, product, location, delivery_window),
        FOREIGN KEY (snapshot_id) REFERENCES supplier_snapshots(snapshot_id)
    );
    """)

        # --- Admin margins (category/product) ---
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

    cur.execute("""
    CREATE TABLE IF NOT EXISTS app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)

    # Defaults
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

    c.commit()
    c.close()


def _set_default(cur, key, value):
    cur.execute("SELECT 1 FROM app_settings WHERE key = ?", (key,))
    if not cur.fetchone():
        cur.execute("INSERT INTO app_settings (key, value) VALUES (?, ?)", (key, value))


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
          unit AS "Unit"
        FROM supplier_prices
        WHERE snapshot_id = ?
        ORDER BY supplier, product, location, delivery_window
    """, c, params=(snapshot_id,))
    c.close()
    return df


def publish_supplier_snapshot(df: pd.DataFrame, published_by: str, source_bytes: bytes) -> str:
    snapshot_id = str(uuid.uuid4())
    published_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    row_count = int(len(df))

    c = conn()
    cur = c.cursor()

    cur.execute("""
        INSERT INTO supplier_snapshots (snapshot_id, published_at_utc, published_by, source_hash, row_count)
        VALUES (?, ?, ?, ?, ?)
    """, (snapshot_id, published_at, published_by, source_hash, row_count))

    rows = []
    for r in df.to_dict("records"):
        rows.append((
            snapshot_id,
            str(r["Supplier"]).strip(),
            str(r.get("Product Category", "")).strip(),
            str(r["Product"]).strip(),
            str(r["Location"]).strip(),
            str(r["Delivery Window"]).strip(),
            float(r["Price"]),
            str(r["Unit"]).strip(),
        ))

    cur.executemany("""
        INSERT INTO supplier_prices
        (snapshot_id, supplier, product_category, product, location, delivery_window, price, unit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    c.commit()
    c.close()
    return snapshot_id


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
    """
    Expects columns: min_t, max_t, charge_per_t, active
    Replaces all tiers in DB.
    """
    work = df.copy()

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
    """, (
        scope_type,
        scope_value,
        float(margin_per_t),
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        user
    ))
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
    """
    Returns the effective margin per scope_type/scope_value.
    If multiple active entries exist for the same scope, the most recent (highest margin_id) wins.
    Output columns: scope_type, scope_value, margin_per_t
    """
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

    # Keep most recent entry per (scope_type, scope_value)
    df = df.drop_duplicates(subset=["scope_type", "scope_value"], keep="first")
    return df[["scope_type", "scope_value", "margin_per_t"]]

    # Require these columns
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
    work["active"] = work["active"].apply(lambda x: 1 if str(x).strip() in ("1", "True", "true") else 0)

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



