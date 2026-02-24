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
    # SQLite-friendly UTC timestamp
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def _ensure_column(cur, table_name: str, col_name: str, col_ddl: str):
    """
    Adds a column if missing.
    col_ddl example: "notes TEXT"
    """
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table_name})").fetchall()]
    if col_name not in cols:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_ddl};")

def _ensure_sell_price_column(cur, table_name: str):
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table_name})").fetchall()]
    if "sell_price" not in cols:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN sell_price REAL;")
    # Backfill any NULLs so old snapshots still work
    cur.execute(f"UPDATE {table_name} SET sell_price = price WHERE sell_price IS NULL;")

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
        notes TEXT,
        cost_per_kg_n TEXT,
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

    _ensure_column(cur, "supplier_prices", "notes", "notes TEXT")
    _ensure_column(cur, "supplier_prices", "cost_per_kg_n", "cost_per_kg_n TEXT")

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
        notes TEXT,
        cost_per_kg_n TEXT,
        PRIMARY KEY (snapshot_id, supplier, product, location, delivery_window),
        FOREIGN KEY (snapshot_id) REFERENCES seed_snapshots(snapshot_id)
    );
    """)

    _ensure_column(cur, "seed_prices", "notes", "notes TEXT")
    _ensure_column(cur, "seed_prices", "cost_per_kg_n", "cost_per_kg_n TEXT")

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_seed_prices_lookup
    ON seed_prices (snapshot_id, product, location, delivery_window);
    """)

    _ensure_sell_price_column(cur, "seed_prices") 

    # --- Admin margins ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS price_margins (
          margin_id INTEGER PRIMARY KEY AUTOINCREMENT,
          role_code TEXT NOT NULL DEFAULT 'trader',
          scope_type TEXT NOT NULL,
          scope_value TEXT NOT NULL,
          margin_per_t REAL NOT NULL,
          active INTEGER NOT NULL DEFAULT 1,
          created_at_utc TEXT NOT NULL,
          created_by TEXT NOT NULL
        );
    """)

    # --- Fertiliser delivery/collection options (Fert-only add-ons) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fert_delivery_options (
        option_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        delta_per_t REAL NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        created_at_utc TEXT NOT NULL,
        created_by TEXT NOT NULL
    );
    """)

    # ---- MIGRATION: add role_code to price_margins if missing ----
    try:
        cur.execute("ALTER TABLE price_margins ADD COLUMN role_code TEXT NOT NULL DEFAULT 'trader';")
    except Exception:
        pass

    # Defaults if empty
    cur.execute("SELECT COUNT(*) FROM fert_delivery_options;")
    if cur.fetchone()[0] == 0:
        now = utc_now_iso()
        cur.executemany("""
            INSERT INTO fert_delivery_options (name, delta_per_t, active, created_at_utc, created_by)
            VALUES (?, ?, 1, ?, ?)
        """, [
            ("Delivered", 0.0, now, "system"),
            ("Bulk delivered", -5.0, now, "system"),
            ("Collected", -10.0, now, "system"),
            ("Collected bulk", -15.0, now, "system"),
        ])

    # --- Today's Offers (time-bounded price overrides) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS todays_offers (
        offer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_code TEXT NOT NULL CHECK (book_code IN ('fert','seed')),
        product TEXT NOT NULL,
        location TEXT NOT NULL,
        delivery_window TEXT NOT NULL,
        supplier TEXT,
        mode TEXT NOT NULL CHECK (mode IN ('delta','fixed')),
        value REAL NOT NULL,
        title TEXT,
        active INTEGER NOT NULL DEFAULT 1,
        starts_at_utc TEXT NOT NULL,
        ends_at_utc TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        created_by TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_todays_offers_active_time
    ON todays_offers (book_code, active, starts_at_utc, ends_at_utc);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_todays_offers_lookup
    ON todays_offers (book_code, product, location, delivery_window, supplier);
    """)

    # --- Presence (who is online) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_presence (
        user TEXT NOT NULL,
        session_id TEXT NOT NULL,
        role TEXT,
        page TEXT,
        context TEXT DEFAULT '',
        online_since_utc TEXT NOT NULL,
        last_seen_utc TEXT NOT NULL,
        PRIMARY KEY (user, session_id)
    );
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_presence_last_seen
    ON user_presence (last_seen_utc);
    """)

    # --- Presence events (history / analytics) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS presence_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_at_utc TEXT NOT NULL,
        user TEXT NOT NULL,
        session_id TEXT NOT NULL,
        role TEXT,
        page TEXT,
        context TEXT DEFAULT ''
    );
    """)

     # ✅ MIGRATION SAFETY: if table already existed from older schema, ensure required columns exist
    _ensure_column(cur, "presence_events", "role", "role TEXT")
    _ensure_column(cur, "presence_events", "page", "page TEXT")
    _ensure_column(cur, "presence_events", "context", "context TEXT DEFAULT ''")

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_presence_events_user_time
    ON presence_events (user, event_at_utc);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_presence_events_session_time
    ON presence_events (session_id, event_at_utc);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_presence_events_page_time
    ON presence_events (page, event_at_utc);
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_presence_events_context_time
    ON presence_events (context, event_at_utc);
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

    # --- Seed treatments (Seed-only add-ons) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS seed_treatments (
        treatment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        charge_per_t REAL NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        created_at_utc TEXT NOT NULL,
        created_by TEXT NOT NULL
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

    _ensure_column(cur, "order_lines", "delivery_method", "delivery_method TEXT")
    _ensure_column(cur, "order_lines", "delivery_delta_per_t", "delivery_delta_per_t REAL")

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

    # --- Schema migration: add context column (safe if already exists) ---
    try:
        cur.execute("ALTER TABLE presence_events ADD COLUMN context TEXT DEFAULT ''")
    except Exception:
        pass
    
    try:
        cur.execute("ALTER TABLE user_presence ADD COLUMN context TEXT DEFAULT ''")
    except Exception:
        pass

    c.commit()
    c.close()

def list_fert_delivery_options(active_only: bool = False) -> pd.DataFrame:
    c = conn()
    if active_only:
        df = pd.read_sql_query("""
            SELECT option_id, name, delta_per_t, active, created_at_utc, created_by
            FROM fert_delivery_options
            WHERE active = 1
            ORDER BY name ASC
        """, c)
    else:
        df = pd.read_sql_query("""
            SELECT option_id, name, delta_per_t, active, created_at_utc, created_by
            FROM fert_delivery_options
            ORDER BY name ASC
        """, c)
    c.close()
    return df

# ---------------- Fert delivery options ----------------

def save_fert_delivery_options(df: pd.DataFrame, user: str):
    """
    Admin editor save:
      expects cols: name, delta_per_t, active
      - marks all inactive
      - upserts incoming names
    """
    work = df.copy()

    for col in ["name", "delta_per_t"]:
        if col not in work.columns:
            raise ValueError(f"Missing column '{col}' in delivery options editor.")

    work["name"] = work["name"].fillna("").astype(str).str.strip()
    work = work[work["name"] != ""]

    work["delta_per_t"] = pd.to_numeric(work["delta_per_t"], errors="raise")

    if "active" not in work.columns:
        work["active"] = 1
    work["active"] = work["active"].apply(lambda x: 1 if bool(x) else 0).astype(int)

    lowered = work["name"].str.lower()
    if lowered.duplicated().any():
        dups = work.loc[lowered.duplicated(), "name"].tolist()
        raise ValueError(f"Duplicate delivery option names detected: {dups}")

    c = conn()
    cur = c.cursor()

    cur.execute("UPDATE fert_delivery_options SET active = 0;")
    now = utc_now_iso()

    for _, r in work.iterrows():
        nm = r["name"]
        dp = float(r["delta_per_t"])
        ac = int(r["active"])

        cur.execute("SELECT option_id FROM fert_delivery_options WHERE name = ?", (nm,))
        ex = cur.fetchone()
        if ex:
            cur.execute("""
                UPDATE fert_delivery_options
                SET delta_per_t = ?, active = ?
                WHERE name = ?
            """, (dp, ac, nm))
        else:
            cur.execute("""
                INSERT INTO fert_delivery_options (name, delta_per_t, active, created_at_utc, created_by)
                VALUES (?, ?, ?, ?, ?)
            """, (nm, dp, ac, now, user))

    c.commit()
    c.close()

# ---------------- Seed treatments ----------------

def list_seed_treatments(active_only: bool = False) -> pd.DataFrame:
    c = conn()
    if active_only:
        df = pd.read_sql_query("""
            SELECT treatment_id, name, charge_per_t, active, created_at_utc, created_by
            FROM seed_treatments
            WHERE active = 1
            ORDER BY name ASC
        """, c)
    else:
        df = pd.read_sql_query("""
            SELECT treatment_id, name, charge_per_t, active, created_at_utc, created_by
            FROM seed_treatments
            ORDER BY name ASC
        """, c)
    c.close()
    return df


def save_seed_treatments(df: pd.DataFrame, user: str):
    """
    Admin editor save:
      expects cols: name, charge_per_t, active
      we implement as "replace active set":
        - mark all existing as inactive
        - upsert incoming names with new values
    """
    work = df.copy()

    for col in ["name", "charge_per_t"]:
        if col not in work.columns:
            raise ValueError(f"Missing column '{col}' in treatments editor.")

    work["name"] = work["name"].fillna("").astype(str).str.strip()
    work = work[work["name"] != ""]  # drop blank names

    work["charge_per_t"] = pd.to_numeric(work["charge_per_t"], errors="raise")

    if "active" not in work.columns:
        work["active"] = 1
    work["active"] = work["active"].apply(lambda x: 1 if bool(x) else 0).astype(int)

    # Validate no duplicates (case-insensitive)
    lowered = work["name"].str.lower()
    if lowered.duplicated().any():
        dups = work.loc[lowered.duplicated(), "name"].tolist()
        raise ValueError(f"Duplicate treatment names detected: {dups}")

    c = conn()
    cur = c.cursor()

    # Inactivate everything first (soft reset)
    cur.execute("UPDATE seed_treatments SET active = 0;")

    now = utc_now_iso()

    # Upsert by name
    for _, r in work.iterrows():
        nm = r["name"]
        ch = float(r["charge_per_t"])
        ac = int(r["active"])

        # If exists: update, else insert
        cur.execute("SELECT treatment_id FROM seed_treatments WHERE name = ?", (nm,))
        ex = cur.fetchone()
        if ex:
            cur.execute("""
                UPDATE seed_treatments
                SET charge_per_t = ?, active = ?
                WHERE name = ?
            """, (ch, ac, nm))
        else:
            cur.execute("""
                INSERT INTO seed_treatments (name, charge_per_t, active, created_at_utc, created_by)
                VALUES (?, ?, ?, ?, ?)
            """, (nm, ch, ac, now, user))

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

def _snapshot_hash_from_df(df: pd.DataFrame) -> str:
    cols = [
        "Supplier","Product Category","Product","Location","Delivery Window",
        "Price","Sell Price","Unit",
        "Notes","Cost/kg N"
    ]
    tmp = df.copy()

    for c in cols:
        if c not in tmp.columns:
            tmp[c] = ""

    tmp = tmp[cols].copy()

    for c in ["Supplier","Product Category","Product","Location","Delivery Window","Unit","Notes","Cost/kg N"]:
        tmp[c] = tmp[c].fillna("").astype(str).str.strip().str.lower()

    tmp["Price"] = pd.to_numeric(tmp["Price"], errors="coerce").fillna(0.0)
    tmp["Sell Price"] = pd.to_numeric(tmp["Sell Price"], errors="coerce").fillna(0.0)

    tmp = tmp.sort_values(cols).reset_index(drop=True)

    payload = tmp.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
          unit AS "Unit",
          COALESCE(notes, '') AS "Notes",
          COALESCE(cost_per_kg_n, '') AS "Cost/kg N"
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
    source_hash = _snapshot_hash_from_df(work)
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
            str(r.get("Notes", "")).strip(),
            str(r.get("Cost/kg N", "")).strip(),
        ))
    
    cur.executemany("""
        INSERT INTO supplier_prices
        (snapshot_id, supplier, product_category, product, location, delivery_window, price, sell_price, unit, notes, cost_per_kg_n)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    c.commit()
    c.close()
    return snapshot_id

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
          unit AS "Unit",
          COALESCE(notes, '') AS "Notes",
          COALESCE(cost_per_kg_n, '') AS "Cost/kg N"
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
    source_hash = _snapshot_hash_from_df(work)
    row_count = int(len(work))

    c = conn()
    cur = c.cursor()

    # Insert snapshot header
    try:
        cur.execute("""
            INSERT INTO seed_snapshots (snapshot_id, published_at_utc, published_by, source_hash, row_count)
            VALUES (?, ?, ?, ?, ?)
        """, (snapshot_id, published_at, published_by, source_hash, row_count))
    except sqlite3.IntegrityError as e:
        c.close()
        raise ValueError("This seed file has already been published (duplicate source_hash).") from e

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
            str(r.get("Notes", "")).strip(),
            str(r.get("Cost/kg N", "")).strip(),
        ))
    
    cur.executemany("""
        INSERT INTO seed_prices
        (snapshot_id, supplier, product_category, product, location, delivery_window, price, sell_price, unit, notes, cost_per_kg_n)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

def add_margin(scope_type: str, scope_value: str, margin_per_t: float, user: str, role_code: str = "trader"):
    scope_type = str(scope_type).strip().lower()
    if scope_type not in ("category", "product"):
        raise ValueError("scope_type must be 'category' or 'product'.")

    scope_value = str(scope_value).strip()
    if not scope_value:
        raise ValueError("scope_value cannot be empty.")

    role_code = str(role_code).strip().lower()
    if not role_code:
        role_code = "trader"

    c = conn()
    cur = c.cursor()
    cur.execute("""
        INSERT INTO price_margins
        (role_code, scope_type, scope_value, margin_per_t, active, created_at_utc, created_by)
        VALUES (?, ?, ?, ?, 1, ?, ?)
    """, (role_code, scope_type, scope_value, float(margin_per_t), utc_now_iso(), user))
    c.commit()
    c.close()


def list_margins(active_only: bool = True, role_code: str | None = None) -> pd.DataFrame:
    c = conn()

    where = []
    params = []

    if active_only:
        where.append("active = 1")
    if role_code:
        where.append("role_code = ?")
        params.append(str(role_code).strip().lower())

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    df = pd.read_sql_query(f"""
        SELECT margin_id, role_code, scope_type, scope_value, margin_per_t, active, created_at_utc, created_by
        FROM price_margins
        {where_sql}
        ORDER BY margin_id DESC
    """, c, params=params)

    c.close()
    return df


def deactivate_margin(margin_id: int):
    c = conn()
    cur = c.cursor()
    cur.execute("UPDATE price_margins SET active = 0 WHERE margin_id = ?", (int(margin_id),))
    c.commit()
    c.close()


def get_effective_margins(role_code: str = "trader") -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query("""
        SELECT margin_id, role_code, scope_type, scope_value, margin_per_t
        FROM price_margins
        WHERE active = 1 AND role_code = ?
        ORDER BY margin_id DESC
    """, c, params=(str(role_code).strip().lower(),))
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
            str(ln.get("Delivery Method", "")).strip(),
            float(ln.get("Delivery Delta Per T", 0.0) or 0.0),
        ))

    cur.executemany("""
        INSERT INTO order_lines
        (order_id, line_no, product_category, product, location, delivery_window, qty, unit, supplier, base_price, sell_price, delivery_method, delivery_delta_per_t)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        SELECT line_no,
               product_category AS "Product Category",
               product AS "Product",
               location AS "Location",
               delivery_window AS "Delivery Window",
               qty AS "Qty",
               unit AS "Unit",
               supplier AS "Supplier",
               base_price AS "Base Price",
               sell_price AS "Sell Price",
               COALESCE(delivery_method, '') AS "Delivery Method",
               COALESCE(delivery_delta_per_t, 0.0) AS "Delivery Delta Per T"
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


def admin_confirm_order(order_id: str, admin_user: str, admin_note: str = "", expected_version: int | None = None):
    h = get_order_header(order_id)
    if not h:
        raise ValueError("Order not found.")

    _transition_order(
        order_id=order_id,
        action_type="CONFIRM",
        action_by=admin_user,
        expected_version=expected_version,
        admin_note=admin_note,
        payload={"admin_note": admin_note} if admin_note else None,
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

def presence_heartbeat(
    user: str,
    role: str,
    page: str,
    session_id: str,
    context: str = "",
    *,
    sample_seconds: int = 30
):
    """
    Upserts a presence heartbeat for this user+session and appends to presence_events (history).

    sample_seconds:
      - to avoid exploding event volume, we only record a new event if the last recorded event
        for this user+session is older than sample_seconds (default 30s).
    """
    if not user or not session_id:
        return

    user = str(user).strip()
    role = "" if role is None else str(role).strip()
    page = "" if page is None else str(page).strip()
    context = "" if context is None else str(context).strip()
    session_id = str(session_id).strip()

    now = utc_now_iso()

    c = conn()
    cur = c.cursor()

    # 1) presence "online now" upsert (existing behaviour)
    cur.execute("""
        INSERT INTO user_presence (user, session_id, role, page, context, online_since_utc, last_seen_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user, session_id) DO UPDATE SET
            role = excluded.role,
            page = excluded.page,
            context = excluded.context,
            last_seen_utc = excluded.last_seen_utc
    """, (user, session_id, role, page, context, now, now))

    # 2) append-only event log (sampled)
    #    Only record if last event for this user+session is older than sample_seconds
    should_insert = True
    if sample_seconds and int(sample_seconds) > 0:
        cur.execute("""
            SELECT event_at_utc, COALESCE(page,''), COALESCE(context,'')
            FROM presence_events
            WHERE user = ? AND session_id = ?
            ORDER BY event_at_utc DESC
            LIMIT 1
        """, (user, session_id))
        r = cur.fetchone()
        if r:
            last_ts = r[0]
            last_page = r[1] or ""
            last_ctx = r[2] or ""
            try:
                # Always record if the page changed (even inside sample window)
                if page != (last_page or "") or context != (last_ctx or ""):
                    should_insert = True
                else:
                    last_dt = datetime.fromisoformat(last_ts)
                    now_dt = datetime.fromisoformat(now)
                    if (now_dt - last_dt).total_seconds() < int(sample_seconds):
                        should_insert = False
            except Exception:
                should_insert = True

    if should_insert:
        cur.execute("""
            INSERT INTO presence_events (event_at_utc, user, session_id, role, page, context)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (now, user, session_id, role, page, context))

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
    cutoff = utc_cutoff_iso(seconds=online_within_seconds)

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
            MAX(context) AS context,
            MIN(online_since_utc) AS online_since_utc,
            MAX(last_seen_utc) AS last_seen_utc
        FROM user_presence
        WHERE last_seen_utc >= ?
        GROUP BY user
        ORDER BY user ASC
    """, c, params=(cutoff,))
    c.close()
    return df

def list_distinct_presence_users(days: int = 365) -> list[str]:
    """
    Distinct users seen in presence_events within last N days.
    Returns sorted list.
    """
    cutoff = utc_cutoff_iso(days=days)
    c = conn()
    cur = c.cursor()
    cur.execute("""
        SELECT DISTINCT user
        FROM presence_events
        WHERE event_at_utc >= ?
        ORDER BY user ASC
    """, (cutoff,))
    out = [r[0] for r in cur.fetchall()]
    c.close()
    return out


def list_distinct_presence_pages(days: int = 365) -> list[str]:
    """
    Distinct pages seen in presence_events within last N days.
    Returns sorted list.
    """
    cutoff = utc_cutoff_iso(days=int(days))
    c = conn()
    cur = c.cursor()
    cur.execute("""
        SELECT DISTINCT page
        FROM presence_events
        WHERE event_at_utc >= ?
          AND COALESCE(page,'') != ''
        ORDER BY page ASC
    """, (cutoff,))
    out = [r[0] for r in cur.fetchall()]
    c.close()
    return out

def list_distinct_presence_contexts(days: int = 365) -> list[str]:
    cutoff = utc_cutoff_iso(days=int(days))
    c = conn()
    cur = c.cursor()
    cur.execute("""
        SELECT DISTINCT COALESCE(context,'') AS context
        FROM presence_events
        WHERE event_at_utc >= ?
          AND COALESCE(context,'') != ''
        ORDER BY context ASC
    """, (cutoff,))
    out = [r[0] for r in cur.fetchall()]
    c.close()
    return out

def list_presence_events(
    *,
    user: str | None = None,
    role: str | None = None,
    page: str | None = None,
    context: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
    limit: int = 20000,
) -> pd.DataFrame:
    """
    Raw presence_events with filters for admin audit.
    """
    q = """
        SELECT
            event_at_utc,
            user,
            session_id,
            COALESCE(role,'') AS role,
            COALESCE(page,'') AS page,
            COALESCE(context,'') AS context
        FROM presence_events
    """
    where = []
    params: list = []

    if user:
        where.append("user = ?")
        params.append(str(user).strip())
    if role:
        where.append("role = ?")
        params.append(str(role).strip())
    if page:
        where.append("page = ?")
        params.append(str(page).strip())
    if start_utc:
        where.append("event_at_utc >= ?")
        params.append(start_utc)
    if end_utc:
        where.append("event_at_utc <= ?")
        params.append(end_utc)
    if context:
        where.append("context = ?")
        params.append(str(context).strip())

    if where:
        q += " WHERE " + " AND ".join(where)

    q += " ORDER BY event_at_utc DESC"
    q += f" LIMIT {int(limit)}"

    c = conn()
    df = pd.read_sql_query(q, c, params=params)
    c.close()
    return df

def presence_daily_logins(start_utc: str, end_utc: str, *, user: str | None = None, context: str | None = None) -> pd.DataFrame:
    """
    Logins per day = count of distinct session_id by their FIRST seen timestamp.
    """
    params: list = [start_utc, end_utc]

    user_clause = ""
    if user:
        user_clause = " AND user = ? "
        params.append(str(user).strip())

    ctx_clause = ""
    if context:
        ctx_clause = " AND COALESCE(context,'') = ? "
        params.append(str(context).strip())

    q = f"""
    WITH first_seen AS (
        SELECT session_id, user, MIN(event_at_utc) AS first_at
        FROM presence_events
        WHERE event_at_utc >= ? AND event_at_utc <= ?
        {user_clause}
        {ctx_clause}
        GROUP BY session_id, user
    )
    SELECT
        DATE(first_at) AS day,
        COUNT(*) AS logins
    FROM first_seen
    GROUP BY DATE(first_at)
    ORDER BY day ASC;
    """

    c = conn()
    df = pd.read_sql_query(q, c, params=params)
    c.close()
    return df

def presence_daily_page_transitions(start_utc: str, end_utc: str, *, user: str | None = None, context: str | None = None) -> pd.DataFrame:
    """
    Counts page transitions (when page differs from previous event within the session),
    grouped by day and page.
    """
    params: list = [start_utc, end_utc]

    user_clause = ""
    if user:
        user_clause = " AND user = ? "
        params.append(str(user).strip())

    ctx_clause = ""
    if context:
        ctx_clause = " AND COALESCE(context,'') = ? "
        params.append(str(context).strip())
        
    q = f"""
    WITH ordered AS (
        SELECT
            event_at_utc,
            user,
            session_id,
            COALESCE(page,'') AS page,
            COALESCE(context,'') AS context,
            LAG(COALESCE(page,'')) OVER (PARTITION BY session_id ORDER BY event_at_utc) AS prev_page,
            LAG(COALESCE(context,'')) OVER (PARTITION BY session_id ORDER BY event_at_utc) AS prev_context
        FROM presence_events
        WHERE event_at_utc >= ? AND event_at_utc <= ?
        {user_clause}
        {ctx_clause}
    ),
    transitions AS (
        SELECT *
        FROM ordered
        WHERE prev_page IS NULL OR page != prev_page OR context != COALESCE(prev_context,'')
    )
    SELECT
        DATE(event_at_utc) AS day,
        page,
        context,
        COUNT(*) AS navigations
    FROM transitions
    WHERE page != ''
    GROUP BY DATE(event_at_utc), page, context
    ORDER BY day ASC, navigations DESC;
    """

    c = conn()
    df = pd.read_sql_query(q, c, params=params)
    c.close()
    return df

def utc_cutoff_iso(*, seconds: int | None = None, days: int | None = None) -> str:
    dt = datetime.utcnow()
    if seconds is not None:
        dt = dt - timedelta(seconds=int(seconds))
    if days is not None:
        dt = dt - timedelta(days=int(days))
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def presence_session_summary(start_utc: str, end_utc: str, *, user: str | None = None, context: str | None = None) -> pd.DataFrame:
    """
    One row per session:
      - first_at, last_at
      - duration_seconds
      - distinct_pages
      - transitions (page changes)
    """
    params: list = [start_utc, end_utc]

    user_clause = ""
    if user:
        user_clause = " AND user = ? "
        params.append(str(user).strip())

    ctx_clause = ""
    if context:
        ctx_clause = " AND COALESCE(context,'') = ? "
        params.append(str(context).strip())

    q = f"""
    WITH base AS (
        SELECT
            event_at_utc,
            user,
            session_id,
            COALESCE(page,'') AS page,
            LAG(COALESCE(page,'')) OVER (PARTITION BY session_id ORDER BY event_at_utc) AS prev_page
        FROM presence_events
        WHERE event_at_utc >= ? AND event_at_utc <= ?
        {user_clause}
        {ctx_clause}
    ),
    per_session AS (
        SELECT
            user,
            session_id,
            MIN(event_at_utc) AS first_at,
            MAX(event_at_utc) AS last_at,
            COUNT(DISTINCT CASE WHEN page != '' THEN page END) AS distinct_pages,
            SUM(CASE WHEN prev_page IS NULL OR page != prev_page THEN 1 ELSE 0 END) AS transitions
        FROM base
        GROUP BY user, session_id
    )
    SELECT
        user,
        session_id,
        first_at,
        last_at,
        CAST((JULIANDAY(last_at) - JULIANDAY(first_at)) * 86400 AS INTEGER) AS duration_seconds,
        distinct_pages,
        transitions
    FROM per_session
    ORDER BY first_at DESC;
    """

    c = conn()
    df = pd.read_sql_query(q, c, params=params)
    c.close()
    return df

def prune_presence_events(keep_days: int = 180) -> int:
    """
    Deletes old presence_events beyond keep_days. Returns deleted rowcount.
    Call manually or from a periodic admin action.
    """
    cutoff = utc_cutoff_iso(days=keep_days)
    c = conn()
    cur = c.cursor()
    cur.execute("DELETE FROM presence_events WHERE event_at_utc < ?", (cutoff,))
    deleted = cur.rowcount
    c.commit()
    c.close()
    return int(deleted or 0)

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

def list_active_offers(book_code: str) -> pd.DataFrame:
    """
    Active now = active=1 AND starts_at <= now < ends_at
    """
    now = utc_now_iso()
    c = conn()
    df = pd.read_sql_query("""
        SELECT
            offer_id,
            book_code,
            product,
            location,
            delivery_window,
            COALESCE(supplier,'') AS supplier,
            mode,
            value,
            COALESCE(title,'') AS title,
            active,
            starts_at_utc,
            ends_at_utc,
            created_at_utc,
            created_by
        FROM todays_offers
        WHERE book_code = ?
          AND active = 1
          AND starts_at_utc <= ?
          AND ends_at_utc > ?
        ORDER BY ends_at_utc ASC
    """, c, params=(book_code, now, now))
    c.close()
    return df

def list_offers(book_code: str | None = None, active_only: bool = False) -> pd.DataFrame:
    c = conn()
    q = """
        SELECT
            offer_id,
            book_code,
            product,
            location,
            delivery_window,
            COALESCE(supplier,'') AS supplier,
            mode,
            value,
            COALESCE(title,'') AS title,
            active,
            starts_at_utc,
            ends_at_utc,
            created_at_utc,
            created_by
        FROM todays_offers
    """
    where = []
    params = []
    if book_code:
        where.append("book_code = ?")
        params.append(book_code)
    if active_only:
        where.append("active = 1")

    if where:
        q += " WHERE " + " AND ".join(where)

    q += " ORDER BY created_at_utc DESC"

    df = pd.read_sql_query(q, c, params=params)
    c.close()
    return df


def create_offer(
    *,
    book_code: str,
    product: str,
    location: str,
    delivery_window: str,
    supplier: str | None,
    mode: str,
    value: float,
    title: str,
    starts_at_utc: str,
    ends_at_utc: str,
    created_by: str,
):
    book_code = str(book_code).strip().lower()
    if book_code not in ("fert", "seed"):
        raise ValueError("book_code must be 'fert' or 'seed'.")

    product = str(product).strip()
    location = str(location).strip()
    delivery_window = str(delivery_window).strip()
    supplier = None if supplier is None or str(supplier).strip() == "" else str(supplier).strip()

    mode = str(mode).strip().lower()
    if mode not in ("delta", "fixed"):
        raise ValueError("mode must be 'delta' or 'fixed'.")

    value = float(value)
    if value < 0:
        raise ValueError("Offer value must be >= 0.")

    if not product or not location or not delivery_window:
        raise ValueError("product/location/delivery_window cannot be blank.")

    s = datetime.fromisoformat(starts_at_utc)
    e = datetime.fromisoformat(ends_at_utc)
    if e <= s:
        raise ValueError("ends_at_utc must be after starts_at_utc.")

    c = conn()
    cur = c.cursor()
    cur.execute("""
        INSERT INTO todays_offers
        (book_code, product, location, delivery_window, supplier, mode, value, title, active,
         starts_at_utc, ends_at_utc, created_at_utc, created_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
    """, (
        book_code, product, location, delivery_window, supplier,
        mode, value, (title or "").strip(),
        starts_at_utc, ends_at_utc,
        utc_now_iso(), created_by
    ))
    c.commit()
    c.close()


def deactivate_offer(offer_id: int):
    c = conn()
    cur = c.cursor()
    cur.execute("UPDATE todays_offers SET active = 0 WHERE offer_id = ?", (int(offer_id),))
    c.commit()
    c.close()


































