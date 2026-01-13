from __future__ import annotations

from itertools import combinations
import pandas as pd


def tier_charge_per_t(tonnes: float, tiers: pd.DataFrame) -> float:
    """
    Returns the applicable small-lot charge (£/t) for a given tonnage, using
    the tier table. Tiers are assumed to be:
      - global (applies to all suppliers),
      - non-overlapping for active tiers,
      - inclusive on both bounds (min_t <= t <= max_t).
    If max_t is null/NaN, the tier is open-ended.
    """
    t = float(tonnes)
    if t <= 0:
        return 0.0

    # Handle active being bool or int
    active = tiers[tiers["active"].astype(int) == 1].copy()
    active = active.sort_values("min_t")

    eps = 1e-9

    for r in active.to_dict("records"):
        mn = float(r["min_t"])
        mx = r.get("max_t", None)

        # Normalize max_t
        if mx is None or (isinstance(mx, float) and pd.isna(mx)):
            mx = None
        else:
            mx = float(mx)

        if (t + eps) >= mn and (mx is None or (t - eps) <= mx):
            return float(r["charge_per_t"])

    # If not covered by tiers, default to no charge
    return 0.0


def optimise_basket(
    supplier_prices: pd.DataFrame,
    basket: list[dict],
    tiers: pd.DataFrame
) -> dict:
    """
    Optimise a basket subject to:
      - No splitting product lines (each basket line goes to exactly one supplier)
      - Tiered small-lot charges applied PER SUPPLIER based on total tonnes allocated
        to that supplier in the basket.

    Inputs:
      supplier_prices columns (case-sensitive as passed in):
        - Supplier
        - Product
        - Location
        - Delivery Window
        - Price

      basket: list of dicts with keys:
        - Product
        - Location
        - Delivery Window
        - Qty

      tiers columns:
        - min_t
        - max_t (nullable)
        - charge_per_t
        - active (bool or int)

    Output:
      dict with:
        ok: bool
        error: str (if ok=False)
        allocation: list[dict]
        lot_charges: list[dict]
        base_cost: float
        lot_charge_total: float
        total: float
    """
    if not basket:
        return {"ok": False, "error": "Basket is empty."}

    required_sp = {"Supplier", "Product", "Location", "Delivery Window", "Price"}
    missing_sp = required_sp - set(supplier_prices.columns)
    if missing_sp:
        return {"ok": False, "error": f"supplier_prices missing columns: {sorted(missing_sp)}"}

    # Build per-line candidate suppliers
    candidates_by_line: list[tuple[dict, pd.DataFrame]] = []
    all_suppliers: set[str] = set()

    for line in basket:
        prod = line["Product"]
        loc = line["Location"]
        win = line["Delivery Window"]

        subset = supplier_prices[
            (supplier_prices["Product"] == prod) &
            (supplier_prices["Location"] == loc) &
            (supplier_prices["Delivery Window"] == win)
        ][["Supplier", "Price"]].copy()

        if subset.empty:
            return {"ok": False, "error": f"No supplier prices for {prod} @ {loc} {win}"}

        subset = subset.sort_values("Price", ascending=True)
        candidates_by_line.append((line, subset))
        all_suppliers.update(subset["Supplier"].tolist())

    all_suppliers = sorted(all_suppliers)
    k = len(all_suppliers)
    if k == 0:
        return {"ok": False, "error": "No suppliers available for this basket."}

    best = None

    # Enumerate subsets of suppliers.
    # For MVP, cap subset size at 15 to avoid combinatorial blow-ups.
    # If supplier universe grows, prune suppliers per line instead.
    for r in range(1, min(k, 15) + 1):
        for S in combinations(all_suppliers, r):
            S = set(S)

            allocation = []
            tonnes_by_supplier = {s: 0.0 for s in S}
            base_cost = 0.0

            feasible = True
            for line, subset in candidates_by_line:
                sub2 = subset[subset["Supplier"].isin(S)]
                if sub2.empty:
                    feasible = False
                    break

                chosen = sub2.iloc[0]
                s = str(chosen["Supplier"])
                price = float(chosen["Price"])
                qty = float(line["Qty"])

                allocation.append({
                    "Product": line["Product"],
                    "Location": line["Location"],
                    "Delivery Window": line["Delivery Window"],
                    "Qty": qty,
                    "Supplier": s,
                    "Price": price,
                    "Line Cost": qty * price
                })

                tonnes_by_supplier[s] += qty
                base_cost += qty * price

            if not feasible:
                continue

            # Tiered small-lot charges per supplier (based on tonnes allocated to that supplier)
            lot_charge_total = 0.0
            lot_charges = []

            for s, t in tonnes_by_supplier.items():
                if t <= 0:
                    continue

                cpt = tier_charge_per_t(t, tiers)
                if cpt > 0:
                    c = t * cpt
                    lot_charge_total += c
                    lot_charges.append({
                        "Supplier": s,
                        "Tonnes": t,
                        "Charge £/t": cpt,
                        "Lot Charge": c
                    })

            total = base_cost + lot_charge_total

            if best is None or total < best["total"]:
                best = {
                    "total": total,
                    "base_cost": base_cost,
                    "lot_charge_total": lot_charge_total,
                    "allocation": allocation,
                    "lot_charges": lot_charges
                }

    if best is None:
        return {"ok": False, "error": "No feasible supplier set found."}

    return {"ok": True, **best}

