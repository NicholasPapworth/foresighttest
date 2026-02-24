# src/routing.py
from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
from typing import Optional, Tuple

import streamlit as st

# You can point this to your self-hosted OSRM later:
# e.g. OSRM_BASE_URL="http://your-osrm-host:5000"
OSRM_BASE_URL = os.environ.get("OSRM_BASE_URL", "https://router.project-osrm.org")

# Free UK postcode geocoder (external dependency; replace with local dataset later if you want)
POSTCODE_API_BASE = os.environ.get("POSTCODE_API_BASE", "https://api.postcodes.io")


def _http_get_json(url: str, timeout: int = 8) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ForesightRouting/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_uk_postcode(postcode: str) -> str:
    if not postcode:
        return ""
    pc = postcode.strip().upper()
    pc = re.sub(r"\s+", "", pc)  # remove spaces
    # reinsert single space before last 3 chars (UK format)
    if len(pc) > 3:
        pc = pc[:-3] + " " + pc[-3:]
    return pc


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 14)  # 14 days
def geocode_uk_postcode(postcode: str) -> Optional[Tuple[float, float]]:
    """
    Returns (lat, lon) for a UK postcode, or None if invalid/unresolved.
    Cached aggressively.
    """
    pc = normalize_uk_postcode(postcode)
    if not pc:
        return None

    url = f"{POSTCODE_API_BASE}/postcodes/{urllib.parse.quote(pc)}"
    try:
        data = _http_get_json(url)
        if not data or data.get("status") != 200:
            return None
        res = data.get("result") or {}
        lat = res.get("latitude")
        lon = res.get("longitude")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)
    except Exception:
        return None


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Earth radius in miles
    R = 3958.7613
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)  # 7 days
def road_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    """
    Returns road distance in miles using OSRM.
    Falls back to haversine if OSRM fails (still returns a number).
    Cached.
    """
    try:
        # OSRM expects lon,lat
        url = (
            f"{OSRM_BASE_URL}/route/v1/driving/"
            f"{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}"
            f"?overview=false&alternatives=false&steps=false"
        )
        data = _http_get_json(url, timeout=8)
        routes = data.get("routes") or []
        if not routes:
            raise RuntimeError("No routes")
        meters = routes[0].get("distance")
        if meters is None:
            raise RuntimeError("No distance")
        miles = float(meters) / 1609.344
        return miles
    except Exception:
        # Soft fallback to great-circle miles
        try:
            return float(haversine_miles(lat1, lon1, lat2, lon2))
        except Exception:
            return None


def distance_store_to_postcode(store_lat: float, store_lon: float, delivery_postcode: str) -> Optional[float]:
    geo = geocode_uk_postcode(delivery_postcode)
    if geo is None:
        return None
    dlat, dlon = geo
    return road_miles(store_lat, store_lon, dlat, dlon)
