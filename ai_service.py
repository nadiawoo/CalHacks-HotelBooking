#!/usr/bin/env python3
"""
ai_service.py
- Core FastAPI app + minimal data helpers used by backend.py
- Provides RecommendRequest model, prefilter_candidates, local_rank
- Uses SQLite `ml_dataset` VIEW produced by data_cleaning.py

This file exists so the backend can import a stable set of
helpers without duplicating logic. Keep it small and testable.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ------------------ Config ------------------
HERE = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("DB_PATH", HERE / "nyc_airbnb.db"))
TOP_K_DEFAULT = 30  # number of candidates to pull from SQL

# ------------------ DB helpers ------------------

def get_con() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at: {DB_PATH}")
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def sql_in_params(values: List[Any], start_index: int = 0) -> Tuple[str, Dict[str, Any]]:
    """Build named placeholders (:p0,:p1,...) and param dict for an IN () clause."""
    params: Dict[str, Any] = {}
    placeholders: List[str] = []
    for i, v in enumerate(values):
        key = f"p{start_index + i}"
        placeholders.append(":" + key)
        params[key] = v
    return ",".join(placeholders), params

# ------------------ API Models ------------------
class RecommendRequest(BaseModel):
    city: Optional[str] = "New York"
    guests: Optional[int] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    neighborhoods: Optional[List[str]] = None
    room_types: Optional[List[str]] = None
    must_have_amenities: Optional[List[str]] = None  # canonical tags (e.g., wifi,kitchen)
    preferred_styles: Optional[List[str]] = None     # e.g., ["industrial","boho"]
    trip_purpose: Optional[str] = None               # e.g., "family", "remote work"
    return_n: Optional[int] = 10

# ------------------ Candidate Generation ------------------

def prefilter_candidates(req: RecommendRequest, top_k: int = TOP_K_DEFAULT) -> pd.DataFrame:
    """Query the `ml_dataset` view and return a small candidate pool.
    Adds a light score (computed in Python) to bias toward highly-rated, well-reviewed, good-value listings.
    """
    con = get_con()
    params: Dict[str, Any] = {}
    where = ["1=1"]

    if req.guests:
        where.append("accommodates >= :guests")
        params["guests"] = int(req.guests)
    if req.min_price is not None:
        where.append("price >= :min_price")
        params["min_price"] = float(req.min_price)
    if req.max_price is not None:
        where.append("price <= :max_price")
        params["max_price"] = float(req.max_price)

    if req.neighborhoods:
        ph, extra = sql_in_params(req.neighborhoods, start_index=len(params))
        where.append(f"neighborhood IN ({ph})")
        params.update(extra)
    if req.room_types:
        ph, extra = sql_in_params(req.room_types, start_index=len(params))
        where.append(f"room_type IN ({ph})")
        params.update(extra)

    sql = f"""
        SELECT *
        FROM ml_dataset
        WHERE {' AND '.join(where)}
        LIMIT :top_k;
    """
    params["top_k"] = int(top_k)

    try:
        df = pd.read_sql(sql, con, params=params)
    finally:
        con.close()

    # Compute score_prefilter in Python (SQLite lacks LOG by default)
    if not df.empty:
        rating = pd.to_numeric(df.get("review_scores_rating"), errors="coerce").fillna(0.0) / 5.0
        nrev = pd.to_numeric(df.get("number_of_reviews"), errors="coerce").fillna(0.0)
        price = pd.to_numeric(df.get("price"), errors="coerce")
        inv_price = price.apply(lambda p: (1.0 / p) if pd.notna(p) and p > 0 else 0.0)
        df["score_prefilter"] = rating + np.log1p(nrev) + inv_price
        df = df.sort_values("score_prefilter", ascending=False).head(int(top_k))

    return df

# ------------------ Local Ranking ------------------

def local_rank(df: pd.DataFrame, req: RecommendRequest) -> pd.DataFrame:
    """Blend several light-weight signals into a final score.
    Returns a new DataFrame sorted by `score_local` descending.
    """
    if df.empty:
        return df

    def norm(col: str) -> pd.Series:
        x = pd.to_numeric(df[col], errors="coerce")
        lo, hi = x.quantile(0.05), x.quantile(0.95)
        rng = max(1e-6, hi - lo)
        return (x.clip(lo, hi) - lo) / rng

    rating = norm("review_scores_rating")
    reviews = norm("number_of_reviews")
    price_inv = 1 - norm("price")  # cheaper is better

    def amenity_match(j: Optional[str]) -> float:
        if not req.must_have_amenities:
            return 0.0
        if not j:
            return -1.0
        try:
            flags = json.loads(j)
        except Exception:
            return -1.0
        present = 0
        for k in req.must_have_amenities:
            present += int(k in flags) or int(flags.get(k, 0) == 1)
        return present / max(1, len(req.must_have_amenities))

    def style_match(t: Optional[str]) -> float:
        if not req.preferred_styles or not t:
            return 0.0
        tags = t.lower()
        hits = sum(1 for s in req.preferred_styles if s.lower() in tags)
        return hits / max(1, len(req.preferred_styles))

    def purpose_match(t: Optional[str]) -> float:
        if not req.trip_purpose:
            return 0.0
        text = ((t or "") + " " + req.trip_purpose).lower()
        buckets = {
            "family": ["family", "kids"],
            "remote": ["remote", "work", "desk", "wifi"],
            "romantic": ["romantic", "quiet", "cozy"],
            "nightlife": ["bars", "nightlife", "restaurants"],
        }
        keys: List[str] = []
        for k, words in buckets.items():
            if k in req.trip_purpose.lower():
                keys = words
                break
        if not keys:
            keys = req.trip_purpose.split()
        return sum(1 for w in keys if w in text) / max(1, len(keys))

    df = df.copy()
    df["score_local"] = (
        0.35 * rating +
        0.20 * reviews +
        0.25 * price_inv +
        0.10 * df["amenity_flags_json"].apply(amenity_match) +
        0.05 * df["style_tags"].apply(style_match) +
        0.05 * df["experience_tags"].apply(purpose_match)
    )
    return df.sort_values("score_local", ascending=False)

# ------------------ FastAPI app ------------------
app = FastAPI(title="Travel AI Service", version="0.1")

@app.get("/health")
def health():
    return {"ok": True, "db": DB_PATH.name}

# Re-export symbols for backend.py convenience
__all__ = [
    "app",
    "DB_PATH",
    "RecommendRequest",
    "prefilter_candidates",
    "local_rank",
]