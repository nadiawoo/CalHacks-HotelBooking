#!/usr/bin/env python3
"""
Robust data cleaning for ML/AI
- Standardizes dtypes, parses text fields, fills missing coords from neighborhood centroids
- Produces: cleaned_listings, amenities_kv / amenities_onehot, and ml_dataset VIEW
- Optional: set EXPORT_PARQUET=1 to write ml_dataset.parquet
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "nyc_airbnb.db"

# ---------------------------
# Helpers
# ---------------------------
MONEY_RE = re.compile(r"[\$,]")
PCT_RE = re.compile(r"%+$")
BATH_FLOAT_RE = re.compile(r"(\d+(?:\.\d+)?)")


def money_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace(MONEY_RE, "", regex=True).str.strip(), errors="coerce")


def percent_to_float(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace(PCT_RE, "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date.astype("string")


def resolve_col(con: sqlite3.Connection, table: str, cands: List[str]) -> str:
    cols = pd.read_sql(f"PRAGMA table_info({table});", con)["name"].tolist()
    for c in cands:
        if c in cols:
            return c
    raise RuntimeError(f"None of {cands} present in {table}")


def fill_latlon_from_neighborhood(df: pd.DataFrame, con: sqlite3.Connection) -> pd.DataFrame:
    # Handle either spelling in neighborhood_coords and alias to `neighborhood`
    try:
        neigh = pd.read_sql("SELECT neighborhood, lat, lon FROM neighborhood_coords;", con)
    except Exception:
        neigh = pd.read_sql("SELECT neighbourhood AS neighborhood, lat, lon FROM neighborhood_coords;", con)

    # Ensure one row per neighborhood (duplicates in coords would explode rows on merge)
    neigh = (
        neigh.groupby("neighborhood", as_index=False)
              .agg({"lat": "mean", "lon": "mean"})
    )

    # Merge using whatever neighborhood column exists in the incoming df
    left_key = "neighborhood" if "neighborhood" in df.columns else ("neighbourhood" if "neighbourhood" in df.columns else None)
    if left_key is None:
        raise RuntimeError("No neighborhood/neighbourhood column found in listings frame for centroid merge")

    df = df.merge(neigh, left_on=left_key, right_on="neighborhood", how="left", suffixes=("", "_centroid"))
    df["latitude"]  = df["latitude"].fillna(df["lat"])  # prefer listing value; fall back to centroid
    df["longitude"] = df["longitude"].fillna(df["lon"]) 
    df.drop(columns=[c for c in ["lat","lon"] if c in df.columns], inplace=True)
    return df


def cap_outliers(series: pd.Series, lo_q: float = 0.01, hi_q: float = 0.99) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    lo, hi = x.quantile(lo_q), x.quantile(hi_q)
    return x.clip(lower=lo, upper=hi)


# ---------------------------
# Cleaning
# ---------------------------

def build_cleaned_listings(con: sqlite3.Connection) -> None:
    print("🧽 Building cleaned_listings…")
    # Resolve neighborhood column defensively (but data_loading ensures 'neighborhood')
    neigh_col = resolve_col(con, "listings", ["neighborhood", "neighbourhood_cleansed", "neighbourhood"])

    raw = pd.read_sql(
        f"""
        SELECT 
          id, listing_url, name, description, neighborhood_overview,
          {neigh_col} AS neighborhood, latitude, longitude,
          property_type, room_type, accommodates, bathrooms_text,
          bedrooms, beds, amenities, price, minimum_nights, maximum_nights,
          host_id, host_name, host_since, host_location, host_is_superhost,
          host_response_time, host_response_rate, host_identity_verified,
          number_of_reviews, reviews_per_month,
          review_scores_rating, estimated_occupancy_l365d, estimated_revenue_l365d,
          first_review, last_review,
          picture_url, instant_bookable
        FROM listings;
        """,
        con,
    )

    # Force numeric ids; drop null ids
    raw["id"] = pd.to_numeric(raw["id"], errors="coerce").astype("Int64")
    before_id = len(raw)
    raw = raw.dropna(subset=["id"]).copy()
    if len(raw) != before_id:
        print(f"• Dropped {before_id - len(raw)} rows with null/non-numeric ids")

    # Deduplicate by id with useful tie-breakers
    _num_reviews = pd.to_numeric(raw.get("number_of_reviews", pd.Series([], dtype="float")), errors="coerce").fillna(0)
    _has_coords = raw.get("latitude").notna() & raw.get("longitude").notna()
    _last_review_dt = pd.to_datetime(raw.get("last_review"), errors="coerce")

    raw = raw.assign(_has_coords=_has_coords, _num_reviews=_num_reviews, _last_review_dt=_last_review_dt)
    before = len(raw)
    raw = (
        raw.sort_values(by=["_has_coords", "_num_reviews", "_last_review_dt"], ascending=[False, False, False])
           .drop_duplicates(subset=["id"], keep="first")
    )
    removed = before - len(raw)
    if removed > 0:
        print(f"• Deduplicated listings: removed {removed} duplicate rows (kept {len(raw)} unique ids)")

    # Fill missing coords from neighborhood centroids (make sure coords table is unique per neighborhood)
    raw = fill_latlon_from_neighborhood(raw, con)

    # Safety: ensure 1 row per id after the merge
    dups_after_merge = raw.duplicated(subset=["id"]).sum()
    if dups_after_merge:
        raw = (
            raw.sort_values(by=["_has_coords", "_num_reviews", "_last_review_dt"], ascending=[False, False, False])
               .drop_duplicates(subset=["id"], keep="first")
        )
        print(f"• Fixed {dups_after_merge} duplicate ids introduced during merge")

    # Drop helper columns now
    raw = raw.drop(columns=["_has_coords", "_num_reviews", "_last_review_dt"])  

    # Type coercions
    raw["price"] = money_to_float(raw["price"])  # $1,234.00 -> 1234.0
    raw["host_response_rate"] = percent_to_float(raw.get("host_response_rate", pd.Series([], dtype="string")))

    for c in ["latitude", "longitude", "bedrooms", "beds", "review_scores_rating",
              "estimated_occupancy_l365d", "estimated_revenue_l365d"]:
        if c in raw.columns:
            raw[c] = to_float(raw[c])
    for c in ["accommodates", "minimum_nights", "maximum_nights", "number_of_reviews"]:
        if c in raw.columns:
            raw[c] = to_int(raw[c])
    for c in ["host_since", "first_review", "last_review"]:
        if c in raw.columns:
            raw[c] = to_date(raw[c])

    # bathrooms_text -> bathrooms (float)
    def parse_bathrooms(x: str) -> float | None:
        if not isinstance(x, str):
            return None
        m = BATH_FLOAT_RE.search(x)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        # handle "Half-bath" as 0.5
        if "half" in x.lower():
            return 0.5
        return None
    raw["bathrooms"] = raw.get("bathrooms_text").apply(parse_bathrooms)

    # Booleans
    def to_bool01(x):
        if isinstance(x, str):
            t = x.strip().lower()
            if t in {"t","true","yes","y","1"}: return 1
            if t in {"f","false","no","n","0"}: return 0
        if isinstance(x, (int, float)):
            return int(bool(x))
        return pd.NA
    raw["host_is_superhost"] = raw["host_is_superhost"].apply(to_bool01).astype("Int64")
    raw["host_identity_verified"] = raw["host_identity_verified"].apply(to_bool01).astype("Int64")
    raw["instant_bookable"] = raw["instant_bookable"].apply(to_bool01).astype("Int64")

    # Fill missing coords from neighborhood centroids
    # (already done above)

    # Engineered helpers
    raw["price_per_bed"]   = (raw["price"] / raw["beds"]).replace([pd.NaT, pd.NA, math.inf, -math.inf], pd.NA)
    raw["price_per_guest"] = (raw["price"] / raw["accommodates"]).replace([pd.NaT, pd.NA, math.inf, -math.inf], pd.NA)

    # Cap outliers
    for c in ["price", "price_per_bed", "price_per_guest"]:
        raw[c] = cap_outliers(raw[c])

    dup_count = raw.duplicated(subset=["id"]).sum()
    if dup_count:
        raise RuntimeError(f"Refusing to write cleaned_listings: {dup_count} duplicate ids remain")

    raw.to_sql("cleaned_listings", con, if_exists="replace", index=False)
    con.executescript(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_cleaned_id ON cleaned_listings(id);
        CREATE INDEX IF NOT EXISTS idx_cleaned_price ON cleaned_listings(price);
        CREATE INDEX IF NOT EXISTS idx_cleaned_neighborhood ON cleaned_listings(neighborhood);
        CREATE INDEX IF NOT EXISTS idx_cleaned_coords ON cleaned_listings(latitude, longitude);
        """
    )
    con.commit()
    print(f"✅ cleaned_listings written: {len(raw)} rows")


def build_amenities_tables(con: sqlite3.Connection) -> None:
    print("🧩 Building amenity tables (kv + onehot)…")
    an = pd.read_sql("SELECT id, amenities_norm_json, amenity_flags_json FROM amenity_norm;", con)

    kv_rows: List[Dict[str, str]] = []
    for _, r in an.iterrows():
        lid = int(r["id"])
        try:
            items = json.loads(r["amenities_norm_json"]) or []
        except Exception:
            items = []
        for a in items:
            kv_rows.append({"id": lid, "amenity": str(a)})
    kv = pd.DataFrame(kv_rows, columns=["id","amenity"]) if kv_rows else pd.DataFrame(columns=["id","amenity"])
    kv.to_sql("amenities_kv", con, if_exists="replace", index=False)

    if not kv.empty:
        onehot = (
            kv.assign(v=1)
              .pivot_table(index="id", columns="amenity", values="v", fill_value=0, aggfunc="max")
              .reset_index()
        )
        onehot.to_sql("amenities_onehot", con, if_exists="replace", index=False)
        con.executescript("CREATE UNIQUE INDEX IF NOT EXISTS ux_amenities_onehot_id ON amenities_onehot(id);")

    con.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_amenities_kv_id ON amenities_kv(id);
        CREATE INDEX IF NOT EXISTS idx_amenities_kv_amenity ON amenities_kv(amenity);
        """
    )
    con.commit()
    print(f"✅ amenities_kv rows: {len(kv_rows)} | onehot columns: {0 if kv.empty else len(onehot.columns)-1}")


def build_ml_view(con: sqlite3.Connection) -> None:
    print("🧱 Creating ml_dataset VIEW…")
    con.executescript(
        """
        DROP VIEW IF EXISTS ml_dataset;
        CREATE VIEW ml_dataset AS
        SELECT 
          L.id,
          L.name,
          L.description,
          L.neighborhood,
          L.latitude,
          L.longitude,
          L.property_type,
          L.room_type,
          L.accommodates,
          L.bedrooms,
          L.beds,
          L.price,
          L.price_per_bed,
          L.price_per_guest,
          L.minimum_nights,
          L.maximum_nights,
          L.number_of_reviews,
          L.reviews_per_month,
          L.review_scores_rating,
          L.instant_bookable,
          AF.amenity_flags_json,
          LF.style_tags,
          LE.experience_tags
        FROM cleaned_listings L
        LEFT JOIN amenity_norm AF ON AF.id = L.id
        LEFT JOIN listing_features LF ON LF.id = L.id
        LEFT JOIN listing_experience LE ON LE.id = L.id;
        """
    )
    con.commit()
    print("✅ ml_dataset VIEW ready")


def maybe_export_parquet(con: sqlite3.Connection) -> None:
    if os.getenv("EXPORT_PARQUET", "0") != "1":
        return
    print("📦 Exporting ml_dataset.parquet …")
    df = pd.read_sql("SELECT * FROM ml_dataset;", con)
    df.to_parquet(HERE / "ml_dataset.parquet", index=False)
    print("✅ Wrote ml_dataset.parquet")


# ---------------------------
# Main
# ---------------------------

def main():
    print(f"[data_cleaning] Using DB: {DB_PATH}")
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=OFF;")

    build_cleaned_listings(con)
    build_amenities_tables(con)
    build_ml_view(con)
    maybe_export_parquet(con)

    con.close()
    print("🎉 Data cleaning complete.")


if __name__ == "__main__":
    main()