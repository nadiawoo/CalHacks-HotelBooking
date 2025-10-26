import geopandas as gpd
import pandas as pd
import sqlite3
import ast
import re

def money_to_float(s):
    return (
        pd.Series(s, dtype="string")
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace({"": None})
        .astype(float)
    )

def to_json_str(x):
    try:
        arr = ast.literal_eval(x) if isinstance(x, str) else []
        return "[" + ",".join([f'"{str(i).strip()}"' for i in arr]) + "]"
    except Exception:
        return "[]"

def clean_text(s):
    if isinstance(s, str):
        # strip invalid surrogate code points
        return re.sub(r'[\ud800-\udfff]', '', s)
    return s

# Connect or create database
con = sqlite3.connect("nyc_airbnb.db")
con.execute("PRAGMA journal_mode=WAL;")
con.execute("PRAGMA synchronous=OFF;")

# Load listings (keep ALL 33 columns you selected)
listings = pd.read_csv("database/listings.csv.gz", low_memory=False, encoding_errors="replace")

keep_cols = [
    # Core info
    "id", "listing_url", "name", "description", "neighborhood_overview",
    "neighbourhood_cleansed", "latitude", "longitude",
    "property_type", "room_type", "accommodates", "bathrooms_text",
    "bedrooms", "beds", "amenities", "price",

    # Host info
    "host_id", "host_name", "host_since", "host_location",
    "host_is_superhost", "host_response_time", "host_response_rate",
    "host_identity_verified",

    # Performance
    "number_of_reviews", "reviews_per_month",
    "review_scores_rating", "estimated_occupancy_l365d",
    "estimated_revenue_l365d", "first_review", "last_review",

    # Display / misc
    "picture_url", "instant_bookable",

    # nights (you referenced these in earlier code)
    "minimum_nights", "maximum_nights",
]
df = listings[keep_cols].rename(columns={"neighbourhood_cleansed": "neighborhood"}).copy()
df["price"] = money_to_float(df["price"])
df["amenities"] = df["amenities"].apply(to_json_str)

for c in ["name", "description", "neighborhood", "bathrooms_text", "amenities", "picture_url"]:
    if c in df.columns:
        df[c] = df[c].apply(clean_text)

# Write WIDE table (all 33 columns) — no slimming
df.to_sql("listings", con, if_exists="replace", index=False)

# Load reviews unchanged
reviews = pd.read_csv("database/reviews.csv.gz", low_memory=False, encoding_errors="replace")
reviews.to_sql("reviews", con, if_exists="replace", index=False)

# Indexes (uniqueness + speed)
con.executescript("""
CREATE UNIQUE INDEX IF NOT EXISTS ux_listings_id ON listings(id);

CREATE INDEX IF NOT EXISTS idx_listings_price           ON listings(price);
CREATE INDEX IF NOT EXISTS idx_listings_accommodates    ON listings(accommodates);
CREATE INDEX IF NOT EXISTS idx_listings_neighborhood    ON listings(neighborhood);
CREATE INDEX IF NOT EXISTS idx_listings_superhost       ON listings(host_is_superhost);
CREATE INDEX IF NOT EXISTS idx_listings_instant         ON listings(instant_bookable);

CREATE INDEX IF NOT EXISTS idx_reviews_listing          ON reviews(listing_id);
CREATE INDEX IF NOT EXISTS idx_reviews_date             ON reviews(date);
""")

con.commit()
con.close()
print("Loaded listings (all columns) + reviews, and added indexes.")


## adding geo data to sqlite
# Load geojson + csv
gdf = gpd.read_file("database/neighbourhoods.geojson")
neigh_csv = pd.read_csv("database/neighbourhoods.csv")

# Combine geojson geometry with CSV names
geo_df = gdf.merge(neigh_csv, on="neighbourhood", how="left")
geo_df["centroid"] = geo_df.geometry.centroid
geo_df["lat"] = geo_df.centroid.y
geo_df["lon"] = geo_df.centroid.x

# Push into SQLite
con = sqlite3.connect("nyc_airbnb.db")
geo_df[["neighbourhood", "lat", "lon"]].to_sql("neighborhood_coords", con, if_exists="replace", index=False)
con.close()
