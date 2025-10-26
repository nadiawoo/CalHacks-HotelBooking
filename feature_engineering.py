#!/usr/bin/env python3
"""
Feature generation pipeline (refactored)
- Separates style, amenities, and experience features
"""

import os
import re
import json
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- Text/lexicon helpers (style-focused) ----------
RE_HTML = re.compile(r"<[^>]+>")
RE_NUM = re.compile(r"\d+")
RE_DUP_SUBSTR = re.compile(r"(\b\w{3,}\b).*(\1)")  # e.g., monthmonth, skylightskylight
JUNK_SUBSTRINGS = {"nbsp", "br", "http", "https", "airbnb", "booking", "stay", "thank"}

# Curated stylistic lexicon (design cues only)
STYLE_LEXICON = {
    "boho", "bohemian", "industrial", "minimalist", "mid century", "mid-century",
    "modern", "contemporary", "rustic", "farmhouse", "vintage", "eclectic",
    "scandinavian", "art deco", "loft", "studio", "prewar", "pre-war",
    "artsy", "heritage", "classic", "brownstone"
}

# Building/architectural features we want to keep when present

BUILDING_FEATURES = {
    "exposed brick", "brick wall", "skylight", "bay window", "high ceiling", "vaulted ceiling",
    "pyramid", "atrium", "rooftop", "terrace", "balcony", "garden", "backyard",
    "fireplace", "walkup", "walk-up", "elevator", "doorman"
}

# ---------- Experiential lexicons ----------
COMMON_NAMES = {
    # a small but effective default list; host_name tokens are also removed per-listing
    "jennifer","irina","cynthia","john","michael","sarah","emily","alex",
    "anna","lisa","maria","james","robert","linda","mark","paul","kevin",
    "david","daniel","andrew","stephanie","laura","matthew","julia","peter",
}

EXPERIENCE_WHITELIST = {
    # subjective/experiential cues we *want* to surface
    "friendly","welcoming","responsive","helpful","communicative","respectful",
    "easy","smooth","quick","fast","self check in","check in","check out",
    "clean","spotless","tidy","quiet","safe","cozy","romantic",
    "family","kids","spacious","views","view","balcony","rooftop",
    "location","neighborhood","near subway","close to subway","near metro",
    "walkable","walk to","short walk","restaurants","coffee","bars","shopping",
    "convenient","comfortable","good for families","business trip","girls night",
}

EXPERIENCE_BLOCK = {
    # words/phrases that add little information in reviews
    "br","nbsp","place","apartment","unit","host name","airbnb","stay","thank",
}

def remove_name_tokens(text: str, host_name: Optional[str]) -> str:
    """Remove per-listing host name tokens AND common first names from text."""
    t = text
    # remove host-name tokens for this listing
    if host_name:
        tokens = re.findall(r"[A-Za-z]+", host_name.lower())
        tokens = [w for w in tokens if len(w) >= 3]
        if tokens:
            pattern = r"\b(" + "|".join(map(re.escape, set(tokens))) + r")\b"
            t = re.sub(pattern, " ", t, flags=re.IGNORECASE)
    # remove common first names
    pattern2 = r"\b(" + "|".join(sorted(COMMON_NAMES)) + r")\b"
    t = re.sub(pattern2, " ", t, flags=re.IGNORECASE)
    # collapse spaces
    return re.sub(r"\s+", " ", t)

def experiential_phrase_ok(p: str) -> bool:
    p = normalize_phrase(p)
    if not p:
        return False
    if any(b in p for b in EXPERIENCE_BLOCK):
        return False
    if looks_noisy(p):
        return False
    # keep 2-3 word phrases only
    wc = len(p.split())
    if wc < 2 or wc > 4:
        return False
    # must include at least one whitelist cue or obvious experiential keyword
    if any(w in p for w in EXPERIENCE_WHITELIST):
        return True
    # additional generic experiential cues
    EXTRA_CUES = ("host", "check in", "checkout", "location", "subway", "restaurant", "walk", "safe", "quiet", "clean")
    return any(w in p for w in EXTRA_CUES)

def normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = RE_HTML.sub(" ", s)
    s = RE_NUM.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def to_hashtag(phrase: str) -> str:
    phrase = phrase.strip().lower()
    parts = [p for p in phrase.split() if p]
    if not parts:
        return ""
    return "#" + parts[0] + "".join(w.capitalize() for w in parts[1:])

def looks_noisy(token: str) -> bool:
    t = token.lower()
    if any(x in t for x in JUNK_SUBSTRINGS):
        return True
    if RE_DUP_SUBSTR.search(t):
        return True
    if not re.search(r"[a-z]", t):
        return True
    # overly long single tokens are usually junky
    if len(t) > 24:
        return True
    return False


# ---------- DB Path ----------
HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "nyc_airbnb.db"

def assert_db_has_listings() -> None:
    with sqlite3.connect(str(DB_PATH)) as con:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", con)["name"].tolist()
        missing = []
        for tbl in ["listings", "reviews", "neighborhood_coords"]:
            if tbl not in tables:
                missing.append(tbl)
        if missing:
            raise RuntimeError(f"Missing required tables in {DB_PATH}: {missing}")
        print(f"✅ All required tables found in {DB_PATH}: listings, reviews, neighborhood_coords")


# ---------- Amenity Normalization ----------
AMENITY_MAP = {
    "wifi": ["wi-fi", "internet"],
    "kitchen": ["stove", "oven", "cookware", "microwave", "full kitchen"],
    "parking": ["garage", "car park", "free parking", "paid parking"],
    "air_conditioning": ["ac", "air conditioning", "a/c"],
    "heating": ["heater", "central heating"],
    "washer": ["washing machine", "laundry"],
    "dryer": ["tumble dryer"],
    "bathtub": ["tub", "bath tub"],
    "pool": ["swimming pool", "outdoor pool", "indoor pool"],
    "gym": ["fitness center"],
    "desk": ["workspace", "work desk"],
    "accessible": ["wheelchair", "step-free", "elevator"],
}
FLAG_GROUPS = {
    "work_friendly": ["wifi", "desk"],
    "family_friendly": ["bathtub", "kitchen"],
    "stay_in": ["kitchen", "heating", "air_conditioning"],
    "leisure": ["pool", "gym"],
    "car_friendly": ["parking"],
}

def normalize_amenities(raw_text: Optional[str]) -> Tuple[List[str], Dict[str, int]]:
    text = (raw_text or "").lower()
    norm = set()
    for canon, variants in AMENITY_MAP.items():
        if any(v in text for v in ([canon] + variants)):
            norm.add(canon)
    flags = {g: int(any(t in norm for t in tags)) for g, tags in FLAG_GROUPS.items()}
    return sorted(norm), flags


# ---------- Hash Utility ----------
def content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()


# ---------- Text Cleaning ----------
SUBJECTIVE_ADJ = {
    "cozy", "spacious", "bright", "comfortable", "lovely", "nice",
    "beautiful", "great", "amazing", "perfect", "wonderful", "charming",
    "fantastic", "excellent", "peaceful", "quiet", "adorable"
}
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")

def clean_text(s: Optional[str], remove_subjective: bool = False) -> str:
    if not s:
        return ""
    s = RE_HTML.sub(" ", s)
    s = RE_NUM.sub(" ", s)
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    toks = TOKEN_RE.findall(s.lower())
    if remove_subjective:
        toks = [t for t in toks if t not in SUBJECTIVE_ADJ]
    # drop junk substrings
    toks = [t for t in toks if t not in JUNK_SUBSTRINGS]
    return " ".join(toks)


# ---------- TF-IDF Extraction ----------
def extract_tfidf_tags(texts: List[str], top_k: int = 6, ngram_range=(1, 2)) -> List[List[str]]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=ngram_range, min_df=3, max_df=0.8)
    tfidf = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    tags_all = []
    for i in range(tfidf.shape[0]):
        row = tfidf.getrow(i)
        if row.nnz == 0:
            tags_all.append([])
            continue
        idx = row.indices
        vals = row.data
        top_idx = vals.argsort()[::-1][:top_k]
        tags_all.append([features[idx[j]] for j in top_idx])
    return tags_all


# ---------- Style Feature Builder ----------
def build_style_tags(con: sqlite3.Connection) -> None:
    print("🧩 Building stylistic tags (style + limited building features)...")
    df = pd.read_sql("SELECT id, name, description FROM listings;", con)

    # Keep a raw lowercase blob for substring checks, plus a cleaned one for TF-IDF
    df["raw"] = (df["name"].fillna("") + " " + df["description"].fillna("")).str.lower().apply(normalize_phrase)
    df["text"] = (df["name"].fillna("") + " " + df["description"].fillna("")).apply(
        lambda x: clean_text(x, remove_subjective=True)
    )

    # Extract tf-idf terms (1-2 grams)
    tfidf_terms = extract_tfidf_tags(df["text"].tolist(), top_k=10, ngram_range=(1, 2))

    out = []
    for (lid, raw, terms) in zip(df["id"], df["raw"], tfidf_terms):
        # 1) Curated style words present in tf-idf terms
        style_hits = []
        for t in terms:
            t_norm = normalize_phrase(t)
            # exact or substring match against style lexicon
            if t_norm in STYLE_LEXICON or any(t_norm == k or t_norm in k or k in t_norm for k in STYLE_LEXICON):
                if not looks_noisy(t_norm):
                    style_hits.append(t_norm)
        # de-dup while preserving order
        seen = set()
        style_hits = [x for x in style_hits if not (x in seen or seen.add(x))]

        # 2) Building features present in raw text (substring search)
        features = []
        for feat in BUILDING_FEATURES:
            if feat in raw:
                features.append(feat)
        # limit to at most 2 building features to avoid over-dominance
        features = features[:2]

        # 3) Assemble hashtags: prefer up to 4 style + up to 2 building features
        chosen = []
        for t in style_hits:
            tag = to_hashtag(t)
            if tag and not looks_noisy(tag):
                chosen.append(tag)
            if len(chosen) >= 4:
                break
        for f in features:
            tag = to_hashtag(f)
            if tag and tag not in chosen and not looks_noisy(tag):
                chosen.append(tag)
            if len(chosen) >= 6:
                break

        hashtags = " ".join(chosen)
        out.append({
            "id": lid,
            "content_hash": content_hash(" ".join(style_hits + features)),
            "style_tags": hashtags,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS listing_features (
          id INTEGER PRIMARY KEY,
          content_hash TEXT,
          style_tags TEXT,
          updated_at TEXT
        )
        """
    )
    con.executemany(
        """
        INSERT INTO listing_features (id, content_hash, style_tags, updated_at)
        VALUES (:id, :content_hash, :style_tags, :updated_at)
        ON CONFLICT(id) DO UPDATE SET
          content_hash=excluded.content_hash,
          style_tags=excluded.style_tags,
          updated_at=excluded.updated_at
        """,
        out,
    )
    con.commit()
    print(f"✅ Generated {len(out)} style tag rows.")


# ---------- Amenity Feature Builder ----------
def build_amenities(con: sqlite3.Connection) -> None:
    print("🏗️ Normalizing amenities...")
    df = pd.read_sql("SELECT id, amenities FROM listings;", con)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        norm, flags = normalize_amenities(row["amenities"])
        rows.append({
            "id": row["id"],
            "amenities_norm_json": json.dumps(norm, ensure_ascii=False),
            "amenity_flags_json": json.dumps(flags, ensure_ascii=False),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    con.execute("""
        CREATE TABLE IF NOT EXISTS amenity_norm (
          id INTEGER PRIMARY KEY,
          amenities_norm_json TEXT,
          amenity_flags_json TEXT,
          updated_at TEXT
        )
    """)
    con.executemany("""
        INSERT INTO amenity_norm (id, amenities_norm_json, amenity_flags_json, updated_at)
        VALUES (:id, :amenities_norm_json, :amenity_flags_json, :updated_at)
        ON CONFLICT(id) DO UPDATE SET
          amenities_norm_json=excluded.amenities_norm_json,
          amenity_flags_json=excluded.amenity_flags_json,
          updated_at=excluded.updated_at
    """, rows)
    con.commit()
    print(f"✅ Normalized amenities for {len(rows)} listings.")


# ---------- Experience Tag Builder ----------
def build_experience_tags(con: sqlite3.Connection, batch_size: int = 3000) -> None:
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", con)["name"].tolist()
    if "reviews" not in tables:
        print("⚠️ Skipping experience tags: no reviews table.")
        return
    print("🌟 Building experiential tags from reviews (batched by listing)…")

    # Allow long concatenations
    try:
        con.execute("PRAGMA group_concat_max_len = 2000000;")
    except Exception:
        pass

    # Build listing -> host_name map to strip names
    host_df = pd.read_sql("SELECT id, host_name FROM listings;", con)
    host_map = dict(zip(host_df["id"], host_df["host_name"]))

    # Get DISTINCT listing ids first; we'll page over these IDs
    id_df = pd.read_sql("SELECT DISTINCT listing_id AS id FROM reviews ORDER BY id;", con)
    ids_all = id_df["id"].tolist()
    n_listings = len(ids_all)
    print(f"Found {n_listings} listings with reviews.")

    # Ensure output table exists
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS listing_experience (
          id INTEGER PRIMARY KEY,
          experience_tags TEXT,
          updated_at TEXT
        )
        """
    )

    processed = 0
    pbar = tqdm(total=n_listings, desc="reviews→experience")

    # Batch over the distinct IDs
    for start in range(0, n_listings, batch_size):
        batch_ids = ids_all[start:start + batch_size]
        if not batch_ids:
            break
        ids_csv = ",".join(map(str, batch_ids))

        rev = pd.read_sql(
            f"""
            SELECT listing_id AS id, GROUP_CONCAT(comments, ' ') AS all_reviews
            FROM reviews
            WHERE listing_id IN ({ids_csv})
            GROUP BY listing_id
            ORDER BY listing_id
            """,
            con,
        )
        if rev.empty:
            pbar.update(len(batch_ids))
            processed += len(batch_ids)
            continue

        # Clean text per listing: remove host names + noise BEFORE TF‑IDF
        cleaned_texts = []
        ids = rev["id"].tolist()
        for lid, txt in zip(ids, rev["all_reviews"].fillna("").tolist()):
            txt = clean_text(txt)  # html/digits/junk
            txt = remove_name_tokens(txt, host_map.get(lid))
            cleaned_texts.append(txt)

        # Vectorize this batch only (2-3 grams)
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(2, 3), min_df=2, max_df=0.9)
        tfidf = vectorizer.fit_transform(cleaned_texts)
        feats = vectorizer.get_feature_names_out()

        # Extract top phrases per listing
        out_rows = []
        for idx, lid in enumerate(ids):
            row = tfidf.getrow(idx)
            if row.nnz == 0:
                phrases = []
            else:
                inds = row.indices
                vals = row.data
                order = vals.argsort()[::-1]
                raw_phrases = [feats[i] for i in inds[order][:24]]  # take more, filter later
                # Filter & de‑dup
                seen = set()
                phrases = []
                for p in raw_phrases:
                    if experiential_phrase_ok(p):
                        if p not in seen:
                            seen.add(p)
                            phrases.append(p)
                    if len(phrases) >= 8:
                        break

            out_rows.append({
                "id": int(lid),
                "experience_tags": json.dumps(phrases, ensure_ascii=False),
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })

        # UPSERT this batch
        con.executemany(
            """
            INSERT INTO listing_experience (id, experience_tags, updated_at)
            VALUES (:id, :experience_tags, :updated_at)
            ON CONFLICT(id) DO UPDATE SET
              experience_tags=excluded.experience_tags,
              updated_at=excluded.updated_at
            """,
            out_rows,
        )
        con.commit()

        pbar.update(len(batch_ids))
        processed += len(batch_ids)

    pbar.close()
    print(f"✅ Generated/updated experience tags for ~{processed} listings.")


# ---------- Pipeline Runner ----------
def run_pipeline():
    print("[feature_pipeline] Using unified DB:", DB_PATH)
    assert_db_has_listings()
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=OFF;")

    build_style_tags(con)
    build_amenities(con)
    build_experience_tags(con)

    con.close()
    print("🎉 All feature sets generated successfully.")


if __name__ == "__main__":
    run_pipeline()
