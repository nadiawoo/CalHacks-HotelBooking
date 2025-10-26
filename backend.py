#!/usr/bin/env python3
"""
Backend entrypoint for the AI service
- Reuses the FastAPI app from ai_service.py
- Adds CORS for local dev / Vercel
- Exposes a friendly root route

Run:
  uvicorn backend:app --reload --port 8000
"""
from __future__ import annotations
import os
import pandas as pd
import requests
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from ai_service import app, RecommendRequest, prefilter_candidates, local_rank

load_dotenv()
import json
from typing import Any, Dict, List
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
from ai_service import app, RecommendRequest, prefilter_candidates, local_rank

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- CORS ---
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://*.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ALLOW_ALL_ORIGINS", "0") == "1" else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Claude rerank helper ---
def _llm_rerank(preferences: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Rerank the top local results using Gemini only (gemini-2.0-flash-lite)."""
    if df.empty:
        return df

    # Dynamically select relevant features based on user prompt
    user_prompt = preferences.pop('user_prompt', None)
    # List all available columns in the DataFrame
    all_features = list(df.columns)
    # Always include id, name, neighborhood, price, accommodates
    base_features = ["id", "name", "neighborhood", "price", "accommodates"]
    # Add style_tags, experience_tags, amenity_flags_json if present
    extra_features = [f for f in ["style_tags", "experience_tags", "amenity_flags_json"] if f in all_features]
    # If user prompt mentions food, add description, neighborhood_overview
    if user_prompt and ("food" in user_prompt.lower() or "restaurant" in user_prompt.lower()):
        extra_features += [f for f in ["description", "neighborhood_overview"] if f in all_features]
    # If user prompt mentions subway, add latitude, longitude
    if user_prompt and "subway" in user_prompt.lower():
        extra_features += [f for f in ["latitude", "longitude"] if f in all_features]
    # If user prompt mentions host, add host_name, host_is_superhost
    if user_prompt and "host" in user_prompt.lower():
        extra_features += [f for f in ["host_name", "host_is_superhost"] if f in all_features]
    # If user prompt mentions style, add property_type, style_tags
    if user_prompt and "style" in user_prompt.lower():
        extra_features += [f for f in ["property_type", "style_tags"] if f in all_features]
    # Remove duplicates
    features = list(dict.fromkeys(base_features + extra_features))
    items = df.head(min(len(df), 12)).to_dict(orient="records")
    listings = [
        {k: r.get(k) for k in features if k in r}
        for r in items
    ]
    prompt = (
        (f"User request: {user_prompt}\n" if user_prompt else "") +
        "You are an expert travel assistant. Given the user's preferences and the following accommodation listings, "
        "rerank the listings in best order for the user. Respond ONLY with a JSON array of listing ids in best order.\n"
        f"User preferences: {json.dumps(preferences)}\n"
        f"Listings: {json.dumps(listings)}"
    )

    if GEMINI_API_KEY:
        try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
            body = {
                "contents": [
                    {"parts": [{"text": json.dumps(prompt)}]}
                ]
            }
            headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
            resp = requests.post(endpoint, headers=headers, json=body, timeout=15)
            resp.raise_for_status()
            j = resp.json()
            print("[Gemini API raw response]", json.dumps(j, indent=2))
            text_candidates = []
            for k in ("candidates", "outputs", "response", "predictions"):
                if k in j and isinstance(j[k], list):
                    for c in j[k]:
                        if isinstance(c, dict):
                            if "output" in c and isinstance(c["output"], str):
                                text_candidates.append(c["output"]) 
                            if "content" in c and isinstance(c["content"], list):
                                for elt in c["content"]:
                                    if isinstance(elt, dict) and elt.get("text"):
                                        text_candidates.append(elt.get("text"))
                            if "text" in c and isinstance(c["text"], str):
                                text_candidates.append(c["text"])
            if not text_candidates:
                for key in ("output", "text", "candidates_text"):
                    if key in j and isinstance(j[key], str):
                        text_candidates.append(j[key])

            text = "\n".join(text_candidates).strip()
            # Extract JSON array from Markdown/code block if present (multi-line safe)
            if text.startswith("```json") or text.startswith("```"):
                # Remove code block markers and any leading/trailing whitespace
                text = text.replace("```json", "").replace("```", "").strip()
            # Find the first [ ... ] block (multi-line safe)
            import re
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                array_str = match.group(0)
                try:
                    arr = json.loads(array_str)
                except Exception as e:
                    print(f"[Gemini API parse error] Could not parse JSON array: {e}\nRaw text: {array_str}")
                    return df
                order = []
                for x in arr:
                    try:
                        order.append(int(x))
                    except Exception:
                        continue
                if order:
                    rank_map = {lid: i for i, lid in enumerate(order)}
                    out = df.copy()
                    out["score_llm"] = out["id"].map(lambda x: rank_map.get(int(x), 10_000))
                    return out.sort_values(["score_llm", "score_local"], ascending=[True, False])
            else:
                print(f"[Gemini API response] No valid JSON array found in: {text}")
                return df
        except Exception as e:
            print(f"[Gemini API exception] {e}")
            return df

    return df

_claude_rerank = _llm_rerank


# --- LLM-powered recommend endpoint ---
@app.post("/recommend_llm")
async def recommend_llm(req: RecommendRequest):
    try:
        base = prefilter_candidates(req, top_k=30)
        if base.empty:
            return {"results": [], "count": 0, "llm": bool(GEMINI_API_KEY)}
        ranked = local_rank(base, req)
        # Pass user_prompt from request if present
        prefs = req.model_dump()
        if hasattr(req, 'user_prompt') and req.user_prompt:
            prefs['user_prompt'] = req.user_prompt
        reranked = _llm_rerank(prefs, ranked)
        out = reranked.head(req.return_n or 10)
        cols = [
            "id","name","neighborhood","price","accommodates","room_type",
            "review_scores_rating","number_of_reviews","style_tags","experience_tags","amenity_flags_json","picture_url"
        ]
        cols = [c for c in cols if c in out.columns]
        return {"results": out[cols].to_dict(orient="records"), "count": int(len(reranked)), "llm": bool(GEMINI_API_KEY), "model": GEMINI_MODEL}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend_claude")
async def recommend_claude_alias(req: RecommendRequest):
    return await recommend_llm(req)

# --- Friendly root ---
@app.get("/")
async def root():
    return {
        "ok": True,
        "service": "CalHacks Travel AI",
        "endpoints": ["/health", "/recommend", "/compare", "/itinerary"],
    }


@app.get("/health_llm")
async def health_llm():
    """Simple health check for the Gemini reranker."""
    if not GEMINI_API_KEY:
        return {"ok": False, "available": False, "reason": "Gemini API key not configured (set GEMINI_API_KEY)"}
    try:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        body = {
            "contents": [
                {"parts": [{"text": "Return only: OK"}]}
            ]
        }
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
        resp = requests.post(endpoint, headers=headers, json=body, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        text_candidates = []
        for k in ("candidates", "outputs", "response", "predictions"):
            if k in j and isinstance(j[k], list):
                for c in j[k]:
                    if isinstance(c, dict):
                        if "output" in c and isinstance(c["output"], str):
                            text_candidates.append(c["output"]) 
                        if "content" in c and isinstance(c["content"], list):
                            for elt in c["content"]:
                                if isinstance(elt, dict) and elt.get("text"):
                                    text_candidates.append(elt.get("text"))
                        if "text" in c and isinstance(c["text"], str):
                            text_candidates.append(c["text"])
        if not text_candidates:
            for key in ("output", "text", "candidates_text"):
                if key in j and isinstance(j[key], str):
                    text_candidates.append(j[key])
        text = "\n".join(text_candidates).strip()
        ok = text and "OK" in text
        return {"ok": ok, "available": True, "model": GEMINI_MODEL, "response": text}
    except Exception as e:
        return {"ok": False, "available": True, "error": str(e)}


@app.get("/keys_status")
async def keys_status():
    """Return whether Gemini API key is present in the running process (masked)."""
    def mask(k: str | None) -> str | None:
        if not k:
            return None
        s = str(k).strip()
        if len(s) <= 8:
            return s[0:1] + "..." + s[-1:]
        return s[:4] + "..." + s[-4:]
    return {
        "gemini_present": bool(GEMINI_API_KEY),
        "gemini_key_masked": mask(GEMINI_API_KEY),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
