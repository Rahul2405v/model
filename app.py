import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import json
from typing import List, Dict, Any, Optional, Tuple
import requests
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://embedding-model-dtv9.vercel.app/embed")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "productsDB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "products")


client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION]

def get_embedding(text: str):
    """Get embedding vector from external API"""
    try:
        response = requests.post(EMBEDDING_API_URL, json={"text": text}, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except Exception as e:
        logger.error(f"Embedding API failed: {e}")
        raise RuntimeError("Embedding generation failed")


def _get_groq_client():
    """Lazily instantiate a Groq client if possible. Returns None on failure or missing keys."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_api_url = os.getenv("GROQ_API_URL")
    if not groq_api_key and not groq_api_url:
        return None
    try:
        from groq import Groq  # type: ignore
    except Exception:
        return None

    try:
        if groq_api_url:
            client = Groq(api_key=groq_api_key, base_url=groq_api_url)
            logger.debug("Groq client created with base_url")
            return client
        client = Groq(api_key=groq_api_key)
        logger.debug("Groq client created with api_key")
        return client
    except TypeError:
        try:
            return Groq(groq_api_key)
        except Exception:
            try:
                return Groq()
            except Exception:
                return None
    except Exception:
        # any other failure
        logger.debug("Failed to instantiate Groq client", exc_info=True)
        return None


def _extract_text_from_groq_response(resp: Any) -> Optional[str]:
    """Try multiple shapes to extract textual content from Groq SDK response."""
    try:
        # choices -> [ { message: { content: ... } } ]
        choices = getattr(resp, "choices", None)
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    return msg.get("content")
                return None
            else:
                msg = getattr(first, "message", None)
                if msg:
                    if isinstance(msg, dict):
                        return msg.get("content")
                    return getattr(msg, "content", None)
                return getattr(first, "text", None)

        # fallback if resp is dict
        if isinstance(resp, dict):
            try:
                return resp["choices"][0]["message"]["content"]
            except Exception:
                return None
    except Exception:
        return None
    try:
        return str(resp)
    except Exception:
        return None


def _call_groq_for_json(prompt_text: str, timeout: int = 8) -> Optional[Dict[str, Any]]:
    """Call Groq via SDK (preferred) or HTTP fallback and try to parse JSON response."""
    groq_api_url = os.getenv("GROQ_API_URL")
    groq_api_key = os.getenv("GROQ_API_KEY")

    groq_client = _get_groq_client()
    if groq_client:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_text}],
                timeout=timeout,
            )
        except Exception as e:
            logger.warning("Groq SDK call failed: %s", e)
            return None
        text = _extract_text_from_groq_response(resp)
        logger.debug("_call_groq_for_json - sdk raw text: %s", text)
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            # If parsing fails, try to find JSON substring
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(text[start : end + 1])
            except Exception:
                return None
        # no valid JSON
        return None

    # HTTP fallback
    if not groq_api_url:
        return None
    headers = {"Content-Type": "application/json"}
    if groq_api_key:
        headers["Authorization"] = f"Bearer {groq_api_key}"
    try:
        resp = requests.post(groq_api_url, json={"prompt": prompt_text}, headers=headers, timeout=timeout)
        resp.raise_for_status()
        logger.debug("_call_groq_for_json - http raw json: %s", resp.text)
        return resp.json()
    except Exception as e:
        logger.warning("Groq HTTP fallback failed: %s", e)
        return None


def identify_price_range_from_prompt(prompt: str) -> Optional[Tuple[float, float]]:
    """Ask Groq to provide a likely price range in INR for the user's prompt.

    Returns (min_inr, max_inr) or None if not available.
    """
    instruction = (
        "You are given a user request. Return a JSON object with numeric fields 'min_inr' and 'max_inr'"
        " indicating the likely price range in Indian Rupees that matches the request."
        " If unsure, return null for both. Output ONLY valid JSON, e.g. {\"min_inr\":100,\"max_inr\":2000}."
        f"\n\nUser request:\n{prompt}"
    )
    out = _call_groq_for_json(instruction)
    if not out:
        return None
    try:
        minv = out.get("min_inr")
        maxv = out.get("max_inr")
        if minv is None or maxv is None:
            return None
        return float(minv), float(maxv)
    except Exception:
        return None


def identify_category_from_description(description: str, categories: List[str]) -> Optional[str]:
    """Ask Groq to pick the best matching category from a list for the given description.

    Returns category string from categories or None.
    """
    # limit categories length
    cats_json = json.dumps(categories[:50])
    instruction = (
        "Given the list of categories and a product description, return a JSON object {\"category\": <one of the categories>}"
        " selecting the best matching category. Output only JSON.\n\n"
        f"Categories: {cats_json}\n\nDescription: {description}"
    )
    out = _call_groq_for_json(instruction)
    if not out:
        return None
    try:
        cat = out.get("category")
        if cat and cat in categories:
            return cat
        # try case-insensitive match
        if cat:
            for c in categories:
                if isinstance(c, str) and c.lower() == str(cat).lower():
                    return c
    except Exception:
        return None
    return None


def _parse_price_inr(value: Any) -> Optional[float]:
    """Try to parse various price formats into a float INR value."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value)
        # Remove common currency symbols and letters
        cleaned = "".join(ch for ch in s if (ch.isdigit() or ch in ".,"))
        cleaned = cleaned.replace(",", "")
        if cleaned == "":
            return None
        return float(cleaned)
    except Exception:
        return None


def _normalize_db_category(name: str) -> str:
    """Normalize a DB collection name like 'cameraDB' -> 'camera'."""
    if not name:
        return ""
    n = str(name).strip()
    # remove trailing 'DB' or 'db' if present
    if n.lower().endswith("db"):
        n = n[: -2]
    # remove common suffix/prefix characters and whitespace
    return n.strip()


def load_available_categories_from_env_or_docs(docs: List[Dict[str, Any]]) -> List[str]:
    """Return a list of available categories.

    Priority:
    - If CATEGORIES_DB env var provided (comma-separated DB names), use it (normalize names and drop 'products').
    - Otherwise, derive from documents' productCategory fields.
    """
    env_val = os.getenv("CATEGORIES_DB")
    cats: List[str] = []
    if env_val:
        for part in env_val.split(","):
            n = _normalize_db_category(part)
            if not n:
                continue
            if n.lower() == "products":
                continue
            if n not in cats:
                cats.append(n)
        if cats:
            logger.info("Loaded categories from CATEGORIES_DB env: %s", cats)
            return cats

    # Fallback: derive from docs
    for d in docs:
        cat = d.get("productCategory")
        if not cat:
            continue
        n = str(cat).strip()
        if n.lower() == "products":
            continue
        if n not in cats:
            cats.append(n)
    logger.info("Derived categories from documents: %s", cats)
    return cats


def filter_products_by_price(products: List[Dict[str, Any]], price_range: Tuple[float, float]) -> List[Dict[str, Any]]:
    """Annotate each product with parsed price and price_in_range, and return only those within range."""
    minv, maxv = price_range
    out = []
    for p in products:
        # Prefer the 'Price' field (capital P) as your dataset uses it; fallback to other common names
        price_val = _parse_price_inr(p.get("Price") or p.get("priceINR") or p.get("price") or p.get("price_inr"))
        # Only include the original product dict if the parsed price is within range
        if price_val is not None and (minv <= price_val <= maxv):
            out.append(p)
    return out

def serialize_doc(doc):
    """Convert ObjectId and keep rest of fields"""
    doc["_id"] = str(doc["_id"])
    return doc

def run_vector_search(query_vector, num_candidates=50, limit=15):
    """MongoDB Vector Search"""
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": limit,
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        cleaned = []
        for doc in results:
            # copy to avoid mutating underlying cursor/document
            d = dict(doc)
            # Remove embedding and description fields if present (not needed in responses)
            if "embedding" in d:
                d.pop("embedding", None)
            if "description" in d:
                d.pop("description", None)
            cleaned.append(serialize_doc(d))
        return cleaned

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

app = Flask(__name__)

CORS_ALLOWED = os.getenv("CORS_ALLOWED_ORIGINS", "*")
CORS(app, resources={r"/*": {"origins": CORS_ALLOWED}})

@app.route("/search", methods=["POST"])
def search():
    """Handle POST /search requests"""
    body = request.get_json(force=True, silent=True)
    if not body or "text" not in body:
        return jsonify({"error": "'text' field required"}), 400

    text = body["text"]
    num_candidates = int(body.get("num_candidates", 50))
    limit = int(body.get("limit", 10))

    try:
        query_vector = get_embedding(text)
        docs = run_vector_search(query_vector, num_candidates, limit)
        logger.info("Vector search returned %d docs", len(docs))

        # Identify a likely price range from the user's prompt (Groq) and filter by Price if available.
        groq_client = _get_groq_client()
        logger.info("Groq client available: %s", bool(groq_client))
        price_range = identify_price_range_from_prompt(text)
        logger.info("identify_price_range_from_prompt returned: %s", str(price_range))

        if price_range:
            logger.info("Filtering %d docs by price range %s", len(docs), price_range)
            filtered = filter_products_by_price(docs, price_range)
            logger.info("Filtered down to %d docs after price filter", len(filtered))
        else:
            filtered = docs

        # Return only original product documents (no added fields) as JSON
        results = [dict(p) for p in filtered]
        resp_obj = {"results": results, "count": len(results)}
        return jsonify(resp_obj)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
