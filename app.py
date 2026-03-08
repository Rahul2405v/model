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
    try:
        response = requests.post(EMBEDDING_API_URL, json={"text": text}, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except Exception as e:
        logger.error(f"Embedding API failed: {e}")
        raise RuntimeError("Embedding generation failed")


def _get_groq_client():
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_api_url = os.getenv("GROQ_API_URL")

    if not groq_api_key and not groq_api_url:
        return None

    try:
        from groq import Groq
    except Exception:
        return None

    try:
        if groq_api_url:
            return Groq(api_key=groq_api_key, base_url=groq_api_url)
        return Groq(api_key=groq_api_key)
    except Exception:
        return None


def _extract_text_from_groq_response(resp):
    try:
        choices = getattr(resp, "choices", None)

        if choices:
            first = choices[0]

            if isinstance(first, dict):
                return first["message"]["content"]

            msg = getattr(first, "message", None)
            if msg:
                return getattr(msg, "content", None)

        return None
    except Exception:
        return None


def _call_groq_for_json(prompt_text: str):
    groq_client = _get_groq_client()

    if not groq_client:
        return None

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt_text}],
        )

        text = _extract_text_from_groq_response(resp)

        if not text:
            return None

        try:
            return json.loads(text)
        except:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])

        return None

    except Exception as e:
        logger.warning(f"Groq call failed: {e}")
        return None


def identify_price_range_from_prompt(prompt: str):

    instruction = (
        "Return JSON with min_inr and max_inr price range.\n"
        f"User request:\n{prompt}"
    )

    out = _call_groq_for_json(instruction)

    if not out:
        return None

    try:
        return float(out["min_inr"]), float(out["max_inr"])
    except:
        return None


def _parse_price_inr(value):

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        s = str(value)
        cleaned = "".join(ch for ch in s if (ch.isdigit() or ch in ".,"))

        cleaned = cleaned.replace(",", "")

        if cleaned == "":
            return None

        return float(cleaned)

    except:
        return None


def filter_products_by_price(products, price_range):

    minv, maxv = price_range

    out = []

    for p in products:

        price_val = _parse_price_inr(
            p.get("Price") or p.get("price") or p.get("priceINR")
        )

        if price_val is not None and (minv <= price_val <= maxv):
            out.append(p)

    return out


def serialize_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc


def run_vector_search(query_vector, num_candidates=50, top_k=10):

    try:

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "embedding": 0,
                    "description": 0,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        cleaned = []

        for doc in results:
            d = dict(doc)
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

    body = request.get_json(force=True, silent=True)

    if not body or "text" not in body:
        return jsonify({"error": "'text' field required"}), 400

    text = body["text"]

    num_candidates = int(body.get("num_candidates", 50))
    top_k = int(body.get("top_k", 10))

    try:

        query_vector = get_embedding(text)

        docs = run_vector_search(query_vector, num_candidates, top_k)

        logger.info("Vector search returned %d docs", len(docs))

        price_range = identify_price_range_from_prompt(text)

        logger.info("Price range detected: %s", str(price_range))

        if price_range:
            docs = filter_products_by_price(docs, price_range)

        return jsonify({
            "results": docs,
            "count": len(docs)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":

    port = int(os.getenv("PORT", 5000))

    app.run(host="0.0.0.0", port=port, debug=True)
