import os
import logging
from flask import Flask, request, jsonify
from pymongo import MongoClient
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://embedding-model-dtv9.vercel.app/embed")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "productsDB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "products")

# ------------------------------------------------------------
# MongoDB Connection
# ------------------------------------------------------------
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION]

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
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

def serialize_doc(doc):
    """Convert ObjectId and keep rest of fields"""
    doc["_id"] = str(doc["_id"])
    return doc

def run_vector_search(query_vector, num_candidates=50, limit=10):
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
            },
            {
                "$project": {
                    "embedding": 0,       # Exclude embedding
                    "description": 0,     # Exclude description
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        return [serialize_doc(doc) for doc in results]

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

# ------------------------------------------------------------
# Flask App
# ------------------------------------------------------------
app = Flask(__name__)

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
        return jsonify({"results": docs, "count": len(docs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
