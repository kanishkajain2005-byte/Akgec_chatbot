import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
QA_FILE = os.path.join("data", "qa_pairs.json")

SIMILARITY_THRESHOLD = 0.4

_embedder = None
_qa_pairs = None
_question_embeddings = None


def _load_resources():
    global _embedder, _qa_pairs, _question_embeddings

    if _embedder is not None:
        return

    print("Loading embedding model...")
    _embedder = SentenceTransformer(EMBEDDING_MODEL)

    if not os.path.exists(QA_FILE):
        raise FileNotFoundError(f"{QA_FILE} not found")

    with open(QA_FILE, "r", encoding="utf-8") as f:
        _qa_pairs = json.load(f)

    print(f"Loaded {len(_qa_pairs)} QA pairs")

    questions = [qa["question"] for qa in _qa_pairs]
    _question_embeddings = _embedder.encode(questions, show_progress_bar=False)

    norms = np.linalg.norm(_question_embeddings, axis=1, keepdims=True)
    _question_embeddings = _question_embeddings / norms


def generate_response(query: str) -> dict:
    _load_resources()

    query_vec = _embedder.encode([query], show_progress_bar=False)[0]
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        return {
            "answer": "Sorry, I could not understand your question.",
            "sources": [],
            "similarity_score": 0.0,
        }

    query_vec = query_vec / query_norm

    scores = np.dot(_question_embeddings, query_vec)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < SIMILARITY_THRESHOLD:
        return {
            "answer": "I don't have that information. Please check the official website.",
            "sources": ["https://www.akgec.ac.in"],
            "similarity_score": round(best_score, 4),
        }

    best_qa = _qa_pairs[best_idx]

    return {
        "answer": best_qa["answer"],
        "sources": ["https://www.akgec.ac.in"],
        "similarity_score": round(best_score, 4),
    }