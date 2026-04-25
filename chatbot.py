import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

QA_FILE = os.path.join("data", "qa_pairs.json")
SIMILARITY_THRESHOLD = 0.2  # lower for TF-IDF

_vectorizer = None
_qa_pairs = None
_question_vectors = None


def _load_resources():
    global _vectorizer, _qa_pairs, _question_vectors

    if _vectorizer is not None:
        return

    print("Loading QA data...")

    if not os.path.exists(QA_FILE):
        raise FileNotFoundError(f"{QA_FILE} not found")

    with open(QA_FILE, "r", encoding="utf-8") as f:
        _qa_pairs = json.load(f)

    print(f"Loaded {len(_qa_pairs)} QA pairs")

    questions = [qa["question"] for qa in _qa_pairs]

    # TF-IDF instead of embeddings
    _vectorizer = TfidfVectorizer()
    _question_vectors = _vectorizer.fit_transform(questions)


def generate_response(query: str) -> dict:
    _load_resources()

    # Transform query
    query_vec = _vectorizer.transform([query])

    # Cosine similarity (sklearn handles it)
    scores = (_question_vectors @ query_vec.T).toarray().flatten()

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