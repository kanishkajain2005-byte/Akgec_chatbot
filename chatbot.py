import os
import json
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

QA_FILE = os.path.join("data", "qa_pairs.json")

# Lower threshold because TF-IDF scores are smaller
SIMILARITY_THRESHOLD = 0.15

_vectorizer = None
_qa_pairs = None
_question_vectors = None


# 🔧 Synonym expansion (lightweight intelligence)
SYNONYMS = {
    "fee": ["fees", "cost", "price"],
    "placement": ["placements", "jobs", "recruitment"],
    "admission": ["admissions", "apply", "enrollment"],
    "hostel": ["accommodation", "dorm"],
    "course": ["courses", "program", "branch"],
}


def normalize(text: str) -> str:
    """Lowercase + remove special characters"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def expand_query(text: str) -> str:
    """Expand query with synonyms"""
    words = text.split()
    expanded = []

    for word in words:
        expanded.append(word)
        for key, vals in SYNONYMS.items():
            if word == key or word in vals:
                expanded.extend([key] + vals)

    return " ".join(expanded)


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

    questions = [normalize(qa["question"]) for qa in _qa_pairs]

    # 🔥 Improved TF-IDF setup
    _vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # words + phrases
        stop_words="english"
    )

    _question_vectors = _vectorizer.fit_transform(questions)


def generate_response(query: str) -> dict:
    _load_resources()

    query = normalize(query)
    query = expand_query(query)

    query_vec = _vectorizer.transform([query])

    # Cosine similarity (dot product for normalized vectors)
    scores = (_question_vectors @ query_vec.T).toarray().flatten()

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    # Second-best score for confidence check
    sorted_scores = np.sort(scores)
    second_best = float(sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0

    # 🔥 Unknown query handling
    if best_score < SIMILARITY_THRESHOLD or (best_score - second_best) < 0.05:
        return {
            "answer": "This information is not available. Please visit the official website for accurate details.",
            "sources": ["https://www.akgec.ac.in"],
            "similarity_score": round(best_score, 4),
        }

    best_qa = _qa_pairs[best_idx]

    return {
        "answer": best_qa["answer"],
        "sources": ["https://www.akgec.ac.in"],
        "similarity_score": round(best_score, 4),
    }