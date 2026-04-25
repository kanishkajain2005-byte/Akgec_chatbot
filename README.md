# AKGEC College Chatbot

An AI-powered chatbot for [Ajay Kumar Garg Engineering College](https://www.akgec.ac.in) built with:
- **Scraping**: `requests` + `BeautifulSoup` + `PyMuPDF` (PDF support)
- **Embeddings**: `sentence-transformers` (local, free — `all-MiniLM-L6-v2`)
- **Vector DB**: `FAISS` (fast similarity search)
- **LLM**: OpenAI `gpt-3.5-turbo` or `gpt-4`
- **API**: `FastAPI`

---

## Project Structure

```
college-chatbot-ml/
├── app/
│   ├── pipeline.py     # scraping + pdf + cleaning + chunking
│   ├── embeddings.py   # embedding + FAISS vector DB
│   ├── chatbot.py      # retrieval + OpenAI response generation
│   ├── api.py          # FastAPI endpoint
│   └── utils.py        # helper functions (cleaning, chunking)
├── data/
│   ├── raw/            # temp files
│   ├── processed/      # chunks.jsonl
│   └── vector_store/   # FAISS index + metadata
├── scripts/
│   └── build_index.py  # run full pipeline once
├── requirements.txt
├── .env
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-key-here    # <-- CHANGE THIS
OPENAI_MODEL=gpt-3.5-turbo         # or gpt-4
```

### 3. Build the vector index (run once)

This scrapes all AKGEC pages, extracts PDFs, chunks content, and builds the FAISS index.

```bash
python scripts/build_index.py
```

This takes a few minutes. You only need to run it once (or when the website updates).

### 4. Start the API

```bash
uvicorn app.api:app --reload --port 8000
```

---

## API Usage

### Health check
```
GET http://localhost:8000/health
```

### Ask a question
```
POST http://localhost:8000/chat
Content-Type: application/json

{
    "question": "What is the fee structure for BTech CSE?"
}
```

**Response:**
```json
{
    "answer": "The fee structure for BTech CSE at AKGEC is...",
    "sources": [
        "https://www.akgec.ac.in/admissions/fee-structure/"
    ]
}
```

---

## Where OpenAI API key is used

- **`.env`** — set `OPENAI_API_KEY=sk-...` here
- **`app/chatbot.py`** — the `OpenAI()` client reads it automatically from the environment

---

## Customization

| What | Where |
|------|-------|
| Add more URLs to scrape | `app/pipeline.py` → `URLS` list |
| Change chunk size | `app/utils.py` → `chunk_text(chunk_size=500)` |
| Change number of retrieved chunks | `app/chatbot.py` → `TOP_K = 5` |
| Change the system prompt / persona | `app/chatbot.py` → `system_prompt` |
| Switch to gpt-4 | `.env` → `OPENAI_MODEL=gpt-4` |