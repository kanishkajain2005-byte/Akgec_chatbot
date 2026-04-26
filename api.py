from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import generate_response, _load_resources
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🔥 ADD THIS (MOST IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_model():
    _load_resources()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Chatbot API is running"}

@app.post("/chat")
def chat(request: QueryRequest):
    result = generate_response(request.question)

    # 🔥 align with frontend
    return {
        "reply": result.get("answer")
    }