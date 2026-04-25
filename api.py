from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import generate_response, _load_resources

app = FastAPI()



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
    return generate_response(request.question)