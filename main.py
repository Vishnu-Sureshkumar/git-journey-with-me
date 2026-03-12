from fastapi import FastAPI
from pydantic import BaseModel
from rag import ingest_documents, ask_question

app = FastAPI()

class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/ingest")
def ingest():
    chunks = ingest_documents()
    return {
        "message": "Documents ingested successfully",
        "chunks_created": chunks
    }


@app.post("/chat")
def chat(query: Query):

    answer = ask_question(query.question)

    return {
        "question": query.question,
        "answer": answer
    }