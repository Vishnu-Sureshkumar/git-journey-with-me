import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

db = lancedb.connect("./embeddings")

model = get_registry().get("sentence-transformers").create()

class DataStore(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

# create table if not exists
table = db.create_table("rag_test", schema=DataStore, mode="create")

def ingest_documents():

    documents = SimpleDirectoryReader("./data_source").load_data()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100,
        chunk_overlap=10
    )

    total_chunks = 0

    for doc in documents:
        if doc.text:
            chunks = text_splitter.split_text(doc.text)

            data_list = [{"text": chunk} for chunk in chunks]

            table.add(data_list)

            total_chunks += len(chunks)

    return total_chunks


def ask_question(query):

    relevant_context = table.search(query).limit(2).to_list()

    context_joined = ",".join(c["text"] for c in relevant_context)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are an AI assistant. Only answer from the context."
        ),
        contents=f"""
        User: {query}
        Context: {context_joined}
        Answer:
        """
    )

    return response.text