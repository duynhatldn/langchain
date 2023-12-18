import os
from document_handler import DocumentHandler
from pydantic import BaseModel
from fastapi import FastAPI
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()
document_handler = DocumentHandler(
    'documents/',
    os.environ['OPENAI_API_KEY'],
    "36261030-cd1f-4454-a508-d7335203f175",
    'langchainvector'
)

class QueryModel(BaseModel):
    query: str

@app.post("/retrieve_answers")
def get_answers(query_data: QueryModel):
    return document_handler.retrieve_answers(query_data.query)
