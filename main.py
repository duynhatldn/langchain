import os
import base64
from document_handler import DocumentHandler
from pydantic import BaseModel
from fastapi import FastAPI, Cookie, Response
from dotenv import load_dotenv
from transformers import GPT2Tokenizer

app = FastAPI()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
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
def get_answers(query_data: QueryModel, response: Response, user_context: str = Cookie(None)):
    if user_context:
        context = decode_from_base64(user_context)
    else:
        context = ""
    updated_context = update_context(context, query_data.query, tokenizer)
    answer = document_handler.retrieve_answers(updated_context)
    encoded_context = encode_to_base64(updated_context)
    response.set_cookie(key="user_context", value=encoded_context)
    return answer

def update_context(current_context, new_query, tokenizer, max_length=4097):
    # combined_context = current_context + " " + new_query
    combined_context = new_query
    context_tokens = tokenizer.encode(combined_context)
    if len(context_tokens) > max_length:
        start_index = len(context_tokens) - max_length
        context_tokens = context_tokens[start_index:]
    updated_context = tokenizer.decode(context_tokens)
    return updated_context

def encode_to_base64(string):
    string_bytes = string.encode('utf-8')
    base64_bytes = base64.b64encode(string_bytes)
    base64_string = base64_bytes.decode('ascii')
    return base64_string

def decode_from_base64(base64_string):
    base64_bytes = base64_string.encode('ascii')
    string_bytes = base64.b64decode(base64_bytes)
    string = string_bytes.decode('utf-8')
    return string
