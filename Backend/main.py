from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import pdfplumber 
from vector_store import (
    add_chunks,
    search_chunks,
    index,
    chunks_store
)
from utils import chunk_text
from llm import generate_answer

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

#creates folder if not found
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_upload_path(filename: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return file_path


def extract_pdf_text(file_path: str):
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


#home
@app.get("/")
def home():
    return {
        "message": "Smuddy AI Backend Running"
    }


#upload
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    safe_filename = os.path.basename(file.filename)

    file_path = os.path.join(
        UPLOAD_FOLDER,
        safe_filename
    )

    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {
        "filename": safe_filename,
        "status": "uploaded successfully"
    }

#Extract
@app.get("/extract-text/{filename}")
def extract_text(filename:str):
    file_path = get_upload_path(filename)
    text = extract_pdf_text(file_path)

    return{
        "filename":filename,
        "text":text
    }

#chunking
@app.get("/chunks/{filename}")
def get_chunks(filename:str):
    file_path = get_upload_path(filename)
    text = extract_pdf_text(file_path)
    chunks = chunk_text(text)

    try:
        add_chunks(chunks)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding model is not ready: {error}"
        ) from error

    return{
        "total_chunks":len(chunks),
        "chunks":chunks[:5]
    }


@app.get("/ask")
def ask(question: str):

    results = search_chunks(question)

    if not results:
        return {
            "question": question,
            "answer": "No document chunks found. Upload and chunk a PDF first."
        }

    context = "\n".join(results)

    answer = generate_answer(
        question,
        context
    )

    return {
        "question": question,
        "answer": answer
    }

@app.get("/debug")
def debug():
    return {
        "vectors": index.ntotal,
        "chunks": len(chunks_store)
    }


