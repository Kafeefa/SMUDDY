from fastapi import FastAPI, UploadFile, File
import os
import pdfplumber 
from vector_store import (
    add_chunks,
    search_chunks,
    index,model,chunks_store
)
from utils import chunk_text

app = FastAPI()
UPLOAD_FOLDER = "uploads"

#creates folder if not found
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


#home
@app.get("/")
def home():
    return {
        "message": "Smuddy AI Backend Running"
    }


#upload
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {
        "filename": file.filename,
        "status": "uploaded successfully"
    }

#Extract
@app.get("/extract-text/{filename}")
def extract_text(filename:str):
    file_path=os.path.join(
        UPLOAD_FOLDER,filename
    )

   
    text=" "
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text+=page.extract_text()+"\n"

    return{
        "filename":filename,
        "text":text
    }

#chunking
@app.get("/chunks/{filename}")
def get_chunks(filename:str):
    file_path=os.path.join(
        UPLOAD_FOLDER,filename
    )

    reader= pdfplumber.open(file_path)
    text=" "
    with pdfplumber.open(file_path)as reader:
        for page in reader.pages:
            page_text=page.extract_text()
        if page_text:
            text+=page_text+"\n"
    chunks=chunk_text(text)
    add_chunks(chunks)

    return{
        "total_chunks":len(chunks),
        "chunks":chunks[:5]
    }
@app.get("/ask")
def ask(question: str):

    results = search_chunks(question)

    return {
        "question": question,
        "relevant_chunks": results
    }

@app.get("/debug")
def debug():
    return {
        "vectors": index.ntotal,
        "chunks": len(chunks_store)
    }


