from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from services.rag_chain import create_rag_chain
from services.add_data import create_embeddings
from services.retriever import PineconeRetrieverWithThreshold
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from fastapi.staticfiles import StaticFiles


templates = Jinja2Templates(directory="templates")

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = PineconeRetrieverWithThreshold()
rag_chain = create_rag_chain(retriever)

class QueryRequest(BaseModel):
    query: str
    chat_history: List[str] = []


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(".", file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Load, split, and embed
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(pages)

        create_embeddings(documents)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    finally:
        # Clean up: delete the PDF after processing
        if os.path.exists(file_path):
            os.remove(file_path)

    return {"status": "success", "message": f"{file.filename} processed and deleted after embedding."}

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    response = rag_chain.invoke({
        "input": request.query,
        "chat_history": request.chat_history
    })
    return {"answer": response["answer"]}


@app.get("/ui/upload", response_class=HTMLResponse)
async def upload_ui(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/ui/ask", response_class=HTMLResponse)
async def ask_ui(request: Request):
    return templates.TemplateResponse("ask.html", {"request": request})
