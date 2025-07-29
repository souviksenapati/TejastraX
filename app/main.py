from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import app as api_app

app = FastAPI(
    title="TejastraX LLM-Powered Intelligent Query-Retrieval System",
    description="RAG system for processing policy documents and answering queries",
    version="1.0.0"
)

# Allow CORS for all origins (you can restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main API routes
app.mount("/api", api_app)

@app.get("/")
def read_root():
    return {"message": "Welcome to TejastraX API. Visit /api/docs for Swagger UI."}
