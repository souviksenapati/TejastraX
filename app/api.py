from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from app.core.rag_engine import run_rag_pipeline, run_rag_pipeline_fast
import asyncio
from functools import lru_cache
import hashlib

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TejastraX LLM-Powered Intelligent Query-Retrieval System",
    description="RAG system for processing policy documents and answering queries"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

from pydantic import HttpUrl

from pydantic import AnyHttpUrl, Field, HttpUrl
from typing import Optional, List
from typing_extensions import Annotated

class ClaimDetails(BaseModel):
    age: int = Field(..., description="Patient age", example=46)
    gender: str = Field(..., description="Patient gender (M/F)", example="M")
    procedure: str = Field(..., description="Medical procedure", example="knee surgery")
    location: str = Field(..., description="Location of treatment", example="Pune")
    policy_duration: str = Field(..., description="Duration since policy inception", example="3 months")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 46,
                "gender": "M",
                "procedure": "knee surgery",
                "location": "Pune",
                "policy_duration": "3 months"
            }
        }
    }

from pydantic import AnyUrl, validator
from urllib.parse import unquote, quote

class PDFUrl(AnyUrl):
    @classmethod
    def validate(cls, value: str) -> str:
        # First, ensure the URL is properly encoded
        parsed = unquote(value)
        # Re-encode it properly
        encoded = quote(parsed, safe=':/?=&')
        return encoded

class QueryRequest(BaseModel):
    documents: str = Field(
        description="URL to the PDF document",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf"
    )
    queries: List[str] = Field(
        description="List of natural language queries about the document",
        example=[
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    )
    
    @validator('documents')
    def validate_pdf_url(cls, v):
        # First decode the URL to handle any encoding
        decoded = unquote(v)
        
        # Extract the base URL without query parameters
        from urllib.parse import urlparse
        parsed = urlparse(decoded)
        path = parsed.path
        
        # Check if the path part ends with .pdf
        if not path.lower().endswith('.pdf'):
            raise ValueError("URL path must point to a PDF file")
            
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
                "queries": [
                    "What is the grace period for premium payment?",
                    "What is the waiting period for pre-existing diseases?"
                ]
            }
        }
    }

class ClauseReference(BaseModel):
    clause_id: str = Field(..., example="Section 2.1.3")
    clause_text: str = Field(..., example="Knee surgery is covered after 3 months waiting period")
    relevance_score: float = Field(..., example=0.85)

class Decision(BaseModel):
    is_approved: bool = Field(..., example=True)
    approval_status: str = Field(
        ..., 
        description="APPROVED, REJECTED, or NEEDS_REVIEW",
        example="APPROVED"
    )
    coverage_amount: Optional[float] = Field(None, example=50000.00)
    waiting_period_cleared: bool = Field(..., example=True)
    location_approved: bool = Field(..., example=True)
    procedure_covered: bool = Field(..., example=True)
    clauses_referenced: List[ClauseReference]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "is_approved": True,
                "approval_status": "APPROVED",
                "coverage_amount": 50000.00,
                "waiting_period_cleared": True,
                "location_approved": True,
                "procedure_covered": True,
                "clauses_referenced": [
                    {
                        "clause_id": "Section 2.1.3",
                        "clause_text": "Knee surgery is covered after 3 months waiting period",
                        "relevance_score": 0.85
                    }
                ]
            }
        }
    }

class ResponseMetadata(BaseModel):
    confidence_score: float = Field(..., example=0.92)
    processing_time: float = Field(..., example=1.23)
    source_sections: List[str] = Field(..., example=["Section 2.1: Coverage for surgical procedures..."])
    reasoning: str = Field(..., example="The claim is approved because...")
    decision_details: Decision

class PerformanceMetrics(BaseModel):
    latency_ms: float = Field(..., example=1234.56)
    tokens_processed: float = Field(..., example=450.0)
    memory_used_mb: float = Field(..., example=128.5)

class QueryResponseItem(BaseModel):
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="The answer to the query")
    confidence_score: float = Field(..., example=0.92)
    source_sections: List[str] = Field(..., description="Relevant sections from the document")

class BatchQueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers for each question")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
                ]
            }
        }
    }

# Valid token for API access
VALID_TOKEN = "c062c70a04bac4e90e7fb7a9b8f62716fa316dfd299996e7ffd1863b7d75ad9c"

async def verify_token(authorization: Optional[str] = Header(None, description="Bearer token")):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Must start with 'Bearer '"
        )
    
    token = authorization.split(" ")[1]
    
    if token != VALID_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    
    return token

@lru_cache(maxsize=100)
def get_cache_key(document: str, question: str) -> str:
    """Generate a cache key for a document-question pair"""
    return hashlib.md5(f"{document}:{question}".encode()).hexdigest()

# Enhanced caching system
document_cache: Dict[str, Tuple[list, list]] = {}
query_embedding_cache: Dict[str, list] = {}  # Cache query embeddings
precomputed_index = None  # Global index cache

import psutil
import time
import os
import requests

async def download_pdf_with_retry(url: str, max_retries: int = 3) -> str:
    """Download PDF from URL with retry logic"""
    import requests
    from urllib.parse import urlparse, unquote
    import backoff
    import PyPDF2
    from io import BytesIO
    
    # Clean and decode the URL
    url = unquote(url)
    print(f"Attempting to download PDF from URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException,),
        max_tries=max_retries
    )
    def fetch_pdf():
        try:
            # Skip HEAD request for speed - go directly to GET
            response = requests.get(url, headers=headers, allow_redirects=True, verify=False, timeout=15, stream=True)
            print(f"GET response status: {response.status_code}")
            
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail="PDF document not found at the specified URL"
                )
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch PDF: Server returned HTTP {response.status_code}"
                )
            
            # Read the content
            content = response.content
            
            # Try to validate PDF using PyPDF2
            try:
                pdf_file = BytesIO(content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                print(f"Successfully validated PDF with {num_pages} pages")
            except Exception as e:
                print(f"PDF validation error: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid PDF format: {str(e)}"
                )
            
            # Save to temp directory
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Use URL hash for unique filename
            file_hash = hashlib.md5(str(url).encode()).hexdigest()[:12]
            temp_path = os.path.join(temp_dir, f"doc_{file_hash}.pdf")
            
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            print(f"PDF successfully saved to {temp_path}")
            return temp_path
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download PDF: {str(e)}"
            )
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )
    
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_pdf)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF document: {str(e)}"
        )
            
@app.post("/hackrx/run", response_model=BatchQueryResponse)
async def hackrx_run(
    request: dict,
    token: str = Depends(verify_token)
):
    temp_file = None
    try:
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        documents = request.get('documents')
        if not documents:
            raise HTTPException(status_code=422, detail="documents field is required")
            
        # Handle questions field
        questions = request.get('questions', [])
        if not questions:
            raise HTTPException(status_code=422, detail="questions field is required")
        
        print(f"Processing request for document: {documents}")
        
        # Check if document is already processed
        doc_cache_key = hashlib.md5(str(documents).encode()).hexdigest()
        
        # Hackathon optimization: Check document cache first
        if doc_cache_key in document_cache:
            print("Using cached document processing")
        else:
            # Download PDF with reduced retries for speed
            temp_file = await download_pdf_with_retry(str(documents), max_retries=1)
            if not temp_file:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to process the PDF document"
                )
            
            # Load and index document once for all questions
            from app.services.document_loader import load_pdf
            from app.services.embedding_search import build_index
            
            docs = load_pdf(temp_file)
            build_index(docs)
            
            # Cache document processing
            document_cache[doc_cache_key] = ("processed", {})
        
        answers = []
        # Parallel processing of questions for speed
        import asyncio
        import concurrent.futures
        
        async def process_question(question):
            try:
                cache_key = get_cache_key(str(documents), question)
                
                if cache_key in document_cache:
                    print(f"Cache hit for question: {question}")
                    result, metadata = document_cache[cache_key]
                else:
                    # Run optimized RAG pipeline
                    loop = asyncio.get_event_loop()
                    result, metadata = await loop.run_in_executor(
                        None, 
                        run_rag_pipeline_fast,
                        question
                    )
                    # Cache the result
                    document_cache[cache_key] = (result, metadata)
                
                return result.get("answer", result.get("decision_summary", ""))
                
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                return f"Error processing question: {str(e)}"
        
        # Maximum parallel processing for speed
        semaphore = asyncio.Semaphore(4)  # Increased to 4 for speed
        
        async def process_with_semaphore(question):
            async with semaphore:
                return await process_question(question)
        
        # Execute all questions concurrently without timeout
        tasks = [process_with_semaphore(question) for question in questions]
        answers = await asyncio.gather(*tasks)
        
        print(f"Request processed in {(time.time() - start_time) * 1000:.2f}ms")
        
        return BatchQueryResponse(answers=answers)
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary files
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_file}: {e}")
