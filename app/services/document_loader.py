import os
import requests
import tempfile
import hashlib
from PyPDF2 import PdfReader

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, filename))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            documents.append({"filename": filename, "text": text})
    return documents

def download_pdf(url: str) -> str:
    """Download PDF from URL and return local file path"""
    try:
        response = requests.get(url, timeout=30, verify=True)  # Add SSL verification
        response.raise_for_status()
        
        # Verify it's a PDF by content type or file extension
        content_type = response.headers.get('content-type', '').lower()
        is_pdf = ('application/pdf' in content_type or 
                 url.lower().endswith('.pdf') or
                 response.content.startswith(b'%PDF-'))
        
        if not is_pdf:
            raise ValueError(f"URL does not point to a PDF file. Content-Type: {content_type}")
        
        # Create temp file with a unique name
        temp_dir = tempfile.gettempdir()
        temp_name = f"policy_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
        temp_path = os.path.join(temp_dir, temp_name)
        
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(response.content)
            
        return temp_path
        
    except requests.RequestException as e:
        raise ValueError(f"Failed to download PDF: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")

# Global variable to store full document text for keyword search
full_document_text = ""

def load_pdf(file_path):
    global full_document_text
    
    # Extract full text for text search fallback
    reader = PdfReader(file_path)
    full_text_pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            full_text_pages.append(text)
    
    full_document_text = "\n".join(full_text_pages)
    
    # Use advanced chunking strategy
    from app.services.advanced_chunking import advanced_pdf_chunking
    
    advanced_chunks = advanced_pdf_chunking(file_path)
    
    # Convert to expected format
    chunks = []
    for chunk_data in advanced_chunks:
        chunks.append({
            "text": chunk_data['text'],
            "page": chunk_data['metadata']['page'],
            "type": chunk_data['metadata']['content_type'],
            "importance": chunk_data['metadata']['importance_score'],
            "metadata": chunk_data['metadata']
        })
    
    return chunks

def get_full_document_text():
    """Get the full document text for keyword search"""
    return full_document_text
