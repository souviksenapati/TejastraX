# TejastraX API

A Gemini-powered Retrieval Augmented Generation (RAG) pipeline for parsing policy documents and answering natural language queries.

## Features
- Google Gemini for LLM and embeddings
- FAISS for fast vector search
- Advanced chunking with recursive splitting
- Query-aware reranking
- FastAPI REST endpoint

## API Endpoint

### POST /api/hackrx/run

**Request:**
```json
{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment.",
        "There is a waiting period of thirty-six months for pre-existing diseases."
    ]
}
```

## Environment Variables

Set the following environment variable:
- `GEMINI_API_KEY`: Your Google Gemini API key

## Deployment

### Hugging Face Spaces
1. Create a new Space with Python SDK
2. Upload all files
3. Set `GEMINI_API_KEY` in Space settings
4. The API will be available at `https://your-space.hf.space/api/hackrx/run`

### Vercel
1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in project directory
3. Set `GEMINI_API_KEY` environment variable
4. Deploy with `vercel --prod`

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

API will be available at `http://localhost:7860/api/hackrx/run`