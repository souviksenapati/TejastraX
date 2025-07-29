import os
import time
from typing import Tuple, List
import google.generativeai as genai

# Load API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load models
from app.config import GEMINI_MODEL
chat_model = genai.GenerativeModel(GEMINI_MODEL)

def generate_answer_with_reasoning(query: str, context: str = "") -> Tuple[str, str]:
    """
    Use Gemini model to generate a response with reasoning from a query and context.
    Returns the answer and the reasoning behind it.
    """
    try:
        prompt = f"""Context: {context}

Question: {query}

Instructions:
1. First, analyze the relevant sections and identify key information
2. Explain your logical reasoning for arriving at the answer
3. Provide evidence from the text to support your reasoning
4. Keep explanations clear and concise
5. If exact information is not found, explain what is missing
6. Include specific values, dates, and numbers if available

Format your response as:
REASONING: 
I analyzed the policy text and found that [key observation]. 
Based on [specific evidence], I concluded that [logical connection].
[Additional context or caveats if needed]

ANSWER: [Clear, concise answer backed by the reasoning]"""
        
        response = chat_model.generate_content(prompt)
        text = response.text
        
        # Split response into reasoning and answer
        parts = text.split("ANSWER:")
        if len(parts) == 2:
            reasoning = parts[0].replace("REASONING:", "").strip()
            answer = parts[1].strip()
        else:
            # Fallback if format is not as expected
            answer = text
            reasoning = "Direct answer generated from context"
            
        return answer, reasoning
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return "Error generating response", "Error in LLM processing"

def generate_answer(query: str, context: str = "") -> str:
    """
    Legacy method that returns just the answer without reasoning.
    """
    try:
        answer, _ = generate_answer_with_reasoning(query, context)
        return answer
    except Exception as e:
        return f"[Gemini generation error] {e}"

def get_embedding(text: str) -> list:
    """Get embedding for a single text using Gemini embedding model."""
    try:
        # Truncate text if too long to avoid API limits
        if len(text) > 2000:
            text = text[:2000]
            
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']
    except Exception as e:
        print(f"[Gemini embedding error] {e}")
        return []

def get_embeddings_batch(texts: list) -> list:
    """Ultra-fast embedding generation for <10s target"""
    import concurrent.futures
    
    embeddings = []
    
    def get_embedding_ultra_fast(text):
        try:
            # Aggressive truncation for maximum speed
            if len(text) > 1000:
                text = text[:1000]
            return get_embedding(text)
        except:
            return None
    
    # Maximum parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_embedding_ultra_fast, text) for text in texts]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                emb = future.result()
                if emb:
                    embeddings.append(emb)
            except:
                continue
    
    return embeddings
