import time
import re
from typing import List, Tuple, Dict, Optional
from pydantic import BaseModel
from app.services.embedding_search import search_similar_chunks, build_index
from app.services.llm_client import generate_answer, generate_answer_with_reasoning
from app.services.document_loader import load_pdf, download_pdf
from app.core.logic_engine import refine_answer

class ClaimAnalysis(BaseModel):
    query_understood: str
    decision_summary: str
    metadata: Dict
    clauses: List[Dict]

def parse_claim_details(query: str) -> Dict:
    """Extract structured information from natural language query"""
    patterns = {
        'age': r'(\d+)(?:[-\s]?(?:year|yr|y))?[,\s]*(?:old|age)?[,\s]*(?:M|F|male|female)',
        'gender': r'(?:^|\s)(M|F|male|female)',
        'procedure': r'(?:surgery|procedure|operation|treatment)[\s:]+([^,\.]+)',
        'location': r'(?:in|at|from)\s+([A-Za-z\s]+?)(?:,|\.|$)',
        'policy_duration': r'(\d+)[\s-](?:month|day|year|yr)[\s-](?:old|policy)',
    }
    
    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            extracted[key] = match.group(1).strip()
        else:
            extracted[key] = 'N/A'  # Provide default value instead of missing key
    
    return extracted

def analyze_coverage(claim_details: Dict, context: str) -> Dict:
    """Analyze if the claim is covered based on policy rules"""
    # Search for relevant clauses about:
    # 1. Procedure coverage
    # 2. Waiting period
    # 3. Location/network restrictions
    # 4. Age restrictions
    
    clauses = []
    is_approved = True
    reasons = []
    
    # Extract relevant clauses for each criterion
    procedure_clauses = search_similar_chunks(f"coverage for {claim_details['procedure']}", top_k=2)
    waiting_clauses = search_similar_chunks(f"waiting period for {claim_details['procedure']}", top_k=2)
    location_clauses = search_similar_chunks(f"hospital network {claim_details['location']}", top_k=2)
    
    # Analyze each aspect
    procedure_covered = any('covered' in doc['text'].lower() for doc in procedure_clauses)
    waiting_cleared = all('waiting period' not in doc['text'].lower() for doc in waiting_clauses)
    location_ok = any('network hospital' in doc['text'].lower() for doc in location_clauses)
    
    return {
        'is_approved': procedure_covered and waiting_cleared and location_ok,
        'approval_status': "APPROVED" if (procedure_covered and waiting_cleared and location_ok) else "REJECTED",
        'waiting_period_cleared': waiting_cleared,
        'location_approved': location_ok,
        'procedure_covered': procedure_covered,
        'clauses_referenced': procedure_clauses + waiting_clauses + location_clauses
    }

def is_claim_query(query: str) -> bool:
    """Determine if query is about a specific claim or general policy question"""
    # Look for specific patient details that indicate a claim
    claim_patterns = [
        r'\d+\s*(?:year|yr)\s*old',  # age pattern
        r'\b[MF]\b.*(?:surgery|procedure|operation|treatment)',  # gender + medical procedure
        r'(?:surgery|procedure|operation|treatment).*\b[MF]\b',  # medical procedure + gender
        r'patient.*(?:surgery|procedure|operation|treatment)',  # patient context
        r'claim.*(?:approved|rejected|coverage)'  # explicit claim language
    ]
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in claim_patterns)

def run_rag_pipeline(query: str, file_path: str) -> Tuple[Dict, Dict]:
    start_time = time.time()
    
    # Load and index the PDF
    docs = load_pdf(file_path)
    build_index(docs)
    
    # Get relevant policy sections
    relevant_docs = search_similar_chunks(query)
    context = "\n".join([doc['text'] for doc in relevant_docs])
    
    # Check if this is a claim query or general policy question
    if is_claim_query(query):
        # Handle as claim analysis
        claim_details = parse_claim_details(query)
        query_understood = f"Patient: {claim_details['age']} year old {claim_details['gender']}, "\
                          f"Procedure: {claim_details['procedure']}, "\
                          f"Location: {claim_details['location']}, "\
                          f"Policy Age: {claim_details['policy_duration']}"
        
        coverage_decision = analyze_coverage(claim_details, context)
        
        prompt = f"Claim Details:\n{query_understood}\n\nRelevant Policy Sections:\n{context}\n\n"\
                f"Based on the policy sections, explain if this claim should be approved or rejected."
        
        answer, reasoning = generate_answer_with_reasoning(prompt, context)
        decision_summary = "Claim APPROVED" if coverage_decision['is_approved'] else "Claim REJECTED"
        
        result = {
            "query_understood": query_understood,
            "decision_summary": decision_summary,
            "answer": answer
        }
    else:
        # Handle as general policy question with optimized prompt
        prompt = f"""Based on the policy document sections below, answer this question directly and concisely:

Question: {query}

Policy Text:
{context}

Provide a clear, factual answer with specific details (numbers, time periods, conditions) from the policy. If not found, state "Information not available in the provided text."

Answer:"""
        
        answer = generate_answer(prompt, context)  # Use faster method without reasoning
        
        result = {
            "answer": answer,
            "decision_summary": answer
        }
    
    # Calculate confidence
    confidence = sum(doc['score'] for doc in relevant_docs) / len(relevant_docs) if relevant_docs else 0.0
    
    # Format response
    source_sections = [f"Section {i+1}: {doc['text'][:100]}..." for i, doc in enumerate(relevant_docs)]
    
    # Set reasoning based on query type
    reasoning = "Direct answer from policy text" if not is_claim_query(query) else "Claim analysis completed"
    
    metadata = {
        "confidence_score": confidence,
        "processing_time": time.time() - start_time,
        "source_sections": source_sections,
        "reasoning": reasoning
    }
    
    return result, metadata

def run_rag_pipeline_fast(query: str) -> Tuple[Dict, Dict]:
    """Enhanced RAG pipeline with multi-strategy search and text fallback"""
    start_time = time.time()
    
    # Step 1: Multi-strategy semantic search
    relevant_docs = search_similar_chunks(query)
    
    # Simplified context building for speed
    context_parts = []
    
    if not relevant_docs:
        result = {
            "answer": "Information not available in the provided text",
            "decision_summary": "Information not available in the provided text"
        }
        
        metadata = {
            "confidence_score": 0.0,
            "processing_time": time.time() - start_time,
            "source_sections": []
        }
        
        return result, metadata
    
    # Enhanced context with full chunks for better accuracy
    context = "\n\n".join([doc['text'] for doc in relevant_docs])  # Use full chunks without truncation
    
    # Enhanced prompt for better accuracy
    prompt = f"""Context:
{context}

Question: {query}

Based on the context above, provide a direct and specific answer. Include exact time periods (like "2 years", "36 months"), percentages, and conditions when available. Look carefully for waiting periods, coverage details, and specific requirements.

Answer:"""
    
    answer = generate_answer(prompt, context)
    
    # Simple confidence calculation
    confidence = sum(doc['score'] for doc in relevant_docs) / len(relevant_docs) if relevant_docs else 0.0
    
    result = {
        "answer": answer,
        "decision_summary": answer
    }
    
    metadata = {
        "confidence_score": min(confidence, 1.0),
        "processing_time": time.time() - start_time,
        "source_sections": [f"Page {doc.get('page', 'N/A')}: {doc['text'][:100]}..." for doc in relevant_docs[:3]]
    }
    
    return result, metadata

def run_rag_pipeline_json(document_url: str, questions: List[str]) -> List[str]:
    # Download PDF from URL
    file_path = download_pdf(document_url)
    
    try:
        # Load and index the PDF once
        docs = load_pdf(file_path)
        build_index(docs)
        
        answers = []
        for question in questions:
            # Search for relevant chunks
            relevant_docs = search_similar_chunks(question)
            context = "\n".join([doc['text'] for doc in relevant_docs])
            
            # Generate answer
            raw_answer = generate_answer(question, context)
            final_answer = refine_answer(raw_answer)
            answers.append(final_answer)
            
        return answers
    finally:
        # Clean up downloaded file
        import os
        if os.path.exists(file_path):
            os.remove(file_path)
