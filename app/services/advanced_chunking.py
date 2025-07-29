import re
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

def recursive_text_splitter(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> List[str]:
    """Recursively split text using multiple separators"""
    
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    
    if len(text) <= max_chars:
        return [text]
    
    # Hierarchical separators (order matters)
    separators = [
        '\n\n\n',  # Multiple newlines
        '\n\n',    # Double newlines (paragraphs)
        '\n',      # Single newlines
        '. ',      # Sentences
        ', ',      # Clauses
        ' '        # Words
    ]
    
    chunks = []
    
    def split_by_separator(text: str, separator: str) -> List[str]:
        if separator not in text:
            return [text]
        
        parts = text.split(separator)
        result = []
        current_chunk = ""
        
        for i, part in enumerate(parts):
            test_chunk = current_chunk + separator + part if current_chunk else part
            
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    result.append(current_chunk)
                    # Add overlap from previous chunk
                    overlap_text = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else current_chunk
                    current_chunk = overlap_text + separator + part if overlap_text else part
                else:
                    current_chunk = part
        
        if current_chunk:
            result.append(current_chunk)
        
        return result
    
    # Try each separator recursively
    def recursive_split(text: str, sep_index: int = 0) -> List[str]:
        if sep_index >= len(separators) or len(text) <= max_chars:
            return [text]
        
        separator = separators[sep_index]
        parts = split_by_separator(text, separator)
        
        result = []
        for part in parts:
            if len(part) > max_chars:
                # Recursively split with next separator
                result.extend(recursive_split(part, sep_index + 1))
            else:
                result.append(part)
        
        return result
    
    return recursive_split(text)

def extract_metadata(text: str, page_num: int, chunk_index: int) -> Dict:
    """Extract metadata from text chunk"""
    
    # Count different types of content
    numbers = len(re.findall(r'\d+', text))
    percentages = len(re.findall(r'\d+(?:\.\d+)?%', text))
    dates = len(re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text))
    
    # Identify content type
    content_type = "general"
    if any(term in text.lower() for term in ['definition', 'means', 'defined as']):
        content_type = "definition"
    elif any(term in text.lower() for term in ['waiting period', 'grace period', 'months', 'days']):
        content_type = "time_period"
    elif any(term in text.lower() for term in ['coverage', 'covered', 'benefit', 'limit']):
        content_type = "coverage"
    elif any(term in text.lower() for term in ['exclusion', 'not covered', 'excluded']):
        content_type = "exclusion"
    
    # Calculate importance score
    importance_score = 0.5  # Base score
    
    # Boost for definitions and time periods
    if content_type in ["definition", "time_period"]:
        importance_score += 0.3
    
    # Boost for numerical content
    if numbers > 0:
        importance_score += 0.1
    if percentages > 0:
        importance_score += 0.2
    
    # Boost for key terms
    key_terms = ['hospital', 'coverage', 'waiting', 'grace', 'benefit', 'limit', 'plan', 'policy']
    term_count = sum(1 for term in key_terms if term in text.lower())
    importance_score += min(term_count * 0.05, 0.2)
    
    return {
        'page': page_num,
        'chunk_index': chunk_index,
        'token_count': estimate_tokens(text),
        'char_count': len(text),
        'numbers_count': numbers,
        'percentages_count': percentages,
        'dates_count': dates,
        'content_type': content_type,
        'importance_score': min(importance_score, 1.0),
        'key_terms_count': term_count
    }

def advanced_pdf_chunking(file_path: str) -> List[Dict]:
    """Ultra-fast chunking for <10s target"""
    
    reader = PdfReader(file_path)
    all_chunks = []
    chunk_id = 0
    
    # Process ALL pages with minimal processing
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        
        # Minimal text cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Balanced chunking: 450 tokens with 80 token overlap for better context
        chunks = recursive_text_splitter(text, max_tokens=450, overlap_tokens=80)
        
        for chunk_index, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 50:
                continue
            
            # Balanced importance scoring
            importance_score = 0.5
            
            # General importance scoring based on content patterns
            text_lower = chunk_text.lower()
            
            # Boost for policy-relevant terms
            policy_terms = ['hospital', 'coverage', 'waiting', 'grace', 'benefit', 'limit', 'policy', 'insured']
            policy_score = sum(1 for term in policy_terms if term in text_lower)
            importance_score += min(policy_score * 0.05, 0.3)
            
            # Boost for numerical content (time periods, percentages, amounts)
            if re.search(r'\d+(?:%|\s*(?:days?|months?|years?|INR|rupees?))', chunk_text):
                importance_score += 0.25
            
            # Boost for definition-like content
            if any(pattern in text_lower for pattern in ['means', 'defined as', 'refers to', 'includes']):
                importance_score += 0.2
            
            # Boost for procedural/medical content
            medical_indicators = len(re.findall(r'\b(?:treatment|surgery|procedure|therapy|medical|clinical)\b', text_lower))
            importance_score += min(medical_indicators * 0.03, 0.15)
            
            all_chunks.append({
                'id': chunk_id,
                'text': chunk_text.strip(),
                'metadata': {
                    'page': page_num + 1,
                    'importance_score': min(importance_score, 1.0),
                    'content_type': 'general'
                }
            })
            
            chunk_id += 1
    
    # Return top 50 chunks for better coverage
    all_chunks.sort(key=lambda x: x['metadata']['importance_score'], reverse=True)
    return all_chunks[:50]

def rerank_chunks(chunks: List[Dict], query: str) -> List[Dict]:
    """Fast reranking with maintained accuracy"""
    
    query_lower = query.lower()
    query_words = query_lower.split()
    
    for chunk in chunks:
        text_lower = chunk['text'].lower()
        
        # Fast term overlap calculation
        overlap_count = sum(1 for word in query_words if word in text_lower)
        term_overlap = overlap_count / len(query_words) if query_words else 0
        
        # Quick phrase matching
        phrase_boost = 0.1 if any(f"{query_words[i]} {query_words[i+1]}" in text_lower 
                                 for i in range(len(query_words)-1)) else 0
        
        # Fast content type boost
        content_boost = 0
        if 'definition' in query_lower and chunk['metadata']['content_type'] == 'definition':
            content_boost = 0.2
        elif any(term in query_lower for term in ['period', 'days', 'months', 'waiting', 'grace']):
            content_boost = 0.2
        
        # Simplified scoring
        rerank_score = (
            chunk['metadata']['importance_score'] * 0.5 +
            term_overlap * 0.3 +
            phrase_boost + content_boost
        )
        
        chunk['rerank_score'] = rerank_score
    
    # Sort by rerank score
    chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    return chunks