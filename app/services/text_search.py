import re
from typing import List, Dict, Optional

def find_text_snippets(full_text: str, search_terms: List[str], context_size: int = 300) -> List[str]:
    """Find text snippets containing search terms with surrounding context"""
    snippets = []
    text_lower = full_text.lower()
    
    for term in search_terms:
        term_lower = term.lower()
        
        # Find all occurrences of the term
        for match in re.finditer(re.escape(term_lower), text_lower):
            start = max(0, match.start() - context_size)
            end = min(len(full_text), match.end() + context_size)
            
            snippet = full_text[start:end].strip()
            if snippet and snippet not in snippets:
                snippets.append(snippet)
    
    return snippets

def search_specific_terms(query: str, full_text: str) -> Optional[str]:
    """Search for specific terms that might be missed by embeddings"""
    
    query_lower = query.lower()
    
    # Define search strategies for different types of questions
    search_strategies = {
        'health_checkup': {
            'terms': ['health check', 'preventive', 'check-up', 'two continuous', 'block of two'],
            'keywords': ['health', 'check', 'preventive']
        },
        'room_rent': {
            'terms': ['room rent', '1%', 'one percent', 'ICU', '2%', 'two percent', 'plan A', 'sub-limit'],
            'keywords': ['room', 'rent', 'ICU', 'plan A', 'limit']
        },
        'grace_period': {
            'terms': ['grace period', '30 days', 'thirty days', 'premium payment'],
            'keywords': ['grace', 'period', 'premium']
        },
        'waiting_period': {
            'terms': ['waiting period', '36 months', 'thirty-six months', 'pre-existing'],
            'keywords': ['waiting', 'period', 'months']
        }
    }
    
    # Determine which strategy to use based on query
    strategy = None
    if 'health check' in query_lower or 'preventive' in query_lower:
        strategy = search_strategies['health_checkup']
    elif 'room rent' in query_lower or 'ICU' in query_lower or 'plan A' in query_lower:
        strategy = search_strategies['room_rent']
    elif 'grace period' in query_lower:
        strategy = search_strategies['grace_period']
    elif 'waiting period' in query_lower:
        strategy = search_strategies['waiting_period']
    
    if strategy:
        # Find relevant snippets
        snippets = find_text_snippets(full_text, strategy['terms'])
        
        if snippets:
            # Return the most relevant snippet (first one found)
            return snippets[0]
    
    return None

def extract_numerical_info(text: str, query: str) -> Optional[str]:
    """Extract numerical information from text based on query context"""
    
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Patterns for different types of numerical information
    patterns = {
        'percentage': r'(\d+(?:\.\d+)?)\s*%',
        'days': r'(\d+)\s*days?',
        'months': r'(\d+)\s*months?',
        'years': r'(\d+)\s*years?',
        'beds': r'(\d+)\s*(?:inpatient|in-patient)?\s*beds?'
    }
    
    results = []
    
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, text_lower)
        if matches:
            for match in matches:
                # Get context around the number
                for num_match in re.finditer(pattern, text_lower):
                    start = max(0, num_match.start() - 100)
                    end = min(len(text), num_match.end() + 100)
                    context = text[start:end].strip()
                    results.append(f"{match} ({pattern_name}): {context}")
    
    return "; ".join(results) if results else None