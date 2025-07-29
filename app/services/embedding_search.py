import faiss
import numpy as np
from app.config import EMBED_MODEL, EMBED_DIM
from app.services.llm_client import get_embedding, get_embeddings_batch

index = faiss.IndexFlatL2(EMBED_DIM)
corpus = []

def reset_index():
    global index, corpus
    index = faiss.IndexFlatL2(EMBED_DIM)
    corpus = []

def build_index(docs):
    global corpus
    reset_index()
    
    if not docs:
        return
        
    # Use all available chunks from advanced chunking
    texts = [doc["text"] for doc in docs]
    embeddings = get_embeddings_batch(texts)
    
    valid_docs = []
    valid_embeddings = []
    
    for doc, emb in zip(docs, embeddings):
        if emb:
            valid_embeddings.append(emb)
            valid_docs.append(doc)
    
    corpus = valid_docs
    if valid_embeddings:
        embeddings_array = np.array(valid_embeddings).astype("float32")
        index.add(embeddings_array)
        
        # Enhanced logging
        content_types = {}
        for doc in valid_docs:
            content_type = doc.get('type', 'general')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        print(f"Built index with {len(valid_docs)} chunks:")
        for content_type, count in content_types.items():
            print(f"  - {content_type}: {count}")
        
        avg_importance = sum(doc.get('importance', 0.5) for doc in valid_docs) / len(valid_docs)
        print(f"  - Average importance score: {avg_importance:.3f}")

def search_similar_chunks(query, top_k=6):  # Increased to 6 for better coverage
    if len(corpus) == 0:
        return []
    
    # Enhanced search with query expansion
    from app.api import query_embedding_cache
    
    # Check embedding cache first
    if query in query_embedding_cache:
        query_emb = np.array([query_embedding_cache[query]]).astype("float32")
    else:
        query_emb = np.array([get_embedding(query)]).astype("float32")
        query_embedding_cache[query] = query_emb[0].tolist()  # Cache for future use
    
    # Search with even more candidates to ensure coverage
    distances, indices = index.search(query_emb, min(top_k * 4, len(corpus)))
    
    results = []
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    for i, distance in zip(indices[0], distances[0]):
        if distance < 2.5:  # More relaxed threshold
            similarity = 1 - min(distance, 2.5) / 2.5
            doc = corpus[i].copy()
            doc["score"] = similarity
            
            # Dynamic query-text relevance boost
            text_lower = doc['text'].lower()
            text_words = set(text_lower.split())
            
            # Calculate word overlap ratio
            word_overlap = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
            doc["score"] += word_overlap * 0.2
            
            # Boost for phrase matches
            query_phrases = [' '.join(query_lower.split()[i:i+2]) for i in range(len(query_lower.split())-1)]
            phrase_matches = sum(1 for phrase in query_phrases if phrase in text_lower)
            doc["score"] += phrase_matches * 0.1
            
            results.append(doc)
    
    # Enhanced sorting with importance and query relevance
    results.sort(key=lambda x: (x['score'] + x.get('importance', 0.5)) / 2, reverse=True)
    
    return results[:top_k]
