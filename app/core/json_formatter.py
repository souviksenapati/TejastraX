# Converts extracted answers into structured JSON
def format_to_json(query, answer, source):
    return {
        "query": query,
        "answer": answer,
        "source": source
    }
