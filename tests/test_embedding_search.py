def test_embeddings_shape():
    from app.services.llm_client import get_embedding
    emb = get_embedding("Hello world")
    assert len(emb) == 768
