def test_rag_output():
    from app.core.rag_engine import run_rag_pipeline
    ans, docs = run_rag_pipeline("What is the coverage?", index=None)
    assert isinstance(ans, str)
