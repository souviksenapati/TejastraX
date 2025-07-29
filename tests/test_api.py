from fastapi.testclient import TestClient
from app.utils.main import app

client = TestClient(app)

def test_query():
    response = client.post("/query", json={"query": "What is the claim process?"})
    assert response.status_code == 200
    assert "answer" in response.json()
