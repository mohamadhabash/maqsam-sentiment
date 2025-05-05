import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create a TestClient instance, which lets us simulate HTTP requests to the FastAPI app
client = TestClient(app)

# We define multiple test cases here. Each tuple is (input_summary, expected_sentiment).
@pytest.mark.parametrize("summary,expected", [
    ("I love this product!", "Positive"),
    ("The issue is still unresolved and frustrating", "Negative"),
])
def test_predict(summary, expected):
    """
    This test function will be run once for each (summary, expected) pair defined above.
    
    Steps:
    1. Send a POST request to /api/predict with the JSON body {"summary": summary}.
    2. Check that the HTTP status code is 200 (meaning "OK").
    3. Parse the response JSON into a Python dict.
    4. Confirm that there is a 'sentiment' key in the response.
    5. Assert that the returned sentiment exactly matches the expected label.
    """

    # 1. Send request
    response = client.post("/api/predict", json={"summary": summary})

    # 2. Verify status code is 200 (OK)
    assert response.status_code == 200, f"Expected status 200 but got {response.status_code}"

    # 3. Parse JSON response
    data = response.json()

    # 4. Ensure 'sentiment' field is present
    assert "sentiment" in data, "Response JSON is missing 'sentiment' key"

    # 5. Check that the model's output matches the expected sentiment
    assert data["sentiment"] == expected, (
        f"For input '{summary}', expected sentiment '{expected}' "
        f"but got '{data['sentiment']}'"
    )
