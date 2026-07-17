import pytest
from fastapi.testclient import TestClient
from task_6.main import api

@pytest.fixture(scope="module")
def client():
    with TestClient(api) as c:
        yield c

def test_single_prediction_low_income(client):
    # Payload for <=50K (young, low education, low hours)
    payload = {
        "age": 18,
        "workclass": "Private",
        "education": "11th",
        "education-num": 7,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data
    assert data["prediction"] == 0
    assert data["label"] == "<=50K"

def test_single_prediction_high_income(client):
    # PAyload fr >50K (mid-career, advanced degree, high capital gain)
    payload = {
        "age": 45,
        "workclass": "Private",
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data
    # High capital gain -> >50K prediction
    assert data["prediction"] == 1
    assert data["label"] == ">50K"

def test_batch_prediction(client):
    payload = [
        {
            "age": 20,
            "workclass": "Private",
            "education": "Some-college",
            "education-num": 10,
            "marital-status": "Never-married",
            "occupation": "Sales",
            "relationship": "Own-child",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 30,
            "native-country": "United-States"
        },
        {
            "age": 50,
            "workclass": "Self-emp-inc",
            "education": "Doctorate",
            "education-num": 16,
            "marital-status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 99999,
            "capital-loss": 0,
            "hours-per-week": 60,
            "native-country": "United-States"
        }
    ]
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    predictions = data["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["prediction"] == 0
    assert predictions[1]["prediction"] == 1
