import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def correct_columns():
    return ['age', 'sex', 'cp', 'trestbps',
            'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal']


@pytest.fixture
def incorrect_columns(correct_columns):
    correct_columns[0] = 'wrong_column_name'
    return correct_columns


@pytest.fixture
def correct_values_sample():
    return [[41.0, 1.0, 1.0, 120.0, 157.0, 0.0, 1.0, 182.0, 0.0, 0.0, 2.0, 0.0, 2.0]]


@pytest.fixture
def incorrect_values_sample(correct_values_sample):
    correct_values_sample[0][0] = 150
    return correct_values_sample


def make_request(client, columns, values):
    request = {
        'columns': columns,
        'values': values,
    }
    return client.get('/predict', json=request)


def test_with_correct_input(client, correct_columns, correct_values_sample):
    response = make_request(client, correct_columns, correct_values_sample)
    assert response.status_code == 200
    response_body = response.json()
    assert len(response_body) == 1
    assert list(response_body[0].keys()) == ['target']
    assert response_body[0]['target'] in (0, 1)


def test_with_several_correct_input_samples(client, correct_columns, correct_values_sample):
    response = make_request(client, correct_columns, correct_values_sample * 2)
    assert response.status_code == 200
    response_body = response.json()
    assert len(response_body) == 2
    assert all([list(x.keys()) == ['target'] for x in response_body])
    assert all([x['target'] in (0, 1) for x in response_body])


def test_incorrect_columns(client, incorrect_columns, correct_values_sample):
    response = make_request(client, incorrect_columns, correct_values_sample)
    assert response.status_code == 400


def test_incorrect_values(client, correct_columns, incorrect_values_sample):
    response = make_request(client, correct_columns, incorrect_values_sample)
    assert response.status_code == 400
