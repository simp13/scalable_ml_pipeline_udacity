import pandas as pd
import pytest
from training.data import process_data
from joblib import load


@pytest.fixture
def dataset():
    """
    Get dataset
    """
    df = pd.read_csv("data/raw_data/census.csv")
    return df


def test_process_data(dataset):
    """
    test process data function
    """
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_test, y_test, _, _ = process_data(
        dataset,
        categorical_features=cat_features,
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)
