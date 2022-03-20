import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from training.data import process_data
from training.model import train_model


def train_test_model():
    """
    Execute model training
    """
    df = pd.read_csv("data/raw_data/census.csv")
    train, _ = train_test_split(df, test_size=0.20)

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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features,
        label="salary", training=True
    )

    trained_model = train_model(X_train, y_train)

    dump(trained_model, "model/model.joblib")
    dump(encoder, "model/encoder.joblib")
    dump(lb, "model/lb.joblib")