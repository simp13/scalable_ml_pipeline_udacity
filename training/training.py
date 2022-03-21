from joblib import dump
from training.data import process_data
from training.model import train_model, compute_model_metrics


def train_test_model(training_data, testing_data):
    """
    Execute model training
    """

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
        training_data, categorical_features=cat_features,
        label="salary", training=True
    )

    trained_model = train_model(X_train, y_train)

    print("[INFO] Saving model...")
    dump(trained_model, "model/model.joblib")
    dump(encoder, "model/encoder.joblib")
    dump(lb, "model/lb.joblib")

    print("[INFO] Running scoring...")
    X_test, y_test, encoder, lb = process_data(
        training_data, categorical_features=cat_features,
        encoder=encoder, lb=lb,
        label="salary", training=False
    )
    preds = trained_model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print("[INFO] Precision {}-Recall {}-Fbeta {}".format(
        precision,
        recall,
        fbeta))
