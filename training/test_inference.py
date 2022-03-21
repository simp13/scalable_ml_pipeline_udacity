import pandas as pd
import numpy as np 
import pytest
from pandas.core.frame import DataFrame
from training.data import process_data
from training.model import inference
from joblib import load

def test_inference_below():
    """
    Test inference function
    """
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    array = np.array([[
                     39,
                     "State-gov",
                     77516,
                     "Bachelors",
                     13,
                     "Never-married",
                     "Adm-clerical",
                     "Not-in-family",
                     "White",
                     "Male",
                     2174,
                     0,
                     40,
                     "United-States"
                     ]])
    df_temp = DataFrame(data=array, columns=[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"
    ])

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

    X, _, _, _ = process_data(
                df_temp,
                categorical_features= cat_features,
                encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"

def test_inference_above():
    """
    Test inference function
    """
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    array = np.array([[
                     42,
                     "Private",
                     159449,
                     "Bachelors",
                     13,
                     "Married-civ-spouse",
                     "Exec-managerial",
                     "Husband",
                     "White",
                     "Male",
                     5178,
                     0,
                     40,
                     "United-States"
                     ]])
    df_temp = DataFrame(data=array, columns=[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"
    ])

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

    X, _, _, _ = process_data(
                df_temp,
                categorical_features= cat_features,
                encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"