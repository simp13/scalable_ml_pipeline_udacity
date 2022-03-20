import pandas as pd
import numpy as np 
import pytest
import joblib
import logging
from training.training import train_test_model

def test_train_test_model():
    """
    Test train test model
    """

    train_test_model()

    try:
        joblib.load('model/model.joblib')
        joblib.load('model/lb.joblib')
        joblib.load('model/encoder.joblib')
        logging.info("Testing train test model: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train test model: The files waeren't found")
        raise err

