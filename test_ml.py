import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics
from ml.data import process_data

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "workclass": ["Private", "Self-emp", "Private", "Gov"],
        "education": ["Bachelors", "HS-grad", "Bachelors", "HS-grad"],
        "marital-status": ["Never-married", "Married", "Divorced", "Married"],
        "occupation": ["Tech-support", "Craft-repair", "Other-service", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Wife"],
        "race": ["White", "Black", "White", "Asian-Pac_Islander"],
        "sex": ["Male", "Female", "Female", "Male"],
        "native-country": ["United-States", "United-States", "United-States", "India"],
        "salary": [0, 1, 0, 1]
    })

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_random_forest(sample_data):
    """
    # atrain_model should return a RandomForestClassifier fitted model
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    #Check its RandomForestClassifier
    assert hasattr(model, "predict"), "Model should have predict method"
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_process_data_output_shapes_and_types(sample_data):
    """
    # process_data returns numpy arrays for features and labels,
    and fitted encoder and lable binarizer when training=True.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    assert hasattr(X, "shape"), "X should be numpy array or sparse matix with shape"
    assert hasattr(y, "shape"), "y should be numpy array with shape"
    assert len(X) == len(y), "features and labels must have the same number of samples"
    assert hasattr(encoder, "transform")
    assert hasattr(lb, "transform")




# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_returns_floats():
    """
    # compute_model_metrics should return floats for precision, recall, and fbeta
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
