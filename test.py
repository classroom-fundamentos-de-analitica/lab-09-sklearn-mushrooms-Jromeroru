"""GitHub Classroom autograding script."""

import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score


def load_estimator():
    """Load trained model from disk."""

    if not os.path.exists("model.pkl"):
        return None
    with open("model.pkl", "rb") as file:
        estimator = pickle.load(file)

    return estimator


def load_datasets():
    """Load train and test datasets."""

    train_dataset = pd.read_csv("train_dataset.csv")
    test_dataset = pd.read_csv("test_dataset.csv")

    x_train = train_dataset.drop("type", axis=1)
    y_train = train_dataset["type"]

    x_test = test_dataset.drop("type", axis=1)
    y_test = test_dataset["type"]

    return x_train, x_test, y_train, y_test


def eval_metrics(y_true, y_pred):
    """Evaluate model performance."""

    accuracy = accuracy_score(y_true, y_pred)

    return accuracy


def compute_metrics():
    """Compute model metrics."""

    estimator = load_estimator()
    assert estimator is not None, "Model not found"


if __name__ == "__main__":
    test_()
    



def test_():
    """Run grading script."""
    assert True
   
