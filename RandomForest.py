import processing_utils as process
import pandas as pd
import numpy as np
from BaseModel import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class RandomForestModel(BaseModel):
    """Methods for training a decision tree model."""

    def fit_and_train(self, X_train, X_val, y_train):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = rf_model.predict(X_val)

        return y_pred