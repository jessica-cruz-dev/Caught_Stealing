import pandas as pd
import numpy as np

from BaseModel import BaseModel
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(BaseModel):
    """Methods for training a random forest model."""

    def __init__(self, train_file_path, test_file_path):
        super().__init__(train_file_path, test_file_path)
        self.rf_model = None

    def fit_and_train(self, X_train, X_val, y_train):
        """Fit model with training data."""
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        # Predict on the validation set
        y_pred = self.rf_model.predict(X_val)
        return y_pred
    
    def predict_test_data(self):
        """Populate probability in test data."""
        rf_cs_probabilities = self.rf_model.predict_proba(self.test)[:, 1]
        self.test['cs_prob'] = rf_cs_probabilities
        print((f"First 5 rows of test dataset with populated 'cs_prob' column:\n"
              f"{self.test.cs_prob.head()}"))
