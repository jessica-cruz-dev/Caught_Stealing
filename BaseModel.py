from typing import Tuple

import pandas as pd
import numpy as np

import plotting_utils as plot
from IPython.display import display

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


class BaseModel(object):
    """Methods for processing all models."""
    def __init__(
            self,
            train_file_path,
            test_file_path
    ):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.train = self._read_data(self.train_file_path)
        self.test = self._read_data(self.test_file_path)

    def _read_data(self, file_path) -> pd.DataFrame:
        """Read in excel workbook."""
        data = pd.read_excel(file_path)
        return data

    def display_shape(self) -> None:
        """Output shape details to screen."""
        print("Shape: ", self.train.shape)

    def display_missing_values(self) -> None:
        """Output any missing values to screen."""
        missing_values = self.train.isnull().sum()
        print("Missing values in Train:\n", missing_values[missing_values > 0])
        missing_values = self.test.isnull().sum()
        print("\nMissing values in Test:\n", missing_values[missing_values > 0])

    def _fill_missing_data(self, data) -> pd.DataFrame:
        """Fill missing data."""
        # Drop columns not used in predictions
        drop_columns = [col for col in data.columns if "_id" in col]
        if 'cs_prob' in data.columns:
            drop_columns += ['cs_prob']
        data.drop(columns=drop_columns, inplace=True)

        # Fill missing values in numerical columns with median
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            data[col].fillna(data[col].median(), inplace=True)

        # Fill missing values in categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)

        return data

    def encode_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode non-numerical columns."""
        categorical_cols = list(data.select_dtypes(include=['object']).columns)

        data = pd.get_dummies(data, columns=categorical_cols)

        return data

    def preprocess_data(self) -> None:
        """Combines all preoprocessing steps for both test and train datasets."""
        self.train = self._fill_missing_data(self.train)
        self.train = self.encode_data(self.train)
        self.test = self._fill_missing_data(self.test)
        self.test = self.encode_data(self.test)
    
    def print_correlation_report(self) -> None:
        """Output training correlation report to screen."""
        correlations = self.train.corrwith(
            self.train['is_cs']
        ).sort_values(ascending=False)
        print(correlations)

    def bin_pitch_columns(self) -> None:
        """Cluster pitches by type."""
        pitching_columns = [p for p in self.train.columns if p.startswith('p_')]
        train_pitching = self.train[pitching_columns]

        # Scaling for K-means processing
        scaler = StandardScaler()
        scaled_pitching_data = scaler.fit_transform(train_pitching)

        # Determine the optimal number of clusters using the Elbow method
        inertia = []
        k_values = range(1, 15)
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_pitching_data)
            inertia.append(kmeans.inertia_)

        # Plot the Elbow curve
        plot.elbow_curve(k_values, inertia)

        # K-means clustering for k=4
        kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels_4 = kmeans_4.fit_predict(scaled_pitching_data)

        train_pitch = train_pitching.copy()
        train_pitch.loc[:,"cluster_k4"] = labels_4

        # Compute the mean values of features for each cluster (k=4)
        cluster_means_k4 = train_pitch.groupby('cluster_k4').mean()

        display(cluster_means_k4.reset_index())

        pitch_labels_k4 = {
            0: "Sinker/Slider",
            1: "Fastball",
            2: "Curveball",
            3: "Changeup/Two_seam",
        }
        # Update dataset with new column
        self.train['pitch_type_k4'] = labels_4
        self.train['pitch_type_k4'] = self.train['pitch_type_k4'].map(pitch_labels_k4)

    def split_train(
            self
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split train data for model fitting."""
        X = self.train.drop(columns=['is_cs'])
        y = self.train['is_cs']
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def fit_and_train(self):
        """Implementation needs to be defined in each respective child model."""
        raise NotImplementedError
    
    def print_classification_report(self, y_pred, y_val) -> np.ndarray:
        """Calculate accuracy and generate classification report."""
        accuracy = accuracy_score(y_val, y_pred)

        # Generate classification report
        class_report = classification_report(y_val, y_pred)

        print(f"Accuracy: {accuracy}\nClass report:\n{class_report}")