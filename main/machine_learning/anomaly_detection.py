from pathlib import Path

import pandas as pd

from data.custom_data_loader import CustomDataLoader
from data.fraud_data import FraudDataset

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import seaborn as sns

import matplotlib.pyplot as plt

from .neural_network import DataPreprocessor as dp

'''
Notes:
- https://medium.com/@corymaklin/isolation-forest-799fceacdda4
- https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.IsolationForest.html
- https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html

'''

class IFModel:
    def __init__(self, fraud_data_frame: pd.DataFrame):
        self.fraud_data_frame = fraud_data_frame
        self.random_seed = 42

    def detect_3d(self):
        df = self.fraud_data_frame = dp.preprocess_data_frame(self, 500000)
        print(f"DF: {df}")

        print(f"Columns: {df.columns}")

        # Select features for anomaly detection
        features = ['amount_norm', 'new_balance_origin_normalized', 'new_balance_destination_normalized']
        X = df[features]

        # Apply Isolation Forest
        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        df['anomaly'] = iso_forest.fit_predict(X)

        # Convert -1 to 1 for anomalies, and 1 to 0 for normal points
        df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

        # 3D scatter plot of the anomalies and normal points
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot normal points
        normal_points = df[df['anomaly'] == 0]
        ax.scatter(normal_points['amount_norm'], normal_points['new_balance_origin_normalized'], 
                normal_points['new_balance_destination_normalized'], color='blue', s=50, alpha=0.6, label='Normal')

        # Plot anomalous points
        anomalies = df[df['anomaly'] == 1]
        ax.scatter(anomalies['amount_norm'], anomalies['new_balance_origin_normalized'], 
                anomalies['new_balance_destination_normalized'], color='red', s=50, alpha=0.6, label='Anomalies')

        # Labels and title
        ax.set_xlabel('Amount Norm')
        ax.set_ylabel('New Balance Origin Normalized')
        ax.set_zlabel('New Balance Destination Normalized')
        ax.set_title("3D Anomaly Detection using Isolation Forest")

        # Show the legend
        ax.legend()

        # Show the plot
        plt.show()

        # Inspect anomalies
        print("Anomalies detected:")
        print(anomalies)

    # Identifies anomalies in the data set based on amount
    def detect(self):
        df = self.fraud_data_frame = dp.preprocess_data_frame(self, 500000)
        print(f"DF: {df}")
        print(f"Columns: {df.columns}")

        # Select features for anomaly detection
        features = ['amount_norm', 'new_balance_origin_normalized', 'new_balance_destination_normalized']
        X = df[features]

        # Apply Isolation Forest
        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        df['anomaly'] = iso_forest.fit_predict(X)

        # Convert -1 to 1 for anomalies, and 1 to 0 for normal points
        df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

        # Visualize anomalies
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='amount_norm', y='new_balance_origin_normalized', hue='anomaly', palette={0: 'blue', 1: 'red'}, s=100, alpha=0.6)
        plt.title("Anomaly Detection using Isolation Forest")
        plt.xlabel("Amount")
        plt.ylabel("New Balance Origin")
        plt.show()

        # Inspect anomalies
        anomalies = df[df['anomaly'] == 1]
        print(anomalies)

    def detect_2(self):
        df = self.fraud_data_frame = dp.preprocess_data_frame(self, 500000)


    def detect(self):
        df = dp.preprocess_data_frame(self, 500000)
        df = self.fraud_data_frame
        print(f"df: {df}")

        print("Fraud Dataframe Created")

        print(f"Columns: {df.columns}")

        majority_df = df[df["isFraud"] == 0]
        minority_df = df[df["isFraud"] == 1]
        minority_downsampled_df = resample(minority_df, replace=True, n_samples=30, random_state=42)
        downsampled_df = pd.concat([majority_df, minority_downsampled_df])

        y = downsampled_df["isFraud"]
        X = downsampled_df.drop("isFraud", axis=1)

        # convert categorical columns to numeric (one-hot encoding)
        print("Converting columns...")
        print(f"Old columns: {X.columns}")
        categorical_columns = ['type', 'nameOrig', 'nameDest']  # Adjust this list based on your dataset

        #X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        # print(f"New columns: {X.columns}")

        #print(f"New columns: {X.columns}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        model = IsolationForest(random_state=42)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        self.create_scatter_plots(X_train, y_pred_train, 'Training Data', X_test, y_pred_test, 'Test Data')

'''

    def create_scatter_plots(self, X1, y1, title1, X2, y2, title2):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Scatter plot for the first set of data
        axes[0].scatter(X1[y1==1, 0], X1[y1==1, 1], color='green', label='Normal')
        axes[0].scatter(X1[y1==-1, 0], X1[y1==-1, 1], color='red', label='Anomaly')
        axes[0].set_title(title1)
        axes[0].legend()

        # Scatter plot for the second set of data
        axes[1].scatter(X2[y2==1, 0], X2[y2==1, 1], color='green', label='Normal')
        axes[1].scatter(X2[y2==-1, 0], X2[y2==-1, 1], color='red', label='Anomaly')
        axes[1].set_title(title2)
        axes[1].legend()

        plt.tight_layout()
        plt.show()
'''