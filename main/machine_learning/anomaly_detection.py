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

from .neural_network import DataPreprocessor

# Represents an Isolation Forest Model to be used for anomaly detection
class IFModel:
    def __init__(self, fraud_data_frame: pd.DataFrame):
        self.fraud_data_frame = fraud_data_frame
        self.random_seed = 42


    # preprocess dataframe
    def preprocess(self):
        print(f"Anomaly value counts before processing: {self.fraud_data_frame['isFraud'].value_counts()}")
        #self.fraud_data_frame = self.fraud_data_frame.sort_values(by=['isFraud'], ascending=False)
        dp = DataPreprocessor(self.fraud_data_frame)
        self.fraud_data_frame = dp._preprocess_data_frame(self.fraud_data_frame) # gets all anomalies (~8000) and equal amount of non-anomaly
        self.fraud_data_frame.sort_index()
       
    # Identifies anomalies in the data set based on amount
    def detect(self):
        self.fraud_data_frame = self.fraud_data_frame.sort_values(by='isFraud', ascending=False).head(16000)
        df = self.fraud_data_frame # save original to compare with later
        self.preprocess()

        # Select features for anomaly detection
        features = ['amount_norm', 'new_balance_origin_normalized', 'new_balance_destination_normalized']
        X = self.fraud_data_frame[features]

        # Apply Isolation Forest
        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self.fraud_data_frame['predictedFraud'] = iso_forest.fit_predict(X)

        # Convert -1 to 1 for anomalies, and 1 to 0 for normal points
        self.fraud_data_frame['predictedFraud'] = self.fraud_data_frame['predictedFraud'].apply(lambda x: 1 if x == -1 else 0)

        anomalies = self.fraud_data_frame[self.fraud_data_frame['predictedFraud'] == 1]

        self.fraud_data_frame.to_csv('fdf.csv')

        self.vis_2d()
        self.vis_3d()

        print(f"anomaly columns: {anomalies.columns}")
        print(f"df columns {df.columns}")

        combined_df = pd.merge(anomalies, df, left_on=features,
                               right_on=['amount', 'newbalanceOrig', 'newbalanceDest'],
                               how='left')
        
        combined_df = combined_df[[
            'amount_norm', 'new_balance_origin_normalized', 'new_balance_destination_normalized', 'predictedFraud', 'isFraud'
        ]]
        
        combined_df.to_csv('combined.csv')

    # Visualizes IsolationForest in two dimensions (amount_norm and new_balance_origin_normalized)
    def vis_2d(self):
        # Visualize anomalies
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.fraud_data_frame, x='amount_norm', y='new_balance_origin_normalized', hue='predictedFraud', palette={0: 'green', 1: 'red'}, s=100, alpha=0.6)
        plt.title("Anomaly Detection using Isolation Forest")
        plt.xlabel("Amount")
        plt.ylabel("New Balance Origin")

        plt.savefig('anomaly_detection_2d.png')
        #plt.show()

        # Inspect anomalies
        anomalies = self.fraud_data_frame[self.fraud_data_frame['predictedFraud'] == 1]
        return anomalies

    # Visualizes IsolationForest in three dimensions (amount_norm, new_balance_origin_normalized, and new_balance_destination_normalized)
    def vis_3d(self):

        # 3D scatter plot of the anomalies and normal points
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot normal points
        normal_points = self.fraud_data_frame[self.fraud_data_frame['predictedFraud'] == 0]
        ax.scatter(normal_points['amount_norm'], normal_points['new_balance_origin_normalized'], 
                normal_points['new_balance_destination_normalized'], color='green', s=50, alpha=0.6, label='Normal')

        # Plot anomalous points
        anomalies = self.fraud_data_frame[self.fraud_data_frame['predictedFraud'] == 1]
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
        #plt.savefig('anomaly_detection_3d.png')
        plt.show()

        return anomalies

