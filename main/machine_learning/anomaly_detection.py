from pathlib import Path

import pandas as pd

from data.custom_data_loader import CustomDataLoader
from data.fraud_data import FraudDataset
from logger import Logger
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from static.constants import Constants
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.utils import resample
import seaborn as sns

import matplotlib.pyplot as plt

from .neural_network import DataPreprocessor

logger: Logger = Logger().get_logger()

class IFProcessor(DataPreprocessor):

    # Override DataPreprocessor to include 'isFraud' column for frequency analysis
    def _get_column_list(self) -> list[str]:
        """
        Creates a list of all column names to be included in the dataframe

        :returns: List[str]: string list of all column names
        """

        return [Constants.TRANSACTION_TYPE,
                Constants.AMOUNT,
                Constants.NEW_TRANSACTION_BALANCE,
                Constants.NEW_RECIPIENT_BALANCE,
                Constants.IS_FRAUD]

    def _get_result_column_list(self) -> list[str]:
        return ['amount_norm',
                'type_PAYMENT', 'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_CASH_IN',
                'new_balance_origin_normalized', 'new_balance_destination_normalized', 'isFraud']



# Represents an Isolation Forest Model to be used for anomaly detection
class IFModel:
    def __init__(self, fraud_data_frame: pd.DataFrame):
        self.fraud_data_frame = fraud_data_frame
        self.random_seed = 42

    # preprocess dataframe
    def preprocess(self):
        ifp = IFProcessor(self.fraud_data_frame)
        self.fraud_data_frame = ifp._preprocess_data_frame(self.fraud_data_frame)
        self.fraud_data_frame.sort_index()
       
    # Identifies anomalies in the data set based on amount
    def detect(self, num_observations=16000):
        logger.info(f"Processing {num_observations} observations")
        if num_observations > len(self.fraud_data_frame[self.fraud_data_frame['isFraud'] == 1]): # if number of observations is greater than the number of anomalies in the dataset
            self.fraud_data_frame = self.fraud_data_frame.sort_values(by='isFraud', ascending=False) # include all anomlies
            
        self.fraud_data_frame = self.fraud_data_frame.head(num_observations)  # only use limited amount to make analysis more efficient
        self.preprocess()

        # Create ID column to track individual transactions after the model is trained and evaluated
        self.fraud_data_frame['ID'] = range(1, len(self.fraud_data_frame) + 1)
        df = self.fraud_data_frame.copy(deep=False) # save original to compare with later
        self.fraud_data_frame.drop('isFraud', axis=1, inplace=True) # drop isFraud for supervised learning


        # Select features for anomaly detection
        features = ['amount_norm', 'new_balance_origin_normalized', 'new_balance_destination_normalized']
        X = self.fraud_data_frame[features]

        # Apply Isolation Forest --> parameters determined through trial and error
        logger.info("Constructing IsolationForest")
        iso_forest = IsolationForest(n_estimators=132, contamination=0.35, max_samples=0.15, random_state=self.random_seed)
        self.fraud_data_frame['predictedFraud'] = iso_forest.fit_predict(X)

        # Convert -1 to 1 for anomalies, and 1 to 0 for normal points
        self.fraud_data_frame['predictedFraud'] = self.fraud_data_frame['predictedFraud'].apply(lambda x: 1 if x == -1 else 0)

        logger.info("Creating 2d visualization")
        self.vis_2d()
        logger.info("Creating 3d visualization")
        self.vis_3d()

        combined_df = pd.merge(self.fraud_data_frame, df, on='ID', how='left')

        # Calculate accuracy
        accuracy = (combined_df['isFraud'] == combined_df['predictedFraud']).mean()
        logger.info(f"Accuracy: {accuracy}")

        # Calculate f1 score
        f1 = f1_score(combined_df['isFraud'], combined_df['predictedFraud'])
        logger.info(f"F1 Score: {f1}")

        self.view_classification(combined_df)


    # Visualize classification matrix
    def view_classification(self, df):
        y_true = df['isFraud']
        y_pred = df['predictedFraud']

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('view_classification.png')
        #plt.show()

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
        plt.savefig('anomaly_detection_3d.png')
        #plt.show()
