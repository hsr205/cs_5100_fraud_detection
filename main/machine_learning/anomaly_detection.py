from pathlib import Path

import pandas as pd

from data.custom_data_loader import CustomDataLoader
from data.fraud_data import FraudDataset

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from static.constants import Constants
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.utils import resample
import seaborn as sns

import matplotlib.pyplot as plt

from .neural_network import DataPreprocessor

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
        self.fraud_data_frame = ifp._preprocess_data_frame(self.fraud_data_frame) # gets all anomalies (~8000) and equal amount of non-anomaly
        self.fraud_data_frame.sort_index()
       
    # Identifies anomalies in the data set based on amount
    def detect(self):
        self.fraud_data_frame = self.fraud_data_frame.sort_values(by='isFraud', ascending=False).head(16000)
        self.preprocess()

        # Create ID column to track individual transactions after the model is trained and evaluated
        self.fraud_data_frame['ID'] = range(1, len(self.fraud_data_frame) + 1)
        df = self.fraud_data_frame.copy(deep=False) # save original to compare with later
        self.fraud_data_frame.drop('isFraud', axis=1, inplace=True) # drop isFraud for supervised learning


        # Select features for anomaly detection
        features = ['amount_norm', 'new_balance_origin_normalized', 'new_balance_destination_normalized']
        X = self.fraud_data_frame[features]

        # Apply Isolation Forest
        iso_forest = IsolationForest(n_estimators=132, contamination=0.05, max_samples=256, random_state=42)
        self.fraud_data_frame['predictedFraud'] = iso_forest.fit_predict(X)

        # Convert -1 to 1 for anomalies, and 1 to 0 for normal points
        self.fraud_data_frame['predictedFraud'] = self.fraud_data_frame['predictedFraud'].apply(lambda x: 1 if x == -1 else 0)

        #self.vis_2d()
        #elf.vis_3d()

        combined_df = pd.merge(self.fraud_data_frame, df, on='ID', how='left')
        
        #combined_df.to_csv('final_df.csv')

        # Calculate accuracy
        accuracy = (combined_df['isFraud'] == combined_df['predictedFraud']).mean()
        print(f"Accuracy: {accuracy}")

        # Calculate f1 score
        f1 = f1_score(combined_df['isFraud'], combined_df['predictedFraud'])
        print(f"F1 Score: {f1}")

    def find_best(self):
        # Hyperparameter grid
        n_estimators_range = [50, 100, 150, 200]
        max_samples_range = [0.3, 0.5, 0.8, 1.0]
        contamination_range = [0.01, 0.05, 0.1, 0.15]

        # Prepare a grid to store the results (accuracy for each combination of hyperparameters)
        results = []

        # Try all combinations of hyperparameters
        for n_estimators in n_estimators_range:
            for max_samples in max_samples_range:
                for contamination in contamination_range:
                    # Train IsolationForest with the current hyperparameters
                    model = IsolationForest(n_estimators=n_estimators, 
                                            max_samples=max_samples, 
                                            contamination=contamination, 
                                            random_state=42)
                    
                    # Split the data (train and test sets)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    # Fit the model
                    model.fit(X_train)
                    
                    # Predict on test set (model returns 1 for inliers and -1 for outliers)
                    y_pred = model.predict(X_test)
                    y_pred = (y_pred == -1).astype(int)  # Convert -1 to 1 (outlier) and 1 to 0 (inlier)
                    
                    # Compute accuracy (assuming 'y_test' is binary: 0 for normal, 1 for fraud/outlier)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Store results (accuracy, n_estimators, max_samples, contamination)
                    results.append((accuracy, n_estimators, max_samples, contamination))

        # Convert results to numpy array for easy manipulation
        results = np.array(results)

        # Extract accuracy, n_estimators, max_samples, contamination
        accuracies = results[:, 0]
        n_estimators_vals = results[:, 1]
        max_samples_vals = results[:, 2]
        contamination_vals = results[:, 3]

        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        sc = ax.scatter(n_estimators_vals, max_samples_vals, contamination_vals, c=accuracies, cmap='viridis')

        # Labels and title
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('max_samples')
        ax.set_zlabel('contamination')
        ax.set_title('Isolation Forest: Hyperparameter Tuning vs Accuracy')

        # Colorbar for accuracy
        plt.colorbar(sc, label='Accuracy')

        # Show plot
        plt.show()


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

        '''# Inspect anomalies
        anomalies = self.fraud_data_frame[self.fraud_data_frame['predictedFraud'] == 1]
        return anomalies'''

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

        '''return anomalies'''

