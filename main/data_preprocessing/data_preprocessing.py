import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import torch
import os


class DataTransformer:
  def __init__(self, df):
    """
        Initializes the DataTransformer with a DataFrame.

        Parameters:
            df (pd.DataFrame): The input data as a pandas DataFrame.
        """
    self.df = df

  def preprocess_data_frame(self):
    """
        Preprocesses the DataFrame by performing several transformations:
        - Drops unnecessary columns.
        - Maps categorical payment types to numeric codes.
        - Standardizes specified numeric columns.
        - Creates new features based on existing columns.
        - Applies quantile transformation to normalize certain columns.
        - Saves the transformed DataFrame to a CSV file with only 10,000 rows.
        """
    # Drop specified columns
    self.df = self.df.drop(['nameOrig', 'nameDest'], axis=1)

    # Map payment types to numeric values
    payment_mapping = {
      'PAYMENT': 0,
      'TRANSFER': 1,
      'CASH_OUT': 2,
      'DEBIT': 3,
      'CASH_IN': 4
    }
    self.df['type'] = self.df['type'].map(payment_mapping)

    # Standard scaling without library
    def standard_scale(column):
      mean = column.mean()
      std = column.std()
      return (column - mean) / std

    self.df['amount'] = standard_scale(self.df['amount'])
    self.df['oldbalanceOrg'] = standard_scale(self.df['oldbalanceOrg'])
    self.df['oldbalanceDest'] = standard_scale(self.df['oldbalanceDest'])

    # Calculate old_to_new_balance_ratio
    self.df['old_to_new_balance_ratio'] = self.df['oldbalanceOrg'] / (
        self.df['newbalanceOrig'] + 1e-10)

    # Calculate balance differences
    self.df['oldbalanceOrg_minus_amount'] = self.df['oldbalanceOrg'] - self.df[
      'amount']
    self.df['oldbalanceDest_minus_amount'] = self.df['oldbalanceDest'] - \
                                             self.df['amount']

    # Log transform of the amount
    self.df['log_amount'] = np.log1p(self.df['amount'])

    # Quantile transformation with sklearn
    qt = QuantileTransformer(output_distribution='normal')
    self.df[['amount', 'oldbalanceOrg', 'oldbalanceDest']] = qt.fit_transform(
        self.df[['amount', 'oldbalanceOrg', 'oldbalanceDest']]
    )

    # Define the file path in the data_preprocessing folder
    output_path = os.path.join("data_preprocessing", "transformed_data.csv")

    # Save the transformed DataFrame to a CSV file
    self.df.to_csv(output_path, index=False)

  def get_y_labels_as_tensor(self):
    """
        Retrieves the target variable 'isFraud' as a PyTorch tensor.

        Returns:
            torch.Tensor: A tensor containing the target labels.
        """
    y_labels = self.df['isFraud'].values
    return torch.tensor(y_labels, dtype=torch.float32)

  def get_x_labels_as_tensor(self):
    """
        Retrieves the feature variables as a PyTorch tensor, excluding the target 'isFraud'.

        Returns:
            torch.Tensor: A tensor containing the feature data.
        """
    x_labels = self.df.drop(columns=['isFraud'])
    return torch.tensor(x_labels.values, dtype=torch.float32)
