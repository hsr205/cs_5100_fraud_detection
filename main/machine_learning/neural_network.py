import pandas as pd
import torch
from logger import Logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

logger: Logger = Logger().get_logger()

"""
We define our neural network by subclassing nn.Module
"""


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Using {device} device")


class DataPreprocessor:

    def __init__(self, fraud_data_frame: pd.DataFrame):
        self.fraud_data_frame = fraud_data_frame

    def preprocess_data_frame(self) -> pd.DataFrame:
        """

        Takes a 'random' sample of size "sample_size" (defaulted is 100,000) observations
        and conducts data preprocessing on each of the features

        :return: pandas data frame containing the preprocesses data
        """

        random_seed: int = 42
        sample_size: int = 100000

        # Lists columns of the data set to include
        # Potentially add columns nameOrig and nameDest (account numbers) to future analysis
        column_list: list[str] = ["type",
                                  "amount",
                                  "newbalanceOrig",
                                  "newbalanceDest",
                                  "isFraud"]

        data_frame: pd.DataFrame = self.fraud_data_frame[column_list].sample(n=sample_size, random_state=random_seed)

        # Specifies the transformations each column needs to undergo
        preprocessor = ColumnTransformer(
            transformers=[
                ('amount_normalized', StandardScaler(), ['amount']),
                ('type_encoded', OneHotEncoder(), ['type']),
                ('new_balance_origin_normalized', StandardScaler(), ['newbalanceOrig']),
                ('new_balance_destination_normalized', StandardScaler(), ['newbalanceDest'])
            ],

            # Allows for columns that aren't being preprocessed to be included in the final data set
            remainder='passthrough'
        )

        processed_data = preprocessor.fit_transform(data_frame)

        # Convert back to DataFrame with column names for easier viewing
        result_column_list: list[str] = ['amount_norm',
                                         'type_PAYMENT', 'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_CASH_IN',
                                         'new_balance_origin_normalized', 'new_balance_destination_normalized',
                                         'isFraud']

        return pd.DataFrame(processed_data, columns=result_column_list)
