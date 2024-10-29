import numpy as np
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

    # TODO: This is just a testing method for preprocessing a single column
    #       Will eventually preprocess all features in a single method
    def preprocess_type_column_data(self) -> np.array:
        """
        Ingests a pandas data frame converting the dataframe into a numpy array
        exclusively for the data frame's `type` column

        Converts the pandas data series into an encoded numpy array

        :return:
        """

        random_seed: int = 42
        sample_size: int = 100000
        # Takes a sample of the data_series and converts it to a 2-D numpy array
        type_numpy_array: np.array = (self.fraud_data_frame['type'].sample(n=sample_size,
                                                                           random_state=random_seed)
                                      .to_numpy().reshape(-1, 1))
        encoder = OneHotEncoder()

        # Encodes the Numpy Array
        type_encoded_array = encoder.fit_transform(type_numpy_array).toarray()

        return type_encoded_array

    def preprocess_amount_column_data(self) -> np.array:
        random_seed: int = 42
        sample_size: int = 100000

        # Takes a sample of the data_series and converts it to a 2-D numpy array
        type_numpy_array: np.array = (self.fraud_data_frame['amount'].sample(n=sample_size,
                                                                             random_state=random_seed)
                                      .to_numpy().reshape(-1, 1))
        scaler = StandardScaler()

        # Standardizes values into a Numpy Array
        amount_encoded_array = scaler.fit_transform(type_numpy_array)

        logger.info(f"amount_encoded_array = {amount_encoded_array}")

        return amount_encoded_array

    # TODO: Need to determine what transformation we want to do for the nameOrig and nameDest columns
    #       Also need to write documentation for this method
    def preprocess_data_frame(self) -> pd.DataFrame:
        random_seed: int = 42
        sample_size: int = 100000

        # Lists columns of the data set to include
        column_list: list[str] = ["type",
                                  "amount",
                                  "nameOrig",
                                  "nameDest",
                                  "newbalanceOrig",
                                  "newbalanceDest",
                                  "isFraud"]



        data_frame: pd.DataFrame = self.fraud_data_frame[column_list].sample(n=sample_size, random_state=random_seed)

        # Specifies the transformations each column needS to undergo
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
        result_column_list: list[str] = ['amount_norm', 'type_PAYMENT', 'type_TRANSFER', 'type_CASH_OUT',
                                         'type_DEBIT', 'type_CASH_IN',
                                         'new_balance_origin_normalized', 'new_balance_destination_normalized',
                                         'nameOrig', 'nameDest', 'isFraud']

        return pd.DataFrame(processed_data, columns=result_column_list)
