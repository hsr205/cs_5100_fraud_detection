import numpy as np
import pandas as pd
import torch
from logger import Logger
from sklearn.preprocessing import OneHotEncoder
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
    encoder = OneHotEncoder()

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

        # Encodes the Numpy Array
        type_encoded_array = self.encoder.fit_transform(type_numpy_array).toarray()

        return type_encoded_array
