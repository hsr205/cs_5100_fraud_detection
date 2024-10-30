import pandas as pd
import torch
from logger import Logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from static.constants import Constants
from torch import nn, Tensor

logger: Logger = Logger().get_logger()

"""
We define our neural network by subclassing nn.Module
"""


class DataPreprocessor:
    random_seed: int = 42
    sample_size: int = 100000

    def __init__(self, fraud_data_frame: pd.DataFrame):
        self.fraud_data_frame = fraud_data_frame

    def get_y_labels_as_tensor(self) -> Tensor:
        """
        Retrieves the isFraud column (series) and takes a predefined sample size (100,000 observations).
        Converts the isFraud series to a tensor object.

        :return: a tensor object that will be fed as labels ('y' values) to the neural network (model)
        """

        data_series: pd.Series = self.fraud_data_frame[Constants.IS_FRAUD].sample(n=self.sample_size,
                                                                                  random_state=self.random_seed)

        # We convert the row vector in a column vector in order to ensure that
        # the shape matches the shape of the model's output tensor
        tensor = torch.tensor(data_series.values).unsqueeze(1)

        return tensor

    def get_x_labels_as_tensor(self) -> Tensor:
        """
        Retrieves the all feature columns ('x' values) and takes a predefined sample size (100,000 observations).
        Converts the feature columns to a tensor object.

        :return: a tensor object that will be fed as features ('x' values) to the neural network (model)
        """

        data_frame: pd.DataFrame = self.preprocess_data_frame()

        tensor = torch.tensor(data_frame.values, dtype=torch.float32)

        return tensor

    def preprocess_data_frame(self) -> pd.DataFrame:
        """

        Takes a 'random' sample of size "sample_size" (defaulted is 100,000) observations
        and conducts data preprocessing on each of the features

        :return: pandas data frame containing the preprocesses data
        """

        # Lists columns of the data set to include
        # Potentially add columns nameOrig and nameDest (account numbers) to future analysis
        column_list: list[str] = [Constants.TRANSACTION_TYPE,
                                  Constants.AMOUNT,
                                  Constants.NEW_TRANSACTION_BALANCE,
                                  Constants.NEW_RECIPIENT_BALANCE]

        data_frame: pd.DataFrame = self.fraud_data_frame[column_list].sample(n=self.sample_size,
                                                                             random_state=self.random_seed)

        # Specifies the transformations each column needs to undergo
        preprocessor = ColumnTransformer(
            transformers=[
                ('amount_normalized', StandardScaler(), [Constants.AMOUNT]),
                ('type_encoded', OneHotEncoder(), [Constants.TRANSACTION_TYPE]),
                ('new_balance_origin_normalized', StandardScaler(), [Constants.NEW_TRANSACTION_BALANCE]),
                ('new_balance_destination_normalized', StandardScaler(), [Constants.NEW_RECIPIENT_BALANCE])
            ],

            # Allows for columns that aren't being preprocessed to be included in the final data set
            remainder='passthrough'
        )

        processed_data = preprocessor.fit_transform(data_frame)

        # Convert back to DataFrame with column names for easier viewing
        result_column_list: list[str] = ['amount_norm',
                                         'type_PAYMENT', 'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_CASH_IN',
                                         'new_balance_origin_normalized', 'new_balance_destination_normalized']

        return pd.DataFrame(processed_data, columns=result_column_list)


# TODO: Still testing this clas
class NeuralNetwork(nn.Module):
    tensor: Tensor = Tensor()
    fraud_data_frame: pd.DataFrame = pd.DataFrame
    data_preprocessor: DataPreprocessor = DataPreprocessor(fraud_data_frame=fraud_data_frame)

    def __init__(self):
        super().__init__()
        device = torch.device(
            Constants.CUDA
            if torch.cuda.is_available()
            else Constants.MPS
            if torch.backends.mps.is_available()
            else Constants.CPU
        )
        tensor = self.tensor.to(device)
        logger.info(f"Using {device} device")
        # Creates the input layer of the neural network: self.data_preprocessor.get_x_labels_as_tensor().shape[1] (4)
        # and specifies the number of output nodes in the following hidden layer (10)
        self.fc1 = nn.Linear(self.data_preprocessor.get_x_labels_as_tensor().shape[1], 10)
        # Following hidden layer
        self.fc2 = nn.Linear(10, 5)
        # Output layer
        self.fc3 = nn.Linear(5, 1)
