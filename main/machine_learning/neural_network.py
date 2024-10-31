from dataclasses import dataclass

import pandas as pd
import torch
from logger import Logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from static.constants import Constants
from torch import nn, Tensor, device
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

logger: Logger = Logger().get_logger()

"""
We define our neural network by subclassing nn.Module
"""


class DataPreprocessor:
    random_seed: int = 42
    sample_size: int = 100000

    def __init__(self, fraud_data_frame: pd.DataFrame):
        self.fraud_data_frame = fraud_data_frame

    def get_tensor_dataset(self, batch_size: int = 32):
        features: Tensor = self.get_x_labels_as_tensor()
        outputs: Tensor = self.get_y_labels_as_tensor()
        dataset: TensorDataset = TensorDataset(features, outputs)
        dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

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
        tensor = torch.tensor(data_series.values, dtype=torch.float32).unsqueeze(1)

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


class NeuralNetwork(nn.Module):
    """
    A feedforward neural network that contains a input layer, hidden layer and an output layer
    """
    tensor: Tensor = Tensor()

    def __init__(self, input_size: int):
        super(NeuralNetwork, self).__init__()
        # Specifies the input_size (default = 8) and the number of nodes in the output layer (8)
        self.input_layer = nn.Linear(input_size, 8)
        # Indicates that after the input layer is completed the ReLU (rectified linear unit)
        # activation function is being called on the first hidden layer's nodes
        self.relu1 = nn.ReLU()
        # Second hidden layer indicating that the number of input nodes is 10 and the number of output nodes is 5
        self.hidden_layer = nn.Linear(8, 5)
        self.relu2 = nn.ReLU()  # ReLU (rectified linear unit) used again on the hidden layer
        self.output_layer = nn.Linear(5, 1)  # Output layer consisting of a single node
        # Applying a sigmoid function on the output layer to retrieve a probability from the output node
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor_obj: Tensor) -> Tensor:
        input_layer: Tensor = self.relu1(self.input_layer(tensor_obj))
        hidden_layer: Tensor = self.relu2(self.hidden_layer(input_layer))
        output_layer: Tensor = self.sigmoid(self.output_layer(hidden_layer))
        return output_layer


@dataclass
class Model:

    def __init__(self, fraud_data_frame: pd.DataFrame, input_size: int = 8):
        self.device = self.get_device()
        self.neural_network = NeuralNetwork(input_size).to(self.device)
        self.criterion = BCELoss()
        self.optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=0.001)
        self.fraud_data_frame = fraud_data_frame
        self.data_preprocessor = DataPreprocessor(fraud_data_frame=self.fraud_data_frame)

    def train_neural_network(self):
        num_epochs = 10
        for epoch in tqdm(range(num_epochs)):
            running_loss: float = 0.0
            for inputs, labels in self.data_preprocessor.get_tensor_dataset():
                inputs = inputs.to(self.device)  # Move tensor inputs to same devise the Model is located
                inputs = inputs.view(inputs.size(0), -1)
                # Move tensor labels to same devise the Model is located and convert values to 32-bit
                labels = labels.to(self.device, dtype=torch.float32)
                self.optimizer.zero_grad()  # Zero the gradients as we are going through a new interation
                outputs = self.neural_network(inputs)  # Forward pass through the neural network
                loss = self.criterion(outputs, labels)  # Compute the loss function based of BCELoss
                loss.backward()  # Backward pass (compute gradients) altering the weights and biases
                self.optimizer.step()  # Update parameters
                running_loss += loss.item()
            epoch_loss: float = running_loss / len(self.data_preprocessor.get_tensor_dataset())
            logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    @staticmethod
    def get_device() -> device:
        device_used: torch.device = torch.device(
            Constants.CUDA
            if torch.cuda.is_available()
            else Constants.MPS
            if torch.backends.mps.is_available()
            else Constants.CPU
        )

        logger.info(f"Using {device_used} device")

        return device_used