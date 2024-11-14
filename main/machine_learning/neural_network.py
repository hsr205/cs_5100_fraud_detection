import os
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np

# adding path to sys for local module imports
sys.path.append(os.path.join(os.getcwd(), "main"))

import pandas as pd
import torch
from logger import Logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from static.constants import Constants
from torch import nn, Tensor, device
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

logger: Logger = Logger().get_logger()


class DataPreprocessor:
    def __init__(self, fraud_data_frame: pd.DataFrame):
        self.fraud_data_frame = fraud_data_frame

    def get_training_loader(self) -> DataLoader:
        """
        Prepares and returns a DataLoader object for the training dataset.

        :return: DataLoader: A DataLoader object containing the training data.
        """

        features_array, labels_array = self._prepare_training_data()
        features_tensor, labels_tensor = self._convert_to_tensors(features_array=features_array,
                                                                  labels_array=labels_array)

        training_loader: DataLoader = self._create_data_loader(features_tensor=features_tensor,
                                                               labels_tensor=labels_tensor)
        return training_loader

    def get_test_loader(self) -> DataLoader:
        """
        Prepares and returns a DataLoader object for the test dataset.

        :return: DataLoader: A DataLoader object containing the test data.
        """

        features_array, labels_array = self._prepare_testing_data()
        features_tensor, labels_tensor = self._convert_to_tensors(features_array=features_array,
                                                                  labels_array=labels_array)

        testing_loader: DataLoader = self._create_data_loader(features_tensor=features_tensor,
                                                              labels_tensor=labels_tensor)
        return testing_loader

    def _create_data_loader(self, features_tensor: torch.Tensor, labels_tensor: torch.Tensor,
                            batch_size: int = 512) -> DataLoader:
        """
        Creates a DataLoader from feature / label tensors.

        :param: features_tensor (torch.Tensor): features tensor.
        :param: labels_tensor (torch.Tensor): labels tensor.
        :param: batch_size (int): Batch size for DataLoader.

        :returns:DataLoader: A PyTorch DataLoader for the to be used in model training / testing.
        """
        dataset: torch.TensorDataset = TensorDataset(features_tensor, labels_tensor)
        data_loader: torch.DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader

    def _convert_to_tensors(self, features_array: np.ndarray, labels_array: np.ndarray) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Converts feature and label arrays to PyTorch tensors.

       :param: features_array The input array containing feature data.
       :param: labels_array  The input array containing label data.

        :return: tuple[torch.Tensor, torch.Tensor]: A tuple containing the converted feature tensor / label tensors.
        """
        features_tensor: torch.Tensor = torch.tensor(features_array, dtype=torch.float32)
        labels_tensor: torch.Tensor = torch.tensor(labels_array, dtype=torch.float32).unsqueeze(1)
        return features_tensor, labels_tensor

    def _prepare_training_data(self, n_samples: int = 6000) -> tuple[np.array, np.array]:
        """
        Extracts fraud and valid observations, shuffles the data,
        and separates features and labels.

        :param: n_samples (int): Number of samples to extract for each class.
        :returns: Tuple[np.ndarray, np.ndarray]: Features and labels as NumPy arrays.
        """

        dataframe: pd.DataFrame = self.fraud_data_frame

        fraud_observations: pd.DataFrame = dataframe[dataframe[Constants.IS_FRAUD] == 1].head(n_samples)
        valid_observations: pd.DataFrame = dataframe[dataframe[Constants.IS_FRAUD] == 0].head(n_samples)

        combined_observations: pd.DataFrame = pd.concat([fraud_observations, valid_observations]).sample(
            frac=1).reset_index(
            drop=True)

        labels_array: np.array = combined_observations[Constants.IS_FRAUD].values
        features_dataframe: pd.DataFrame = combined_observations.drop(columns=[Constants.IS_FRAUD])

        processed_observations: pd.DataFrame = self._preprocess_data_frame(input_dataframe=features_dataframe)
        features_array: np.array = processed_observations.values

        return features_array, labels_array

    def _prepare_testing_data(self, n_samples: int = 6000) -> tuple[np.array, np.array]:
        """
        Extracts fraud and valid observations, shuffles the data,
        and separates features and labels.

        :param: n_samples (int): Number of samples to extract for each class.
        :returns: Tuple[np.ndarray, np.ndarray]: Features and labels as NumPy arrays.
        """
        max_rows: int = 8000
        dataframe: pd.DataFrame = self.fraud_data_frame

        fraud_observations: pd.DataFrame = dataframe[dataframe[Constants.IS_FRAUD] == 1][n_samples:max_rows]
        valid_observations: pd.DataFrame = dataframe[dataframe[Constants.IS_FRAUD] == 0][n_samples:max_rows]

        combined_observations: pd.DataFrame = pd.concat([fraud_observations, valid_observations]).sample(
            frac=1).reset_index(
            drop=True)

        labels_array: np.array = combined_observations[Constants.IS_FRAUD].values
        features_dataframe: pd.DataFrame = combined_observations.drop(columns=[Constants.IS_FRAUD])

        processed_observations: pd.DataFrame = self._preprocess_data_frame(input_dataframe=features_dataframe)
        features_array: np.array = processed_observations.values

        return features_array, labels_array

    def _preprocess_data_frame(self, input_dataframe: pd.DataFrame) -> pd.DataFrame:
        """

        Takes in a dataframe and preprocesses the values for later model consumption

        :param: input_dataframe: the input dataframe that the method will
        :return: pandas data frame containing the preprocesses data
        """

        column_list: list[str] = self._get_column_list()
        dataframe: pd.DataFrame = input_dataframe[column_list]
        preprocessor: ColumnTransformer = self._get_column_transformer()
        processed_data = preprocessor.fit_transform(dataframe)

        result_column_list: list[str] = self._get_result_column_list()

        return pd.DataFrame(processed_data, columns=result_column_list)

    def _get_result_column_list(self) -> list[str]:
        return ['amount_norm',
                'type_PAYMENT', 'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_CASH_IN',
                'new_balance_origin_normalized', 'new_balance_destination_normalized']

    def _get_column_transformer(self) -> ColumnTransformer:
        """
        Creates a ColumnTransformer objects that converts the values of the dataframe into values that will later become a tensor

        :returns: ColumnTransformer: a column transformer object
        """

        column_transformer: ColumnTransformer = ColumnTransformer(
            transformers=[
                (Constants.AMOUNT_NORMALIZED, StandardScaler(), [Constants.AMOUNT]),
                (Constants.TYPE_ENCODED, OneHotEncoder(), [Constants.TRANSACTION_TYPE]),
                (Constants.NEW_BALANCE_ORIGIN_NORMALIZED, StandardScaler(), [Constants.NEW_TRANSACTION_BALANCE]),
                (Constants.NEW_BALANCE_DESTINATION_NORMALIZED, StandardScaler(), [Constants.NEW_RECIPIENT_BALANCE])
            ],

            # Allows for columns that aren't being preprocessed to be included in the final data set
            remainder='passthrough'
        )
        return column_transformer

    def _get_column_list(self) -> list[str]:
        """
        Creates a list of all column names to be included in the dataframe

        :returns: List[str]: string list of all column names
        """

        return [Constants.TRANSACTION_TYPE,
                Constants.AMOUNT,
                Constants.NEW_TRANSACTION_BALANCE,
                Constants.NEW_RECIPIENT_BALANCE]


class NeuralNetwork(nn.Module):
    """
    A feedforward neural network that contains an input layer, two hidden layers, two drop out layers and an output layer
    """
    _input_size: int = 8
    _hidden_input_size: int = 8
    _output_size: int = 1

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.dropout_layer_1 = nn.Dropout(0.2)
        self.input_layer = nn.Linear(in_features=self._input_size, out_features=self._hidden_input_size)
        self.relu1 = nn.ReLU()
        self.hidden_layer_1 = nn.Linear(in_features=self._hidden_input_size, out_features=self._hidden_input_size)
        self.dropout_layer_2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(in_features=self._hidden_input_size, out_features=self._hidden_input_size)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(in_features=self._hidden_input_size, out_features=self._output_size)

    def forward(self, tensor_obj: Tensor) -> Tensor:
        drop_out_layer_1: Tensor = self.dropout_layer_1(tensor_obj)
        input_layer: Tensor = self.relu1(self.input_layer(drop_out_layer_1))
        hidden_layer_1: Tensor = self.relu2(self.hidden_layer_1(input_layer))
        drop_out_layer_2: Tensor = self.dropout_layer_2(hidden_layer_1)
        hidden_layer_2: Tensor = self.relu3(self.hidden_layer_2(drop_out_layer_2))
        output_layer: Tensor = self.output_layer(hidden_layer_2)
        return output_layer


@dataclass
class Model:
    model_file_path: str

    def __init__(self, fraud_data_frame: pd.DataFrame):
        self.device = self._get_device()
        self.neural_network = NeuralNetwork().to(self.device)
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=0.001)
        self.data_preprocessor = DataPreprocessor(fraud_data_frame=fraud_data_frame)

    def train_neural_network(self, epochs:int) -> list[list[float]]:

        epoch_loss_matrix: list[list[float]] = [[]]
        training_loader: DataLoader = self.data_preprocessor.get_training_loader()

        logger.info("Starting Neural Network Training")
        logger.info("===============================================")

        for epoch in tqdm(range(epochs), "Neural Network Training Progress"):
            running_loss: float = 0.0
            epoch_loss_list = []
            for inputs, labels in training_loader:
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
                epoch_loss_list.append(running_loss)

            epoch_loss_matrix.append(epoch_loss_list)
            logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}')

        logger.info("===============================================")
        logger.info("Completed Neural Network Training")
        logger.info("===============================================")

        return epoch_loss_matrix

    def test_neural_network(self) -> None:

        neural_network_obj: NeuralNetwork = NeuralNetwork()
        neural_network_obj.load_state_dict(
            torch.load(self.model_file_path, map_location=self.device, weights_only=True))
        neural_network_obj.to(self.device)
        neural_network_obj.eval()

        testing_loader: DataLoader = self.data_preprocessor.get_test_loader()

        logger.info("Starting Neural Network Testing")
        logger.info("===============================================")

        total_observations: int = 0
        correctly_predicted_observations: int = 0

        with torch.no_grad():
            for inputs, labels in tqdm(testing_loader, "Neural Network Testing Progress"):
                input_tensor: torch.Tensor = inputs.to(self.device)
                target_tensor: torch.Tensor = labels.to(self.device)

                neural_network_output = neural_network_obj(input_tensor)

                # Apply sigmoid to convert logits to probabilities
                probabilities = torch.sigmoid(neural_network_output)

                # Apply threshold to get binary predictions
                predicted_values = (probabilities >= 0.5).float()

                # Flatten tensors to ensure they are 1D and of the same shape
                predicted_values = predicted_values.view(-1)
                target_tensor = target_tensor.view(-1)

                correctly_predicted_observations += (predicted_values == target_tensor).sum().item()
                total_observations += target_tensor.size(0)

        logger.info("==============================================")
        logger.info(f"Total Observations = {total_observations:,}")
        logger.info(f"Correctly Predicted Observations = {correctly_predicted_observations:,}")
        logger.info(f'Neural Network Accuracy: {(correctly_predicted_observations / total_observations) * 100:.2f}%')
        logger.info("==============================================")

        logger.info("Completed Neural Network Testing")
        logger.info("==============================================")

    def write_results(self, epoch_loss_list: list[list[float]]) -> None:
        """
        Writes the results of neural network training to a specified log file
        :param epoch_loss_list: the results of neural network training represented in a list
        :return: None
        """
        output_directory_path: Path = Path.cwd() / "machine_learning" / "neural_network_execution_results"

        writer = SummaryWriter(log_dir=str(output_directory_path))
        for epoch in range(len(epoch_loss_list)):
            for loss in epoch_loss_list[epoch]:
                writer.add_scalar("Loss/train", loss, epoch)

        writer.close()
        logger.info(f"Saved neural network execution results: {output_directory_path}")

    def save_model_state(self) -> None:
        """
        Saves neural network training results to a specified file
        :return: None
        """
        output_directory_path: Path = Path.cwd() / "machine_learning" / "model_states"
        output_directory_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
        current_date_time: str = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        file_name: str = f"model_{current_date_time}.pth"
        full_output_path: str = str(output_directory_path / file_name)
        torch.save(self.neural_network.state_dict(), full_output_path)
        self.model_file_path = full_output_path
        logger.info(f"Saved neural network state: {full_output_path}")

    def launch_tensor_board(self) -> None:
        """
        Used in to launch TensorBoard, a tool for visualizing machine learning metrics from our neural network output
        """
        logger.info("Launching TensorBoard:")
        output_directory_path: Path = Path.cwd() / "machine_learning" / "neural_network_execution_results"
        os.system("tensorboard --logdir=" + str(output_directory_path))

    def _get_device(self) -> device:
        """
        Assigns the device the model with run on

        :return: the device the machine as chosen to run the model on (can be CUDA, MPS or CPU)
        """
        device_used: torch.device = torch.device(
            Constants.CUDA
            if torch.cuda.is_available()
            else Constants.MPS
            if torch.backends.mps.is_available()
            else Constants.CPU
        )

        device_str: str = str(device_used).upper()
        logger.info(f"Using {device_str} Device")

        return device_used
