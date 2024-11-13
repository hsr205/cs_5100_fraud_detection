import os
import sys
from dataclasses import dataclass
from datetime import datetime

# adding path to sys for local module imports
sys.path.append(os.path.join(os.getcwd(), "main"))

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
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

logger: Logger = Logger().get_logger()


class DataPreprocessor:
    def __init__(self, fraud_data_frame: pd.DataFrame):
        self.fraud_data_frame = fraud_data_frame

    def get_training_dataloader(self, sample_data_size: int = 6000, batch_size: int = 128) -> DataLoader:
        """
        Combines the x-features / y-labels of the data set into a single DataLoader object

        :param sample_data_size: sets the number of observations to retrieve
        :param batch_size: The size of the inputs to inject into the one at one time
        :return: a DataLoader object that is compatible with the PyTorch framework for model creation
        """
        training_dataframe: pd.DataFrame = self.get_training_dataframe()
        features, labels = self.get_tensors(data_frame=training_dataframe)

        logger.info(f"features.size() = {features.size()}")
        logger.info(f"labels.size() = {labels.size()}")

        dataset: TensorDataset = TensorDataset(features, labels)
        dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def get_testing_dataloader(self, sample_data_size: int = 6000, batch_size: int = 128) -> DataLoader:
        """
        Combines the x-features / y-labels of the data set into a single DataLoader object

        :param sample_data_size: sets the number of observations to retrieve
        :param batch_size: The size of the inputs to inject into the one at one time
        :return: a DataLoader object that is compatible with the PyTorch framework for model creation
        """

        testing_dataframe: pd.DataFrame = self.get_testing_dataframe()
        features, labels = self.get_tensors(data_frame=testing_dataframe)
        dataset: TensorDataset = TensorDataset(features, labels)
        dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def get_training_dataframe(self, train_size: int = 4000) -> pd.DataFrame:
        """
        Generates a balanced training dataset with the specified number of fraud and valid transactions.

        :param train_size: Number of fraud and valid samples to include in the training set (default 3,000 for each).
        :return: DataFrame containing the training data.
        """
        data_frame = self.fraud_data_frame

        # Separate fraud and valid transactions
        fraud_df = data_frame[data_frame[Constants.IS_FRAUD] == 1].sample(frac=1).reset_index(
            drop=True)  # Shuffle fraud data
        valid_df = data_frame[data_frame[Constants.IS_FRAUD] == 0].sample(frac=1).reset_index(
            drop=True)  # Shuffle valid data

        print(f"len(fraud_df) = {len(fraud_df)}")
        print(f"len(valid_df) = {len(valid_df)}")

        # Ensure there are enough samples for training
        fraud_train = fraud_df.head(train_size)
        valid_train = valid_df.head(train_size)

        print(f"len(fraud_train) = {len(fraud_train)}")
        print(f"len(valid_train) = {len(valid_train)}")

        # Combine fraud and valid for training set
        train_data = pd.concat([fraud_train, valid_train]).sample(frac=1).reset_index(
            drop=True)  # Shuffle combined data

        print(f"len(train_data) = {len(train_data)}")

        return train_data

    def get_testing_dataframe(self, train_size: int = 4000) -> pd.DataFrame:
        """
        Generates a balanced testing dataset from the remaining fraud and valid transactions after training data selection.

        :param train_size: Number of fraud and valid samples used in the training set (default 3,000 for each).
        :return: DataFrame containing the testing data.
        """
        data_frame = self.fraud_data_frame

        # Separate fraud and valid transactions
        fraud_df = data_frame[data_frame[Constants.IS_FRAUD] == 1].sample(frac=1).reset_index(
            drop=True)  # Shuffle fraud data
        valid_df = data_frame[data_frame[Constants.IS_FRAUD] == 0].sample(frac=1).reset_index(
            drop=True)  # Shuffle valid data

        # The remaining data for testing
        fraud_test = fraud_df.iloc[train_size:]
        valid_test = valid_df.iloc[train_size:]
        test_data = pd.concat([fraud_test, valid_test]).sample(frac=1).reset_index(drop=True)

        return test_data

    def get_tensors(self, data_frame: pd.DataFrame) -> tuple[Tensor, Tensor]:
        """
        Converts a dataframe into two separate tensors. A tensor of features (x-values) and a tensor of labels (y-values)

        :param: data_frame
        :param: sample_data_size: sets the number of observations to retrieve
        :return: a tuple containing both the feature and label tensors
        """

        labels_data_series: pd.Series = data_frame.query(f"{Constants.IS_FRAUD} == 1")[Constants.IS_FRAUD]

        features_data_frame: pd.DataFrame = data_frame.drop(columns=[Constants.IS_FRAUD])

        processed_features_data_frame: pd.DataFrame = self.preprocess_data_frame(input_dataframe=features_data_frame)

        features_tensor = torch.tensor(processed_features_data_frame.values, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_data_series.values, dtype=torch.float32).unsqueeze(1)

        return features_tensor, labels_tensor

    def preprocess_data_frame(self, input_dataframe: pd.DataFrame) -> pd.DataFrame:
        """

        Takes in a dataframe and preprocesses the values for later model consumption

        :param: input_dataframe: the input dataframe that the method will
        :return: pandas data frame containing the preprocesses data
        """

        # Lists columns of the data set to include
        # Potentially add columns nameOrig and nameDest (account numbers) to future analysis
        column_list: list[str] = [Constants.TRANSACTION_TYPE,
                                  Constants.AMOUNT,
                                  Constants.NEW_TRANSACTION_BALANCE,
                                  Constants.NEW_RECIPIENT_BALANCE]

        dataframe: pd.DataFrame = input_dataframe[column_list]

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

        processed_data = preprocessor.fit_transform(dataframe)

        # Convert back to DataFrame with column names for easier viewing
        result_column_list: list[str] = ['amount_norm',
                                         'type_PAYMENT', 'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_CASH_IN',
                                         'new_balance_origin_normalized', 'new_balance_destination_normalized']

        return pd.DataFrame(processed_data, columns=result_column_list)


# TODO: Look at the issue of overfitting, add F1-Score
#       Also split data into training and test sets (70% - 30%)
class NeuralNetwork(nn.Module):
    """
    A feedforward neural network that contains an input layer, hidden layer and an output layer
    """
    input_size: int = 8
    hidden_input_size: int = 8
    hidden_output_size: int = 5

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # dropout layer, will randomly zero an element of the input tensor with probability 0.2
        self.dropout = nn.Dropout(0.2)
        # Specifies the input_size (default = 8) and the number of nodes in the output layer (8)
        self.input_layer = nn.Linear(self.input_size, self.hidden_input_size)
        # Indicates that after the input layer is completed the ReLU (rectified linear unit)
        # activation function is being called on the first hidden layer's nodes
        self.relu1 = nn.ReLU()
        # Second hidden layer indicating that the number of input nodes is 10 and the number of output nodes is 5
        self.hidden_layer = nn.Linear(self.hidden_input_size, self.hidden_output_size)
        # a second dropout layer, will randomly zero a hidden neuron probability 0.5
        self.dropout = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()  # ReLU (rectified linear unit) used again on the hidden layer
        self.output_layer = nn.Linear(self.hidden_output_size, 1)  # Output layer consisting of a single node

    def forward(self, tensor_obj: Tensor) -> Tensor:
        drop_out_layer1: Tensor = self.dropout(tensor_obj)
        input_layer: Tensor = self.relu1(self.input_layer(drop_out_layer1))
        hidden_layer: Tensor = self.relu2(self.hidden_layer(input_layer))
        drop_out_layer2: Tensor = self.dropout(hidden_layer)
        output_layer: Tensor = self.output_layer(drop_out_layer2)
        return output_layer


@dataclass
class Model:
    model_file_path: str

    def __init__(self, fraud_data_frame: pd.DataFrame):
        self.device = self.get_device()
        self.neural_network = NeuralNetwork().to(self.device)
        self.criterion = BCELoss()
        self.optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=0.001)
        self.fraud_data_frame = fraud_data_frame
        self.data_preprocessor = DataPreprocessor(fraud_data_frame=self.fraud_data_frame)

    def train_neural_network(self, batch_size: int = 128) -> list[list[float]]:

        num_epochs = 10
        epoch_loss_matrix: list[list[float]] = [[]]
        training_loader: DataLoader = self.data_preprocessor.get_training_dataloader()

        num_observations: int = len(training_loader)

        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Number of Data Set Observations Used: {num_observations:,}")
        logger.info("===============================================")

        logger.info("Starting Neural Network Training")
        logger.info("===============================================")

        for epoch in tqdm(range(num_epochs), "Neural Network Training Progress"):
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
            epoch_loss: float = running_loss / (num_observations / batch_size)
            epoch_loss_matrix.append(epoch_loss_list)
            logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        logger.info("Completed Neural Network Training")
        logger.info("===============================================")

        return epoch_loss_matrix

    # TODO: This method is in progress - Henry or Pablo to complete
    def test_neural_network(self) -> float:

        result_float: float = 0.0

        neural_network_obj: NeuralNetwork = NeuralNetwork()
        neural_network_obj.load_state_dict(torch.load(self.model_file_path))
        neural_network_obj.eval()

        test_loader: DataLoader = self.data_preprocessor.get_testing_dataloader(sample_data_size=2000)

        logger.info("Starting Neural Network Testing")
        logger.info("===============================================")

        correctly_predicted_observations: int = 0
        total_observations: int = 0

        with torch.no_grad():
            for data in tqdm(test_loader, "Neural Network Testing Progress"):
                tensor, target_tensor = data

                neural_network_output = neural_network_obj(tensor)

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
        logger.info(f'Neural Network Accuracy: {correctly_predicted_observations / total_observations}')
        logger.info("==============================================")

        logger.info("Completed Neural Network Testing")
        logger.info("==============================================")

        return result_float

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

    @staticmethod
    def launch_tensor_board() -> None:
        """
        Used in to launch TensorBoard, a tool for visualizing machine learning metrics from our neural network output
        """
        logger.info("Launching TensorBoard:")
        output_directory_path: Path = Path.cwd() / "machine_learning" / "neural_network_execution_results"
        os.system("tensorboard --logdir=" + str(output_directory_path))

    @staticmethod
    def get_device() -> device:
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
