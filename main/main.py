from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from data.custom_data_loader import CustomDataLoader
from data.fraud_data import FraudDataset
from data.synthetic_data_creation import tSNE
from data_preprocessing.data_preprocessing import DataTransformer
from machine_learning.anomaly_detection import IFModel
from machine_learning.k_means_learning import KMeansLearning
from machine_learning.neural_network import Accuracy
from machine_learning.neural_network import Model
from machine_learning.random_forest import RandomForest


def main() -> int:
    data_loader: CustomDataLoader = CustomDataLoader()
    file_path_to_data: str = str(
        Path.cwd() / "data" / "fraud_detection_data_set" / "synthetic_financial_datasets_log.zip")
    file_name: str = "synthetic_financial_datasets_log.csv"

    fraud_data: FraudDataset = FraudDataset(data_loader=data_loader,
                                            file_path=file_path_to_data,
                                            file_name=file_name,
                                            transform_to_tensor=None,
                                            target_transform=None)

    fraud_data_frame: pd.DataFrame = fraud_data.data_loader.get_data_frame_from_zip_file(file_path=file_path_to_data,
                                                                                         file_name=file_name)
    execute_random_forest(fraud_data_frame=fraud_data_frame)
    execute_k_means(fraud_data_frame=fraud_data_frame)
    execute_tsne(fraud_data_frame=fraud_data_frame)
    execute_isolation_forest(fraud_data_frame=fraud_data_frame)
    execute_neural_network(fraud_data_frame=fraud_data_frame)
    return 0


def execute_tsne(fraud_data_frame: pd.DataFrame) -> None:
    tsne = tSNE(fraud_data_frame, 8_000)
    tsne.visualize()


def execute_k_means(fraud_data_frame: pd.DataFrame) -> None:
    k_means: KMeansLearning = KMeansLearning(data_frame=fraud_data_frame, k=3)

    k_means.execute_clustering(3)
    k_means.visualize_clusters(6, 7)


def execute_isolation_forest(fraud_data_frame: pd.DataFrame) -> None:
    isolation_forest: IFModel = IFModel(fraud_data_frame=fraud_data_frame)
    isolation_forest.detect(16000)
    execute_neural_network(fraud_data_frame=fraud_data_frame)


def execute_neural_network(fraud_data_frame: pd.DataFrame) -> None:
    model: Model = Model(fraud_data_frame=fraud_data_frame)
    epoch_loss_list: list[list[float]] = model.train_neural_network(epochs=20)
    model.write_results(epoch_loss_list=epoch_loss_list)
    model.save_model_state()
    accuracy_results: Accuracy = model.test_neural_network()
    model.display_testing_results(accuracy_obj=accuracy_results)
    model.launch_tensor_board()


def execute_random_forest(fraud_data_frame: pd.DataFrame) -> None:
    transformed_file_path = "data_preprocessing/transformed_data.csv"

    print("Preprocessing data...")
    data_transformer = DataTransformer(fraud_data_frame)
    data_transformer.preprocess_data_frame()  # Preprocess and save transformed data

    # Step 2: Load the transformed data
    print("Loading transformed data...")
    transformed_data = pd.read_csv(transformed_file_path)

    # Step 3: Initialize and train the RandomForest model
    print("Fitting RandomForest model...")
    random_forest_model = RandomForest(num_trees=7, max_depth=9, min_samples_split=7)
    X_test, y_test = random_forest_model.fit(
        transformed_data, target_column='isFraud', fraud_samples=6000, non_fraud_samples=6000
    )

    # Step 4: Predict on the test set
    print("Making predictions on the test set...")
    predictions = random_forest_model.predict(X_test)

    # Step 5: Calculate accuracy
    accuracy = accuracy_score(y_test, predictions) * 100
    print(f"Test Set Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
