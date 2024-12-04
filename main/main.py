from pathlib import Path

import pandas as pd

from data.custom_data_loader import CustomDataLoader
from data.fraud_data import FraudDataset
from main.execution import Execution


def main() -> int:
    execution_object: Execution = Execution()
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

    execution_object.execute_isolation_forest(fraud_data_frame=fraud_data_frame)
    execution_object.execute_k_means(fraud_data_frame=fraud_data_frame)
    execution_object.execute_neural_network(fraud_data_frame=fraud_data_frame)

    return 0
