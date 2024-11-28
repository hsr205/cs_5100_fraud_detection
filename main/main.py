from pathlib import Path

import pandas as pd

from data.custom_data_loader import CustomDataLoader
from data.fraud_data import FraudDataset
from machine_learning.neural_network import Model
from machine_learning.neural_network import DataPreprocessor as dp
from machine_learning.anomaly_detection import IFModel


def main() -> int:
    data_loader: CustomDataLoader = CustomDataLoader()
    file_path_to_data: str = str(
        Path.cwd() / "main" / "data" / "fraud_detection_data_set" / "synthetic_financial_datasets_log.zip")
    file_name: str = "Synthetic_Financial_datasets_log.csv"

    fraud_data: FraudDataset = FraudDataset(data_loader=data_loader,
                                            file_path=file_path_to_data,
                                            file_name=file_name,
                                            transform_to_tensor=None,
                                            target_transform=None)

    fraud_data_frame: pd.DataFrame = fraud_data.data_loader.get_data_frame_from_zip_file(file_path=file_path_to_data,
                                                                                         file_name=file_name)


    # 
    if_model: IFModel = IFModel(fraud_data_frame=fraud_data_frame)
    if_model.detect(32000) # change number of observations

    return 0


if __name__ == "__main__":
    main()
