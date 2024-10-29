import pandas as pd

from data.custom_data_loader import CustomDataLoader
from data.fraud_data import FraudDataset
from logger import Logger
from machine_learning.neural_network import DataPreprocessor

logger: Logger = Logger().get_logger()

pd.set_option('display.max_columns', None)


def main() -> int:
    data_loader: CustomDataLoader = CustomDataLoader()

    # Include local file path to data
    file_path: str = ""
    file_name: str = "synthetic_financial_datasets_log.csv"

    fraud_data: FraudDataset = FraudDataset(data_loader=data_loader,
                                            file_path=file_path,
                                            file_name=file_name,
                                            transform_to_tensor=None,
                                            target_transform=None)

    fraud_data_frame: pd.DataFrame = fraud_data.data_loader.get_data_frame_from_zip_file(file_path=file_path,
                                                                                         file_name=file_name)

    data_preprocessor: DataPreprocessor = DataPreprocessor(fraud_data_frame=fraud_data_frame)

    print(f"{data_preprocessor.preprocess_data_frame().head()}")

    return 0


if __name__ == "__main__":
    main()
