import pandas as pd

from data.custom_data_loader import CustomDataLoader
from logger import Logger
from machine_learning.k_means_learning import KMeansLearning
from static.constants import Constants

logger: Logger = Logger().get_logger()

pd.set_option('display.max_columns', None)


def main() -> int:
    data_loader: CustomDataLoader = CustomDataLoader()

    file_path: str = ""
    file_name: str = "synthetic_financial_datasets_log.csv"

    k_means: KMeansLearning = KMeansLearning(data_loader=data_loader,
                                             file_path=file_path,
                                             file_name=file_name,
                                             transform_to_tensor=None,
                                             target_transform=None)

    k_means.display_sample_of_data_points(num_data_points=1000, x_axis_str=Constants.BEFORE_TRANSACTION_BALANCE, y_axis_str=Constants.NEW_TRANSACTION_BALANCE)

    # fraud_data: FraudDataset = FraudDataset(data_loader=data_loader,
    #                                         file_path=file_path,
    #                                         file_name=file_name,
    #                                         transform_to_tensor=None,
    #                                         target_transform=None)
    #
    # fraud_data_frame: pd.DataFrame = fraud_data.data_loader.get_data_frame_from_zip_file(file_path=file_path,
    #                                                                                      file_name=file_name)

    # logger.info(f"Unique Name Original Account Numbers: {len(fraud_data_frame['nameOrig'].unique())}") # 6,353,307
    # logger.info(f"Unique Name Destination Account Numbers: {len(fraud_data_frame['nameDest'].unique())}") # 2,722,362

    # logger.info(f"Num of rows indicating fraud: {fraud_data_frame[fraud_data_frame['isFraud'] == 1].shape[0]}") # 8,213
    # logger.info(f"Num of rows NOT indicating fraud: {fraud_data_frame[fraud_data_frame['isFraud'] == 0].shape[0]}") # 6,354,407

    # logger.info(f"{fraud_data_frame.head()}")

    return 0


if __name__ == "__main__":
    main()
