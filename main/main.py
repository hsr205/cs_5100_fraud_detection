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
                                            #  transform_to_tensor=None,
                                            #  target_transform=None, 
                                             k=3)

    k_means.display_sample_of_data_points(num_data_points=1000, x_axis_str=Constants.BEFORE_TRANSACTION_BALANCE, y_axis_str=Constants.NEW_TRANSACTION_BALANCE)

    k_means.execute_clustering()

    return 0


if __name__ == "__main__":
    main()
