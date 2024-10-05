import pandas as pd
from logger import Logger
from data_loader import DataLoader

logger: Logger = Logger().get_logger()

pd.set_option('display.max_columns', None)


def main() -> int:
    data_loader: DataLoader = DataLoader()

    file_path: str = ""
    file_name: str = "fraudulent_ecommerce_transaction_data.csv"

    fraud_data_frame: pd.DataFrame = data_loader.get_data_frame_from_zip_file(file_path=file_path, file_name=file_name)

    logger.info(f"{fraud_data_frame.head()}")

    return 0


if __name__ == "__main__":
    main()
