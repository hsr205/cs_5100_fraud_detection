import zipfile
from dataclasses import dataclass

import pandas as pd
from logger import Logger

logger: Logger = Logger().get_logger()


@dataclass
class CustomDataLoader:

    def get_data_frame_from_csv_file(self, file_path: str) -> pd.DataFrame:
        data_frame: pd.DataFrame = pd.read_csv(file_path)
        return data_frame

    def get_data_frame_from_zip_file(self, file_path: str, file_name: str) -> pd.DataFrame:
        with zipfile.ZipFile(file_path, 'r') as zipFile:
            file_name_list: list[str] = zipFile.namelist()

            for file in file_name_list:
                logger.info(f"File found in zip: {file}")

            data_frame: pd.DataFrame = self.convert_zip_file_data_to_data_frame(file_name=file_name, zipFile=zipFile)

            return data_frame

    def convert_zip_file_data_to_data_frame(self, file_name: str, zipFile: zipfile.ZipFile) -> pd.DataFrame:
        with zipFile.open(file_name) as csv_file:
            data_frame: pd.DataFrame = pd.read_csv(csv_file)
            return data_frame
