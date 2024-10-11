from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from logger import Logger
from matplotlib import ticker
from static.constants import Constants

logger: Logger = Logger().get_logger()

sns.set(style="whitegrid")


class KMeansLearning:

    def __init__(self, data_loader: Any, file_path: str, file_name: str):
        self.data_loader = data_loader
        self.file_path = file_path
        self.file_name = file_name
        self.data_frame = data_loader.get_data_frame_from_zip_file(file_path=file_path, file_name=file_name)

    def display_sample_of_data_points(self, num_data_points: int, x_axis_str: str, y_axis_str: str) -> None:
        sample_data_frame: pd.DataFrame = self.get_sample_data_from_data_frame(num_data_points=num_data_points)

        x_axis: pd.Series = sample_data_frame[x_axis_str]
        y_axis: pd.Series = sample_data_frame[y_axis_str]

        self.add_text_labels(x_axis_str=x_axis_str, y_axis_str=y_axis_str)

        self.convert_values_to_decimal_format()

        self.add_color_indicating_fraud_data_points(sample_data_frame=sample_data_frame, x_axis=x_axis, y_axis=y_axis)

        plt.grid(True)

        plt.show()

    def get_sample_data_from_data_frame(self, num_data_points: int) -> pd.DataFrame:
        return self.data_frame.iloc[:num_data_points]

    def add_color_indicating_fraud_data_points(self, sample_data_frame: pd.DataFrame, x_axis: pd.Series,
                                               y_axis: pd.Series) -> None:
        fraud_status: pd.Series = sample_data_frame[Constants.IS_FRAUD]
        plt.scatter(x_axis, y_axis, c=fraud_status, cmap='coolwarm', alpha=0.75)
        cbar = plt.colorbar()
        cbar.set_label('Fraud Status')

    def add_text_labels(self, x_axis_str: str, y_axis_str: str) -> None:
        plt.title("Sample Data Display")
        plt.xlabel(x_axis_str)
        plt.ylabel(y_axis_str)

    def convert_values_to_decimal_format(self) -> None:
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
