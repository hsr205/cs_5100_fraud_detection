from typing import Any

import pandas as pd


class KMeansLearning:

    def __init__(self, data_loader: Any, file_path: str, file_name: str, transform_to_tensor=None,
                 target_transform=None):
        self.data_loader = data_loader
        self.file_path = file_path
        self.file_name = file_name
        self.data_frame = data_loader.get_data_frame_from_zip_file(file_path=file_path, file_name=file_name)
        self.transform_to_tensor = transform_to_tensor
        self.target_transform = target_transform

    def display_sample_of_data_points(self, num_data_points: int) -> None:
        sample_data_frame: pd.DataFrame = self.data_frame.iloc[:num_data_points]

        print(sample_data_frame)

        # x_axis: pd.Series = sample_data_frame['']
        # y_axis: pd.Series = sample_data_frame['']
        #
        # plt.plot(x_axis, y_axis)
        #
        # plt.title("Sample Data Display")
        # plt.xlabel("X axis")
        # plt.ylabel("Y axis")
        #
        # plt.scatter()
        # plt.grid(True)
        #
        # plt.show()
