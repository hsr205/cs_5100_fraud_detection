from typing import Any
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from logger import Logger
from matplotlib import ticker
from static.constants import Constants

logger: Logger = Logger().get_logger()

sns.set(style="whitegrid")


class KMeansLearning:

    def __init__(self, data_loader: Any, file_path: str, file_name: str, k: int):
        """ initializes KMeansLearning object

        Params
        ------
        data_loader : any
            a data_loader object to help process data
        
        file_path : str
            path to data
        
        file_name : str
            name of data file

        k : int
            "k" - number of centroids to be used in analysis

        Returns
        -------
        None
        """

        self.data_loader = data_loader
        self.file_path = file_path
        self.file_name = file_name
        self.data_frame = data_loader.get_data_frame_from_zip_file(file_path=file_path, file_name=file_name)
        self.k_value = k

    def display_sample_of_data_points(self, num_data_points: int, x_axis_str: str, y_axis_str: str) -> None:

        """ Displays a sample of the data provided in initialization 
        Params
        ------
        num_data_points : int
            number of data points to display

        x_axis_str : str
            text to display on the x-axis

        y_axis_str : str
            text to display on the y-axis

        Returns
        -------
        None
        """

        # creating a dataframe containing number of samples
        sample_data_frame: pd.DataFrame = self.get_sample_data_from_data_frame(num_data_points=num_data_points)

        # creating x, y data series
        x_axis: pd.Series = sample_data_frame[x_axis_str]
        y_axis: pd.Series = sample_data_frame[y_axis_str]

        self.add_text_labels(x_axis_str=x_axis_str, y_axis_str=y_axis_str)

        self.convert_values_to_decimal_format()

        self.add_color_indicating_fraud_data_points(sample_data_frame=sample_data_frame, x_axis=x_axis, y_axis=y_axis)

        plt.grid(True)

        plt.show()

    def get_sample_data_from_data_frame(self, num_data_points: int) -> pd.DataFrame:

        # returns the first "num_data_points" data points

        # should we use pd.sample?
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

    def forgy_centroids(self):
        """ a function that generates initial centroids with Forgy method
        Params
        ------
        None

        Returns
        -------
        k_centroids : dataframe
            a dataframe of length k representing random samples of the 
            original dataset
        """

        return self.data_frame.sample(n=self.k_value)
    
    def euclidean_distance(self, a, b):
        """ returns the Euclidean distance between two points """

        total = 0
        for element in range(len(a)):
            diff = a[element] - b[element]
            total += (diff * diff)

        return math.sqrt(total)
    
    def cluster_mean(self, cluster):
        """ Returns the mean value of the given cluster """

        return pd.Dataframe(cluster).mean(axis=1)
    
    def execute_clustering(self, init_type="forgy"):

        # initializing centroids
        match init_type:

            case "forgy":
                centroids = self.forgy_centroids()
            case _:
                print("Invalid initialization, defaulting to Forgy")
                centroids = self.forgy_centroids()

        # initializing cluster list
        # clusters = [[] for _ in range(self.k_value)]

        converged = False

        while not converged:

            clusters = [[] for _ in range(self.k_value)]

            # assign points to clusters
            for point in self.data_frame:
                distances = [self.euclidean_distance(point, centroid) for centroid in centroids]

                cluster_classification = centroids.index(min(distances))

                clusters[cluster_classification].append([point])

                new_centroids = [self.cluster_mean(cluster) for cluster in clusters]

                converged = (new_centroids == centroids)

                centroids = new_centroids

                if converged:
                    return clusters

