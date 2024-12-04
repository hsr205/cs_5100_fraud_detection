import math
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from sklearn.compose import ColumnTransformer

from .neural_network import DataPreprocessor as dp

sns.set(style="whitegrid")


def data_processing(dataframe):
    """ brief data preprocessing """

    dataframe = dataframe.sort_values(by='isFraud', ascending=False)

    processor = dp(dataframe)

    preprocessor: ColumnTransformer = processor._get_column_transformer()
    processed_df = pd.DataFrame(preprocessor.fit_transform(dataframe)).head(16000)

    # dropping identifying columns
    processed_df.drop([8, 9, 10, 11, 12, 14], axis=1, inplace=True)

    # dropping columns not used in neural network

    return processed_df


class KMeansLearning:

    def __init__(self, data_frame: Any, k: int):
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
        print("Starting K-Means")
        print("===============================================")
        self.data_frame = data_processing(data_frame)

        self.fraud = self.data_frame[13]

        self.data_frame.drop(13, axis=1, inplace=True)

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

        # plt.show()

    def get_sample_data_from_data_frame(self, num_data_points: int) -> pd.DataFrame:

        # returns the first "num_data_points" data points

        # should we use pd.sample?
        return self.data_frame.iloc[:num_data_points]

    def add_color_indicating_fraud_data_points(self, sample_data_frame: pd.DataFrame, x_axis: pd.Series,
                                               y_axis: pd.Series) -> None:
        fraud_status: pd.Series = sample_data_frame["isFraud"] == 1
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

    def forgy_centroids(self, k):
        """ a function that generates initial centroids with Forgy method
        Params
        ------
        k : int
            number of centroids to return

        Returns
        -------
        k_centroids : dataframe
            a dataframe of length k representing random samples of the
            original dataset
        """

        return self.data_frame.sample(n=k)

    def euclidean_distance(self, a, b):
        """ returns the Euclidean distance between two points """

        total = 0
        diff = a - b

        for element in diff:
            total += element * element

        return math.sqrt(total)

    def cluster_mean(self, cluster):
        """ Returns the mean value of the given cluster """

        return pd.DataFrame(cluster).mean(axis=0)

    def execute_clustering(self, k, init_type="forgy"):

        self.data_frame = self.data_frame.drop(self.data_frame.columns[0], axis=1)

        # initializing centroids
        match init_type:

            case "forgy":
                centroids = self.forgy_centroids(k)
            case _:
                print("Invalid initialization, defaulting to Forgy")
                centroids = self.forgy_centroids(k)

        # initializing cluster list
        # clusters = [[] for _ in range(self.k_value)]

        converged = False

        count = 0

        while not converged:

            clusters = [[] for _ in range(k)]

            # assign points to clusters
            for index, row in self.data_frame.iterrows():

                distances = []
                for index, centroid in centroids.iterrows():
                    distances.append(self.euclidean_distance(row, centroid))

                cluster_classification = distances.index(min(distances))

                clusters[cluster_classification].append(row)

            new_centroids = pd.DataFrame([self.cluster_mean(cluster) for cluster in clusters])

            # new_centroids.index = centroids.reset_index().index
            # new_centroids.columns = centroids.columns

            diff = (new_centroids - centroids).abs()

            converged = (diff <= 1e-5).all().all()

            centroids = new_centroids

            print(f"{count + 1} iteration completed")
            count += 1

            if converged:
                self.clusters = clusters

    def visualize_clusters(self, x_axis, y_axis):

        if len(self.clusters) == 0:
            print("Run clustering algorithm before visualizing clusters!")
            return

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        count = 0

        num_fraud = len(self.fraud[self.fraud == 1])

        for clust in self.clusters:
            df = pd.DataFrame(clust)
            indices = df.index
            df["fraud"] = self.fraud.iloc[indices]

            fraudulent_transactions = df[df["fraud"] == 1]

            x_normal = df[x_axis]
            y_normal = df[y_axis]
            x_fraud = fraudulent_transactions[x_axis]
            y_fraud = fraudulent_transactions[y_axis]
            ax.scatter(x_normal, y_normal, label="Cluster " + (str(count + 1)), alpha=0.9)
            ax.scatter(x_fraud, y_fraud, marker="o", edgecolor="red", alpha=0.1)
            count += 1

            fraud_trans = len(fraudulent_transactions)

            print(
                f"Proportion of all fraudulent transactions in cluster {count}: {round((fraud_trans / num_fraud) * 100, 2):,%}")

        print("===============================================")
        print("Ending K-Means")
        print("===============================================")

        plt.title("Fraudulent Data, Clustered")
        plt.xlabel("New Balance Origin, Normalized")
        plt.ylabel("New Balance Destination, Normalized")
        plt.legend()
