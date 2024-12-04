import pandas as pd

from machine_learning.anomaly_detection import IFModel
from machine_learning.k_means_learning import KMeansLearning
from machine_learning.neural_network import Model, Accuracy


class Execution:
    def execute_isolation_forest(self, fraud_data_frame: pd.DataFrame) -> None:
        isolation_forest: IFModel = IFModel(fraud_data_frame=fraud_data_frame)
        isolation_forest.detect(16000)

    def execute_k_means(self, fraud_data_frame: pd.DataFrame) -> None:
        k_means: KMeansLearning = KMeansLearning(data_frame=fraud_data_frame, k=3)
        k_means.execute_clustering(3)
        k_means.visualize_clusters(6, 7)

    def execute_neural_network(self, fraud_data_frame: pd.DataFrame) -> None:
        model: Model = Model(fraud_data_frame=fraud_data_frame)
        epoch_loss_list: list[list[float]] = model.train_neural_network(epochs=20)
        model.write_results(epoch_loss_list=epoch_loss_list)
        model.save_model_state()
        accuracy_results: Accuracy = model.test_neural_network()
        model.display_testing_results(accuracy_obj=accuracy_results)
        model.launch_tensor_board()
