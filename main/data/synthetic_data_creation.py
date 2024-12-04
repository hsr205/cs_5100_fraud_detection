import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import logging
from pathlib import Path
from static.constants import Constants

# Assuming Constants is a module or class with these constants defined.
# from your_project import Constants

# Initialize logger if not already done
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class tSNE:
    def __init__(self, fraud_data_frame: pd.DataFrame, samples: int):
        logger.info("Initializing t-SNE object .....")

        observations: pd.DataFrame = fraud_data_frame.sample(samples)

        self.labels_array: np.array = observations[Constants.IS_FRAUD].values

        features_dataframe = observations.drop(columns=[Constants.IS_FRAUD])

        column_list = [
            Constants.TRANSACTION_TYPE,
            Constants.AMOUNT,
            Constants.NEW_TRANSACTION_BALANCE,
            Constants.NEW_RECIPIENT_BALANCE
        ]

        features_dataframe = features_dataframe[column_list]

        column_transformer = ColumnTransformer(
            transformers=[
                (Constants.AMOUNT_NORMALIZED, StandardScaler(), [Constants.AMOUNT]),
                (Constants.TYPE_ENCODED, OneHotEncoder(), [Constants.TRANSACTION_TYPE]),
                (Constants.NEW_BALANCE_ORIGIN_NORMALIZED, StandardScaler(), [Constants.NEW_TRANSACTION_BALANCE]),
                (Constants.NEW_BALANCE_DESTINATION_NORMALIZED, StandardScaler(), [Constants.NEW_RECIPIENT_BALANCE])
            ],
            remainder='passthrough'
        )

        processed_data = column_transformer.fit_transform(features_dataframe)

        result_column_list = [
            'amount_norm', 'type_PAYMENT', 'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_CASH_IN',
            'new_balance_origin_normalized', 'new_balance_destination_normalized'
        ]

        features_dataframe = pd.DataFrame(processed_data, columns=result_column_list)

        self.features_array: np.array = features_dataframe.values

    def visualize(self):
        logger.info("Computing 2D t-SNE ......")

        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=50, verbose=1)
        data_tsne_2d = tsne_2d.fit_transform(self.features_array)

        logger.info("Plotting 2D t-SNE ......")
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121)

        scatter2d = ax1.scatter([], [], c=[], cmap='viridis', s=50)
        ax1.set_title('2D t-SNE')

        num_points = data_tsne_2d.shape[0]
        batch_size = int(num_points/25)
        pbar2d = tqdm(total=num_points, desc="2D Plotting")

        for i in range(0, num_points, batch_size):
            end_idx = min(i + batch_size, num_points)
            scatter2d = ax1.scatter(data_tsne_2d[i:end_idx, 0], data_tsne_2d[i:end_idx, 1],
                                    c=self.labels_array[i:end_idx], cmap='viridis', s=50)
            pbar2d.update(batch_size)

        pbar2d.close()

        logger.info("Computing 3D t-SNE ......")

        tsne_3d = TSNE(n_components=3, random_state=42, perplexity=75, verbose=1)
        data_tsne_3d = tsne_3d.fit_transform(self.features_array)

        logger.info("Plotting 3D t-SNE ......")
        ax2 = fig.add_subplot(122, projection='3d')
        scatter3d = ax2.scatter([], [], [], c=[], cmap='viridis', s=50)

        num_points = data_tsne_3d.shape[0]
        pbar3d = tqdm(total=num_points, desc="3D Plotting")

        for i in range(0, num_points, batch_size):
            end_idx = min(i + batch_size, num_points)
            scatter3d = ax2.scatter(data_tsne_3d[i:end_idx, 0], data_tsne_3d[i:end_idx, 1],
                                    data_tsne_3d[i:end_idx, 2], c=self.labels_array[i:end_idx], cmap='viridis', s=50)
            pbar3d.update(batch_size)

        pbar3d.close()

        ax2.set_title('3D t-SNE')
        fig.colorbar(scatter3d, ax=ax2, label='Fraudulence')

        # Save and display the plot
        image_path = Path.cwd() / "data" / "t-SNE_results"
        image_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        plt.savefig(str(image_path / "t-SNE_result.png"), dpi=300)

        logger.info("Finished plotting!")
