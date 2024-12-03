# Anomaly Detection Documentation

<p>
This documentation goes over the contents of the anomaly_detection.py file to identify fraudulant transactions based on the given dataset.

</p>

---

## Table of Contents

- [Classes](#Classes)
    - [IFProcessor](#IFProcessor)
    - [IFModel](#IFModel)
    - [Execution_Code](#Execution_Code)
    - [Sample_Execution_Output](#Sample_Execution_Output)
    - [Results](#Results)

## Classes

---

## IFProcessor

This is a subclass of `DataPreprocessor` from the file `neural_network.py`. It overrides two methods (`_get_column_list()` and `_get_result_column_list()`) to include the column `isFraud` to be used in calculating the accuracy and f1 score of the model

---

## IFModel

This class represents an Isolation Forest model to be used for anomaly detection.

`__init__()` - Initializes the model with a DataFrame which will be utilized and transformed, as well as a random seed to be used in creating the dimensional splits of data. 

`preprocess()` - Normalizes the data set with the `DataPreprocessor` class in `neural_network.py`.

`detect(num_observations)` - Identifies fraudulant transactions based on specified features.

`view_classification(df)` - Visualizes the amount of true/false positive/negative classifications using a heatmap

`vis_2d()` - Visualizes the IsolationForest in two dimensions, `amount_norm` and `new_balance_origin_normalized`

`vis_3d()` - Visualizes the IsolationForest in three dimensions, `amount_norm`, `new_balance_origin_normalized`, and `new_balance_destination_normalized`

---

## Execution_Code

<p>
Run the following code in the main.py file in order to execute the anomaly detection using an IsolationForest:
</p>

```python
def main() -> int:
    data_loader: CustomDataLoader = CustomDataLoader()
    file_path_to_data: str = str(
        Path.cwd() / "data" / "fraud_detection_data_set" / "synthetic_financial_datasets_log.zip")
    file_name: str = "synthetic_financial_datasets_log.csv"

    fraud_data: FraudDataset = FraudDataset(data_loader=data_loader,
                                            file_path=file_path_to_data,
                                            file_name=file_name,
                                            transform_to_tensor=None,
                                            target_transform=None)

    fraud_data_frame: pd.DataFrame = fraud_data.data_loader.get_data_frame_from_zip_file(file_path=file_path_to_data,
                                                                                         file_name=file_name)

    if_model: IFModel = IFModel(fraud_data_frame=fraud_data_frame)
    if_model.detect(16000) # change number of observations
```

---

## Sample_Execution_Output

```
2024-11-27 05:27:18 PM - INFO - File found in zip: data_dictionary_20241009.xlsx
2024-11-27 05:27:18 PM - INFO - File found in zip: __MACOSX/._data_dictionary_20241009.xlsx
2024-11-27 05:27:18 PM - INFO - File found in zip: Synthetic_Financial_datasets_log.csv
2024-11-27 05:27:18 PM - INFO - File found in zip: __MACOSX/._Synthetic_Financial_datasets_log.csv
2024-11-27 05:27:22 PM - INFO - File found in zip: data_dictionary_20241009.xlsx
2024-11-27 05:27:22 PM - INFO - File found in zip: __MACOSX/._data_dictionary_20241009.xlsx
2024-11-27 05:27:22 PM - INFO - File found in zip: Synthetic_Financial_datasets_log.csv
2024-11-27 05:27:22 PM - INFO - File found in zip: __MACOSX/._Synthetic_Financial_datasets_log.csv
Processing... 16000 observations
Accuracy: 0.5271875
F1 Score: 0.4523275175559256
```

---

Resources used in the creation of this model and methods

* https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.IsolationForest.html

* https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html