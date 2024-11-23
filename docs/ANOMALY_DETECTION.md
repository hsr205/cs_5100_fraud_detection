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

`detect()` - Identifies fraudulant transactions based on specified features .

`vis_2d()` - Visualizes the IsolationForest in two dimensions, `amount_norm` and `new_balance_origin_normalized`

`vis_3d()` - Visualizes the IsolationForest in three dimensions, `amount_norm`, `new_balance_origin_normalized`, and `new_balance_destination_normalized`



---

Resources used in the creation of this model and methods

* https://chatgpt.com

* https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.IsolationForest.html

* https://medium.com/@corymaklin/isolation-forest-799fceacdda4

* https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html