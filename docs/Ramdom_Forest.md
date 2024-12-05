# Random Forest Documentation

This documentation explains the functionality of the `random_forest.py` file, which implements a Random Forest Classifier to detect fraudulent transactions in a dataset. It consists of two main classes: `DecisionTree` and `RandomForest`.

---

## Table of Contents

- [Classes](#Classes)
    - [DecisionTree](#DecisionTree)
    - [RandomForest](#RandomForest)
- [Execution Code](#Execution_Code)
- [Sample Execution Output](#Sample_Execution_Output)

---

## Classes

### DecisionTree

A simple implementation of a binary classification Decision Tree.

#### Methods:

- **`__init__(max_depth=None, min_samples_split=2)`**
  - Initializes the tree with a specified depth and minimum samples for splitting.

- **`fit(X, y, depth=0)`**
  - Recursively fits the Decision Tree to the dataset by minimizing Gini impurity.

- **`predict(X)`**
  - Predicts labels for the provided dataset.

- **`_best_split(X, y)`**
  - Finds the best feature and threshold to split the dataset.

- **`_split(X_column, split_threshold)`**
  - Splits the data based on the given threshold.

---

### RandomForest

An ensemble learning method combining multiple `DecisionTree` models for robust classification.

#### Methods:

- **`__init__(num_trees=7, max_depth=9, min_samples_split=7)`**
  - Initializes the Random Forest with the specified number of trees, depth, and minimum samples for splitting.

- **`fit(data, target_column='isFraud', fraud_samples=6000, non_fraud_samples=6000)`**
  - Trains the Random Forest using balanced subsets of fraudulent and non-fraudulent transactions.

- **`predict(X)`**
  - Aggregates predictions from all trees using majority voting.

- **`_bootstrap_sample(X, y)`**
  - Creates bootstrapped samples of data for training individual trees.

- **`_create_training_testing_sets(data, target_column, fraud_samples, non_fraud_samples)`**
  - Splits the dataset into training and testing sets.

---

## Execution Code
<p>
Run the following code from main.py file to run Random Forest
</p>

![image](https://github.com/user-attachments/assets/6571a139-5b9f-47bd-94bc-2de6a4ea712f)

---

## Sample_Execution_Output

```
Loading raw data...
Preprocessing data...
Loading transformed data...
Fitting RandomForest model...
Training set size: (12000, 12), Testing set size: (4213, 12)
Making predictions on the test set...
Test Set Accuracy: 89.22%
```

---


