# Neural Network Documentation

---
<p>
This documentation express the contents of each class within the 
neural_network.py module and explains the intricacies of each classes methods
We additional aim to provide reasoning for the choices we made 
with regard to the specific approaches used within our neural network.
</p>

---

## Table of Contents
- [Classes](#Classes)
  - [DataPreprocessor](#DataPreprocessor)
  - [NeuralNetwork](#NeuralNetwork)
  - [Model](#Model)
  - [Execution_Code](#Execution_Code)
  - [Sample_Execution_Output](#Sample_Execution_Output)
  - [Results](#Results)


## Classes

---

## DataPreprocessor

---


`__init__()` - ADD DESCRIPTION

`preprocess_data_frame()` - ADD DESCRIPTION

`get_y_labels_as_tensor()` - ADD DESCRIPTION

`get_x_labels_as_tensor()` - ADD DESCRIPTION


---


## NeuralNetwork

---


`__init__()`

- In the initializer we aimed to construct our neural network using an input layer of the following features
  1. Amount Transferred
  2. Transaction Type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH-IN)
  3. New Balance Origin
  4. New Balance Destination

`forward()` - ADD DESCRIPTION

---

## Model

---

`__init__()` - ADD DESCRIPTION

---

## Execution_Code

<p>
Run the following code in the main.py file in order to train the neural network:
</p>

```python
def main() -> int:
    data_loader: CustomDataLoader = CustomDataLoader()

    # Include local file path to zip file
    file_path: str = ""
    file_name: str = "synthetic_financial_datasets_log.csv"

    fraud_data: FraudDataset = FraudDataset(data_loader=data_loader,
                                            file_path=file_path,
                                            file_name=file_name,
                                            transform_to_tensor=None,
                                            target_transform=None)

    fraud_data_frame: pd.DataFrame = fraud_data.data_loader.get_data_frame_from_zip_file(file_path=file_path,
                                                                                         file_name=file_name)

    model:Model = Model(fraud_data_frame=fraud_data_frame)
    model.train_neural_network()
    return 0
```

---

## Sample_Execution_Output

```
(.venv) <USER-NAME>@<USER-MACHINE>:cs_5100_fraud_detection$ python3 main/main.py 
2024-10-31 11:34:14 AM - INFO - File found in zip: synthetic_financial_datasets_log.csv
2024-10-31 11:34:19 AM - INFO - File found in zip: synthetic_financial_datasets_log.csv
2024-10-31 11:34:24 AM - INFO - Using mps device
  0%|          | 0/10 [00:00<?, ?it/s]2024-10-31 11:34:32 AM - INFO - Epoch 1/10, Loss: 0.0402
 10%|█         | 1/10 [00:07<01:05,  7.27s/it]2024-10-31 11:34:39 AM - INFO - Epoch 2/10, Loss: 0.0077
 20%|██        | 2/10 [00:14<00:57,  7.24s/it]2024-10-31 11:34:46 AM - INFO - Epoch 3/10, Loss: 0.0073
 30%|███       | 3/10 [00:21<00:50,  7.23s/it]2024-10-31 11:34:54 AM - INFO - Epoch 4/10, Loss: 0.0069
 40%|████      | 4/10 [00:28<00:42,  7.16s/it]2024-10-31 11:35:01 AM - INFO - Epoch 5/10, Loss: 0.0067
 50%|█████     | 5/10 [00:36<00:36,  7.21s/it]2024-10-31 11:35:08 AM - INFO - Epoch 6/10, Loss: 0.0064
 60%|██████    | 6/10 [00:43<00:28,  7.15s/it]2024-10-31 11:35:15 AM - INFO - Epoch 7/10, Loss: 0.0063
 70%|███████   | 7/10 [00:50<00:21,  7.14s/it]2024-10-31 11:35:22 AM - INFO - Epoch 8/10, Loss: 0.0061
 80%|████████  | 8/10 [00:57<00:14,  7.17s/it]2024-10-31 11:35:29 AM - INFO - Epoch 9/10, Loss: 0.0059
 90%|█████████ | 9/10 [01:04<00:07,  7.13s/it]2024-10-31 11:35:36 AM - INFO - Epoch 10/10, Loss: 0.0058
100%|██████████| 10/10 [01:11<00:00,  7.16s/it]
```


---

## Results

---

| Model Num | Num Input Layer Nodes | Num Hidden Layer Nodes | Num Output Layer Nodes  |      Criterion       | Optimizer | Learning Rate |  Num Epochs  | Loss Function Result After N-Epochs |
|:---------:|:---------------------:|:----------------------:|:-----------------------:|:--------------------:|:---------:|:-------------:|:------------:|:-----------------------------------:|
|     1     |           8           |           5            |            1            | Binary Cross Entropy |   Adam    |     0.001     |      10      |               0.0058                |
|           |                       |                        |                         |                      |           |               |              |                                     |


---



