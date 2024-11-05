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


`__init__()` -

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

`train_neural_network()` - will train the neural network and return a list of lists called epoch_loss_matrix. The list
at position k will be contain the running loss for the kth epoch.

`get_device()` - will check to see if cuda is available and if so move the model tensor to the available GPU

`write_results()` - using the takes the results of the most recent model training and stores it in a

`save_model_state()` - saves the model weights as a dictionary stored in a .pth file to be used later

`launch_tensor_board()` - will launch tensor board at the local host http://localhost:6006/ where the loss function over
time of all runs found in main/machine_learning/neural_network_execution_results

---

## Execution_Code

<p>
Run the following code in the main.py file in order to train the neural network:
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

    batch_size: int = 128
    num_observations: int = 500000

    model: Model = Model(fraud_data_frame=fraud_data_frame)
    epoch_loss_list: list[list[float]] = model.train_neural_network(num_observations=num_observations,
                                                                    batch_size=batch_size)
    model.write_results(epoch_loss_list=epoch_loss_list)
    model.save_model_state()
    model.launch_tensor_board()
    return 0
```

---

## Sample_Execution_Output

---

```
2024-11-05 07:57:23 AM - INFO - File found in zip: synthetic_financial_datasets_log.csv
2024-11-05 07:57:29 AM - INFO - File found in zip: synthetic_financial_datasets_log.csv
2024-11-05 07:57:34 AM - INFO - Using MPS Device
2024-11-05 07:57:35 AM - INFO - Batch Size: 128
2024-11-05 07:57:35 AM - INFO - Number of Data Set Observations Used: 500,000
2024-11-05 07:57:35 AM - INFO - ===============================================
Neural Network Training Progress:   0%|          | 0/10 [00:00<?, ?it/s]2024-11-05 07:57:45 AM - INFO - Epoch 1/10, Loss: 0.0955
Neural Network Training Progress:  10%|█         | 1/10 [00:10<01:33, 10.34s/it]2024-11-05 07:57:55 AM - INFO - Epoch 2/10, Loss: 0.0196
Neural Network Training Progress:  20%|██        | 2/10 [00:20<01:20, 10.02s/it]2024-11-05 07:58:05 AM - INFO - Epoch 3/10, Loss: 0.0110
Neural Network Training Progress:  30%|███       | 3/10 [00:30<01:09,  9.97s/it]2024-11-05 07:58:15 AM - INFO - Epoch 4/10, Loss: 0.0095
Neural Network Training Progress:  40%|████      | 4/10 [00:39<00:59,  9.92s/it]2024-11-05 07:58:25 AM - INFO - Epoch 5/10, Loss: 0.0092
Neural Network Training Progress:  50%|█████     | 5/10 [00:49<00:49,  9.94s/it]2024-11-05 07:58:34 AM - INFO - Epoch 6/10, Loss: 0.0092
Neural Network Training Progress:  60%|██████    | 6/10 [00:59<00:39,  9.90s/it]2024-11-05 07:58:45 AM - INFO - Epoch 7/10, Loss: 0.0089
Neural Network Training Progress:  70%|███████   | 7/10 [01:09<00:29,  9.98s/it]2024-11-05 07:58:55 AM - INFO - Epoch 8/10, Loss: 0.0089
Neural Network Training Progress:  80%|████████  | 8/10 [01:20<00:20, 10.07s/it]2024-11-05 07:59:05 AM - INFO - Epoch 9/10, Loss: 0.0089
Neural Network Training Progress:  90%|█████████ | 9/10 [01:30<00:10, 10.07s/it]2024-11-05 07:59:15 AM - INFO - Epoch 10/10, Loss: 0.0088
Neural Network Training Progress: 100%|██████████| 10/10 [01:40<00:00, 10.02s/it]
2024-11-05 07:59:16 AM - INFO - Saved neural network execution results: <FILE-PATH>
2024-11-05 07:59:16 AM - INFO - Saved neural network state: <FILE-PATH>

Process finished with exit code 0
```

---

## Results

---

| Model Num | Num Input Layer Nodes | Num Hidden Layer Nodes | Num Output Layer Nodes | Learning Rate | Num Epochs | Used Drop Out Layers | Loss Function Result After N-Epochs |
|:---------:|:---------------------:|:----------------------:|:----------------------:|:-------------:|:----------:|:--------------------:|:-----------------------------------:|
|     1     |           8           |           5            |           1            |     0.001     |     10     |          No          |               0.0058                |
|     2     |           8           |           5            |           1            |     0.001     |     11     |         Yes          |               0.0088                |

---


