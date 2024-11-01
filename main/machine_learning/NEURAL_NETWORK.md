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

`train_neural_network()` - ADD DESCRIPTION

`get_device()` - ADD DESCRIPTION

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

    model: Model = Model(fraud_data_frame=fraud_data_frame)
    epoch_loss_list: list[float] = model.train_neural_network()
    model.write_results(epoch_loss_list=epoch_loss_list)
    model.save_model_state()
    model.launch_tensor_board()
    return 0
```

---

## Sample_Execution_Output

```
2024-11-01 10:01:08 AM - INFO - File found in zip: synthetic_financial_datasets_log.csv
2024-11-01 10:01:13 AM - INFO - File found in zip: synthetic_financial_datasets_log.csv
2024-11-01 10:01:19 AM - INFO - Using MPS Device
training neural network :   0%|          | 0/10 [00:00<?, ?it/s]2024-11-01 10:01:27 AM - INFO - Epoch 1/10, Loss: 0.0586
training neural network :  10%|█         | 1/10 [00:08<01:14,  8.29s/it]2024-11-01 10:01:35 AM - INFO - Epoch 2/10, Loss: 0.0201
training neural network :  20%|██        | 2/10 [00:15<01:03,  7.92s/it]2024-11-01 10:01:42 AM - INFO - Epoch 3/10, Loss: 0.0130
training neural network :  30%|███       | 3/10 [00:23<00:53,  7.68s/it]2024-11-01 10:01:50 AM - INFO - Epoch 4/10, Loss: 0.0112
training neural network :  40%|████      | 4/10 [00:30<00:45,  7.60s/it]2024-11-01 10:01:57 AM - INFO - Epoch 5/10, Loss: 0.0104
training neural network :  50%|█████     | 5/10 [00:38<00:37,  7.55s/it]2024-11-01 10:02:05 AM - INFO - Epoch 6/10, Loss: 0.0103
training neural network :  60%|██████    | 6/10 [00:45<00:30,  7.51s/it]2024-11-01 10:02:12 AM - INFO - Epoch 7/10, Loss: 0.0101
training neural network :  70%|███████   | 7/10 [00:53<00:22,  7.49s/it]2024-11-01 10:02:20 AM - INFO - Epoch 8/10, Loss: 0.0099
training neural network :  80%|████████  | 8/10 [01:00<00:14,  7.47s/it]2024-11-01 10:02:27 AM - INFO - Epoch 9/10, Loss: 0.0099
training neural network :  90%|█████████ | 9/10 [01:08<00:07,  7.45s/it]2024-11-01 10:02:34 AM - INFO - Epoch 10/10, Loss: 0.0098
training neural network : 100%|██████████| 10/10 [01:15<00:00,  7.54s/it]
2024-11-01 10:02:34 AM - INFO - Saved neural network execution results: <ABSOLUTE-FILE-PATH>
2024-11-01 10:02:34 AM - INFO - Saved neural network state: <ABSOLUTE-FILE-PATH>
2024-11-01 10:02:34 AM - INFO - Launching TensorBoard
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.18.0 at http://localhost:6006/ (Press CTRL+C to quit)

Process finished with exit code 0
```


---

## Results

---

| Model Num | Num Input Layer Nodes | Num Hidden Layer Nodes | Num Output Layer Nodes  |      Criterion       | Optimizer | Learning Rate |  Num Epochs  | Loss Function Result After N-Epochs |
|:---------:|:---------------------:|:----------------------:|:-----------------------:|:--------------------:|:---------:|:-------------:|:------------:|:-----------------------------------:|
|     1     |           8           |           5            |            1            | Binary Cross Entropy |   Adam    |     0.001     |      10      |               0.0058                |
|           |                       |                        |                         |                      |           |               |              |                                     |


---



