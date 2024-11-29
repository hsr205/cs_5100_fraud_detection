# Neural Network Documentation

<p>
This documentation express the contents of each class within the 
neural_network.py module and explains the intricacies of each classes methods
We additional aim to provide reasoning for the choices we made 
with regard to the specific approaches used within our neural network.
</p>


## Table of Contents

- [Classes](#Classes)
    - [DataPreprocessor](#DataPreprocessor)
    - [NeuralNetwork](#NeuralNetwork)
    - [Model](#Model)
    - [Execution_Code](#Execution_Code)
    - [Sample_Execution_Output](#Sample_Execution_Output)

## Classes


## DataPreprocessor


`__init__()` - Initializes the DataTransformer with a DataFrame, setting it up for further transformations and tensor
conversions.

`get_training_loader()` - FILL IN

`get_test_loader()` - FILL IN

---

## NeuralNetwork


`__init__()`

- In the initializer we aimed to construct our neural network using an input layer of the following features
    1. Amount Transferred
    2. Transaction Type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH-IN)
    3. New Balance Origin
    4. New Balance Destination

`forward()` - ADD DESCRIPTION

---

## Model

`__init__()` - ADD DESCRIPTION

`train_neural_network()` - will train the neural network and return a list of lists called epoch_loss_matrix. The list
at position k will be contained the running loss for the kth epoch.

`test_neural_network()` - FILL IN

`write_results()` - using the takes the results of the most recent model training and stores it in a

`launch_tensor_board()` - will launch tensor board at the local host http://localhost:6006/ where the loss function over
time of all runs found in main/machine_learning/neural_network_execution_results

`display_testing_results()` - FILL IN

`save_model_state()` - saves the model weights as a dictionary stored in a .pth file to be used later

`get_device()` - will check to see if cuda is available and if so move the model tensor to the available GPU

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
    epoch_loss_list: list[list[float]] = model.train_neural_network(epochs=20)
    model.write_results(epoch_loss_list=epoch_loss_list)
    model.save_model_state()
    accuracy_results: Accuracy = model.test_neural_network()
    model.display_testing_results(accuracy_obj=accuracy_results)
    model.launch_tensor_board()
    return 0
```

---

## Sample_Execution_Output

```
2024-11-18 12:45:48 PM - INFO - File found in zip: synthetic_financial_datasets_log.csv
2024-11-18 12:45:53 PM - INFO - File found in zip: synthetic_financial_datasets_log.csv
2024-11-18 12:45:59 PM - INFO - Using MPS Device
2024-11-18 12:46:00 PM - INFO - Starting Neural Network Training
2024-11-18 12:46:00 PM - INFO - ===============================================
Neural Network Training Progress:   0%|          | 0/20 [00:00<?, ?it/s]2024-11-18 12:46:00 PM - INFO - Epoch 1/20, Loss: 16.5401
Neural Network Training Progress:   5%|▌         | 1/20 [00:00<00:07,  2.45it/s]2024-11-18 12:46:00 PM - INFO - Epoch 2/20, Loss: 16.3824
2024-11-18 12:46:00 PM - INFO - Epoch 3/20, Loss: 16.0559
Neural Network Training Progress:  15%|█▌        | 3/20 [00:00<00:02,  5.67it/s]2024-11-18 12:46:00 PM - INFO - Epoch 4/20, Loss: 15.3904
2024-11-18 12:46:00 PM - INFO - Epoch 5/20, Loss: 14.3259
Neural Network Training Progress:  25%|██▌       | 5/20 [00:00<00:02,  7.42it/s]2024-11-18 12:46:01 PM - INFO - Epoch 6/20, Loss: 12.9335
Neural Network Training Progress:  30%|███       | 6/20 [00:00<00:01,  7.38it/s]2024-11-18 12:46:01 PM - INFO - Epoch 7/20, Loss: 11.5259
2024-11-18 12:46:01 PM - INFO - Epoch 8/20, Loss: 10.1343
Neural Network Training Progress:  40%|████      | 8/20 [00:01<00:01,  8.48it/s]2024-11-18 12:46:01 PM - INFO - Epoch 9/20, Loss: 9.0562
2024-11-18 12:46:01 PM - INFO - Epoch 10/20, Loss: 8.4102
Neural Network Training Progress:  50%|█████     | 10/20 [00:01<00:01,  9.18it/s]2024-11-18 12:46:01 PM - INFO - Epoch 11/20, Loss: 7.8778
2024-11-18 12:46:01 PM - INFO - Epoch 12/20, Loss: 7.5632
Neural Network Training Progress:  60%|██████    | 12/20 [00:01<00:00,  9.62it/s]2024-11-18 12:46:01 PM - INFO - Epoch 13/20, Loss: 7.3796
2024-11-18 12:46:01 PM - INFO - Epoch 14/20, Loss: 7.3268
Neural Network Training Progress:  70%|███████   | 14/20 [00:01<00:00,  9.91it/s]2024-11-18 12:46:01 PM - INFO - Epoch 15/20, Loss: 7.0651
2024-11-18 12:46:02 PM - INFO - Epoch 16/20, Loss: 7.0095
Neural Network Training Progress:  80%|████████  | 16/20 [00:01<00:00, 10.12it/s]2024-11-18 12:46:02 PM - INFO - Epoch 17/20, Loss: 6.9883
2024-11-18 12:46:02 PM - INFO - Epoch 18/20, Loss: 7.0293
Neural Network Training Progress:  90%|█████████ | 18/20 [00:02<00:00, 10.20it/s]2024-11-18 12:46:02 PM - INFO - Epoch 19/20, Loss: 6.9040
2024-11-18 12:46:02 PM - INFO - Epoch 20/20, Loss: 6.9420
Neural Network Training Progress: 100%|██████████| 20/20 [00:02<00:00,  8.85it/s]
2024-11-18 12:46:02 PM - INFO - ===============================================
2024-11-18 12:46:02 PM - INFO - Completed Neural Network Training
2024-11-18 12:46:02 PM - INFO - ===============================================
2024-11-18 12:46:02 PM - INFO - Saved neural network execution results: <FILE-PATH>
2024-11-18 12:46:02 PM - INFO - Saved neural network state: <FILE-PATH>
2024-11-18 12:46:02 PM - INFO - Starting Neural Network Testing
2024-11-18 12:46:02 PM - INFO - ===============================================
Neural Network Testing Progress: 100%|██████████| 8/8 [00:00<00:00, 155.46it/s]
2024-11-18 12:46:02 PM - INFO - ==============================================
2024-11-18 12:46:02 PM - INFO - Total Observations = 4,000
2024-11-18 12:46:02 PM - INFO - Correctly Predicted Observations = 3,604
2024-11-18 12:46:02 PM - INFO - Neural Network Accuracy: 90.10%
2024-11-18 12:46:02 PM - INFO - ==============================================
2024-11-18 12:46:02 PM - INFO - Completed Neural Network Testing
2024-11-18 12:46:02 PM - INFO - ==============================================
2024-11-18 12:46:02 PM - INFO - Launching TensorBoard:
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.18.0 at http://localhost:6006/ (Press CTRL+C to quit)

Process finished with exit code 0
```