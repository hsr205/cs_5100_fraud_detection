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

## Results

---

- ADD MODEL RESULTS TO THIS SECTION

| Model Num | Num Input Layer Nodes | Num Hidden Layer Nodes | Num Output Layer Nodes  |      Criterion       | Optimizer | Learning Rate |  Num Epochs  | Loss Function Result After Epochs |
|:---------:|:---------------------:|:----------------------:|:-----------------------:|:--------------------:|:---------:|:-------------:|:------------:|:---------------------------------:|
|     1     |          10           |           5            |            1            | Binary Cross Entropy |   Adam    |     0.001     |      10      |                                   |
|           |                       |                        |                         |                      |           |               |              |                                   |



---

