# Foundations of Artificial Intelligence - Fraud Detection Application

## Table of Contents

- [Introduction](#Introduction)
- [Module Descriptions](#Module-Descriptions)
- [Getting Started](#Getting-Started)
- [Requirements](#Requirements)
- [Installation and Execution](#Installation-and-Execution)
- [Acknowledgements](#Acknowledgements)

## Introduction

The financial fraud detection system contains both supervised and unsupervised learning methods on a financial dataset to determine which transactions can and cannot be classified as fraud. The system has been trained on a large data set of more than six million data points and can accurately predict fraud in up to 90% of different cases. By leveraging various approaches such as K-Means, random forest, anomaly detection, and a feed-forward neural network we as a team were able to gain valuable insights into how financial fraud can and cannot be properly classified.

## Module Descriptions

- main.py - main script for execution the application
- logger.py - custom logger for terminal output
- constants.py - class to encapsulate all constant fields
- fraud_data.py - class logic to extract our data from a locally downloaded zip file
- custom_data_loader.py - implementation of the fraud_data.py script to extract data
- k_means_learning.py - script that encapsulates all k-means unsupervised learning logic 
- neural_network.py - script that encapsulates all neural network supervised learning logic
- requirements.txt - holds all relevant application dependencies
- anomaly_detection.py - script that encapsulate all isolation forest unsupervised learning logic
- random_forest.py - MALHAR TO FILL IN
- data_dictionary_20241009.xlsx - a simple excel file that outlines the description of the feature and labels contained in our dataset

## Getting Started

### Requirements

- Python 3.12.4
- tqdm~=4.66.6
- torch~=2.4.1
- pandas~=2.2.3
- seaborn~=0.13.2
- matplotlib~=3.9.2
- scikit-learn~=1.5.2
- tensorboard~=2.18.0
- tensorboard-data-server~=0.7.2

### Installation and Execution
#### In order to execute the application you must download the following dataset, <a href="https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset/data">Financial Fraud Detection Dataset</a>.

1. After downloading the dataset, the dataset must be compressed into a ZIP file in order for the application to work as intended.


2. The dataset must be placed into the following directory 

   ```bash
   cs_5100_fraud_detection/main/data/fraud_detection_data_set
   ```

3. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cs_5100_fraud_detection.git
   ```

4. Navigate to the project directory:
   ```bash
   cd cs_5100_fraud_detection/main
   ```

5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

6. Run the application:
   ```bash
   python3 main.py
   ```

## Acknowledgements

The following resources were used throughout the course of this project:

- <a href="https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset/data">Financial Fraud Detection Dataset</a> as the primary data set we leveraged</li>
