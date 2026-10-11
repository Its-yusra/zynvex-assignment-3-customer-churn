# Customer Churn Prediction with a Tuned Neural Network

## Assignment Information

- **Organization:** Zynvex
- **Track:** AI/ML Internship — Week 3
- **Assignment:** Graded Assignment 03 — Weekend Deep Learning Project
- **Student:** Yusra Komal
- **Language:** Python 3

## Project Overview

This project predicts whether a telecommunications customer is likely to leave a company using a neural network built with TensorFlow and Keras.

The Telco Customer Churn dataset contains customer demographics, subscribed services, account information, contract types, monthly charges, total charges, and the target variable `Churn`.

## Objectives

- Explore and clean the customer churn dataset.
- Handle missing values and convert `TotalCharges` to numeric.
- Encode categorical features and standardize numerical features.
- Build and evaluate a baseline neural network.
- Improve the model using regularization and Early Stopping.
- Compare baseline and tuned model performance.
- Evaluate the model using precision, recall, F1-score, and a confusion matrix.
- Save the trained model and predict churn for a hypothetical customer.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TensorFlow and Keras
- Google Colab

## Dataset

**Source:** [Telco Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset includes customer information and the `Churn` target column, indicating whether a customer left the service.

The `TotalCharges` column is converted to numeric, and records with invalid or missing values in this column are removed. The `customerID` column is excluded from model training because it is an identifier.

## Project Workflow

### 1. Data Exploration and Cleaning

- Inspect dataset shape, column types, missing values, and duplicates.
- Analyze the distribution of churn classes.
- Visualize churn by contract type.
- Convert `TotalCharges` to numeric and handle invalid values.

### 2. Data Preprocessing

- Encode the target as 0 for No Churn and 1 for Churn.
- Apply one-hot encoding to categorical features.
- Split the data into training and test sets.
- Standardize features using `StandardScaler`, fitted only on training data.

### 3. Baseline Neural Network

Build a Keras Sequential model with two hidden Dense layers and a sigmoid output layer. Train the model and visualize training and validation loss and accuracy.

### 4. Model Tuning

Apply the following techniques to help reduce overfitting:

- Dropout
- L2 regularization
- Batch Normalization
- Early Stopping

Compare the tuned model with the baseline using test accuracy.

### 5. Model Evaluation

Evaluate the tuned model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics provide a broader view of model performance, particularly when churn classes are imbalanced.

### 6. Model Saving and Prediction

Save the trained model in Keras format and reload it for inference. Use a hypothetical customer record to demonstrate churn prediction.

## Repository Files

| File | Description |
|---|---|
| `Assignment3_YusraKomal.ipynb` | Complete notebook containing code, visualizations, training, evaluation, and prediction |
| `best_churn_model.keras` | Saved tuned neural network |

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Download the dataset from Kaggle.
3. Upload the CSV file when prompted by the notebook.
4. Run all cells in order.
5. Review the generated visualizations, evaluation metrics, and hypothetical customer prediction.
6. Save the notebook and the trained model file.

## Results

The notebook generates the baseline and tuned model test accuracy, classification metrics, and confusion matrix from the actual model runs.

**Note:** Performance values are not specified here because they must be taken from the executed notebook.

## Conclusion

This project demonstrates an end-to-end deep learning workflow for customer churn prediction, including data preprocessing, neural network development, model tuning, performance evaluation, model saving, and inference on a hypothetical customer.

The project applies foundational deep learning techniques to a real-world classification problem in the telecommunications domain.
