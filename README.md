# Breast Cancer Diagnosis Model

## Overview
This project implements a machine learning model to classify breast cancer tumors as malignant or benign using the Wisconsin Breast Cancer dataset. The model employs a Random Forest classifier, which is a robust ensemble learning technique.

## Description of the Code
1. **Data Fetching**: 
   - The code begins by importing necessary libraries, including `numpy`, `pandas`, `tensorflow`, and various components from `sklearn`.
   - It fetches the Wisconsin Breast Cancer dataset using the `fetch_ucirepo` function with a specific ID (17).

2. **Data Preparation**:
   - The dataset is split into features (X) and targets (y), where the target variable indicates whether the tumor is malignant (`M`) or benign (`B`).
   - The target variable `y` is encoded to binary values: `1` for malignant and `0` for benign.

3. **Data Splitting**:
   - The dataset is divided into training and evaluation sets using an 80-20 split with `train_test_split`.

4. **Data Preprocessing**:
   - Categorical and numeric features are identified.
   - A preprocessing pipeline is created for numeric features (using `StandardScaler`) and categorical features (using `OneHotEncoder`).
   - A `ColumnTransformer` combines these pipelines to transform the feature set.

5. **Model Training**:
   - A Random Forest classifier is instantiated with 100 estimators.
   - The model is trained on the preprocessed training data.

6. **Model Evaluation**:
   - Predictions are made on the evaluation set, and the model's accuracy is calculated using `accuracy_score`.
   - The accuracy score is printed to the console.

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
