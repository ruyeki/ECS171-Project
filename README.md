# Bankruptcy Detector

This repository contains code for building and tuning a Support Vector Machine (SVM) model to predict bankruptcy using financial data. The project involves data preprocessing, model selection, hyperparameter tuning, and comprehensive evaluation of the final model.

## Overview

The project is divided into the following key steps:

1. **Data Preprocessing**:
   - **Loading the Dataset**: The financial dataset (`data.csv`) is loaded using the Pandas library. 
   - **Handling Missing Values**: Any missing values in the dataset are identified and filled using the median value of each feature to ensure completeness.
   - **Removing Duplicates**: Duplicate entries, if any, are removed from the dataset to maintain data integrity.
   - **Outlier Detection and Removal**: Outliers are detected using the Interquartile Range (IQR) method and subsequently removed to prevent them from affecting model performance.
   - **Feature Scaling**: Feature scaling is performed to standardize the range of features, ensuring that no single feature dominates the learning process.

2. **Exploratory Data Analysis (EDA)**:
   - **Understanding Data Characteristics**: Statistical summaries and visualizations are generated to gain insights into the distribution and relationships between features.
   - **Correlation Analysis**: Correlation matrices are computed to identify potential correlations between features and the target variable (bankruptcy).

3. **Model Selection**:
   - **Initial Exploration**: Given the binary classification nature of the problem and the need for handling complex relationships in high-dimensional data, Support Vector Machines (SVMs) are considered suitable candidates.
   - **Evaluation Metrics**: Various evaluation metrics such as accuracy, precision, recall, and F1 score are used to assess model performance.
   - **Learning Curves**: Learning curves are plotted to visualize the model's performance as the training size increases, helping to identify any issues related to bias or variance.

4. **Model Training and Validation**:
   - **Splitting the Dataset**: The dataset is split into training and testing sets to train and evaluate the models, respectively.
   - **Cross-Validation**: Cross-validation techniques such as Stratified K-Fold are employed to ensure robust model evaluation and mitigate overfitting.

5. **Hyperparameter Tuning**:
   - **Grid Search**: Hyperparameter tuning is performed using Grid Search with cross-validation to find the optimal combination of hyperparameters (C, gamma, kernel).
   - **Round 1 (F1 Optimization)**: Grid search is conducted to optimize the F1 score, ensuring a balance between precision and recall.
   - **Round 2 (Recall Optimization)**: Grid search is repeated, this time prioritizing recall to correctly identify companies at risk of bankruptcy.
   - **Round 3 (Fine-Tuning)**: Further hyperparameter tuning is performed to improve both F1 score and recall, ensuring a balanced performance.

6. **Model Evaluation (Final)**:
   - **Comprehensive Metrics**: The final SVM model, selected based on the best performance metrics from hyperparameter tuning, is evaluated on the dataset.
   - **Confusion Matrix Analysis**: The confusion matrix is analyzed to understand the model's ability to correctly classify instances into true positives, true negatives, false positives, and false negatives.
   - **Precision-Recall Tradeoff**: Precision, recall, and F1 score are computed to provide insights into the model's effectiveness in identifying bankrupt companies.

7. **Conclusion and Future Work**:
   - **Performance Analysis**: A summary of the model's performance and its implications for real-world financial analysis scenarios.
   - **Future Directions**: Potential areas for improvement or further research, such as feature engineering or exploring different machine learning algorithms.

## Files

- `bankruptcy_prediction.ipynb`: Jupyter Notebook containing the Python code for the entire project.
- `data.csv`: Dataset used for training and testing the models.

## Requirements

The project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Usage

To reproduce the results:

1. Clone the repository:

```bash
git clone https://github.com/your-username/bankruptcy-prediction.git
