# Predicting Customer Churn in Telecom Using Ensemble Methods

## Overview

This project presents a machine learning model designed to predict customer churn in the telecommunications industry. The model leverages various classification algorithms to analyze customer data and predict whether a customer is likely to churn. The model has been trained and evaluated on a comprehensive dataset, achieving high accuracy and robust performance.

## Model Details

- **Model Type**: Ensemble of multiple classifiers (Voting Classifier, Gradient Boosting Classifier, Logistic Regression, etc.)
- **Evaluation Results**:
  - **Accuracy**: 79.89%
  - **ROC AUC Mean**: 84.79%

The model was trained on a dataset containing various customer attributes and their churn status, making it a reliable choice for predicting customer churn in the telecommunications sector.

## Intended Use

This model is designed to predict customer churn based on their profile and usage patterns. You can use it for applications like:
- Customer retention strategies
- Targeted marketing campaigns
- Customer service improvements
- Any other application where predicting customer churn is useful

## Limitations

- The model's performance may vary depending on the specific characteristics of the input data.
- The dataset used for training is specific to the telecommunications industry, so the model may not generalize well to other industries.

## Training Procedure

The model was trained using the following setup:

- **Data Preparation**:
  - Data cleaning and preprocessing to handle missing values and encode categorical variables.
  - Feature scaling using StandardScaler.
- **Model Training**:
  - Multiple classifiers were trained and evaluated, including Logistic Regression, Support Vector Classifier, K-Nearest Neighbour, Gaussian Naive Bayes, Decision Tree Classifier, Random Forest, AdaBoost, Gradient Boosting Classifier, and Voting Classifier.
  - Hyperparameter tuning and cross-validation were performed to optimize model performance.
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1 Score, and ROC AUC were used to evaluate model performance.

### Training Results:
- **Logistic Regression**: Accuracy = 74.64%, ROC AUC = 84.30%
- **Support Vector Classifier**: Accuracy = 79.07%, ROC AUC = 82.94%
- **Random Forest**: Accuracy = 78.83%, ROC AUC = 82.92%
- **Gradient Boosting Classifier**: Accuracy = 79.36%, ROC AUC = 84.63%
- **Voting Classifier**: Accuracy = 79.89%, ROC AUC = 84.79%

## Framework and Libraries

- **Scikit-learn**: 1.2.2
- **Pandas**: 1.5.3
- **NumPy**: 1.24.3
- **Matplotlib**: 3.7.1
- **Seaborn**: 0.12.2
- **Plotly**: 5.15.0

## Usage

To use the model, you can load it with the following code:

```python
import joblib

# Load the trained model
model = joblib.load('telecom_churn_model.pkl')

# Example input data (features)
input_data = [[0, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 2, 0, 1, 2, 56.95, 1889.5]]

# Predict churn status
prediction = model.predict(input_data)

print("Churn Prediction:", prediction)
```

This will return the predicted churn status for the given input data.

## Conclusion

This **Predicting Customer Churn in Telecom Using Ensemble Methods** is a powerful tool for analyzing customer churn in the telecommunications industry. It's accurate, robust, and can be easily integrated into various applications where predicting customer churn is required.
