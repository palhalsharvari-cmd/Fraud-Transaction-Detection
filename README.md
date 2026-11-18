
# Fraud Transaction Detection System

## Project Overview
A Machine Learning-based application that identifies potentially fraudulent financial transactions using classification algorithms.

The project includes a Flask web application that accepts transaction details and displays the model's prediction and estimated fraud probability.

## Dataset
- Dataset: BankSim Synthetic Transaction Dataset
- Total transactions: 594,643
- Legitimate transactions: 587,443
- Fraudulent transactions: 7,200
- Fraud percentage: Approximately 1.21%

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- HTML and CSS
- Joblib

## Machine Learning Models
1. Logistic Regression
2. Random Forest Classifier

## Final Model
Random Forest Classifier with class balancing and validation-based threshold optimization.

## Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | 99.53% |
| Precision | 81.05% |
| Recall | 80.21% |
| F1 Score | 80.63% |
| ROC-AUC | 99.34% |

## Model Optimization
The classification threshold was optimized using validation data to improve the F1 score.

Optimized threshold: 0.34

## Web Application
The Flask application allows users to:
- Enter transaction details.
- Submit transaction information.
- View the predicted transaction category.
- View the estimated fraud probability.

## Project Structure

```text
Fraud_Transaction_Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── templates/
    └── index.html
