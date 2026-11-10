# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to stay or leave using classification algorithms.

## Project Overview

Customer churn occurs when customers discontinue a company's services. This project analyzes telecom customer data and builds machine learning models to identify customers who are likely to churn.

Two classification algorithms are implemented and compared:
- Decision Tree Classifier
- Random Forest Classifier

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib

## Key Features

- Data cleaning and preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Decision Tree and Random Forest model comparison
- Model evaluation using Accuracy, Precision, Recall, F1-score and ROC-AUC
- Confusion matrix and ROC curve visualizations
- Feature importance analysis
- Saving and loading the trained machine learning pipeline

## Model Performance

Random Forest results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 76.37% |
| Precision (Churn) | 54.85% |
| Recall (Churn) | 62.03% |
| F1-score (Churn) | 58.22% |
| ROC-AUC | 82.27% |

## Dataset

IBM Telco Customer Churn dataset from Kaggle, containing 7,043 customer records.

## Project Structure

```text
Customer-Churn-Prediction/
├── churn_prediction.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/
    ├── feature_importance.png
    └── roc_curve.png
```

## How to Run

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the project:

```bash
python churn_prediction.py
```

## Future Development

A web-based customer churn intelligence dashboard using Flask, MySQL and the trained machine learning model.

## Author

Shalini T
