# Customer Churn Prediction
# Step 1: Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")
# Step 2: Load dataset

file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print(df.head())
# Step 3: Dataset overview

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())
# Step 4: Data Cleaning

# Remove customerID because it is not useful for prediction
df = df.drop("customerID", axis=1)

# Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nMissing values after conversion:")
print(df.isnull().sum())

print("\nData types after cleaning:")
print(df.dtypes)
# Step 5: Handle missing values

median_value = df["TotalCharges"].median()

df["TotalCharges"] = df["TotalCharges"].fillna(median_value)

print("\nMissing values after imputation:")
print(df.isnull().sum().sum())

print("\nTotalCharges missing values handled successfully!")
# Step 6: Define features and target

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nChurn distribution:")
print(y.value_counts())
# Step 7: Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset split completed!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))
# Step 8: Preprocessing Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(include=["str", "object"]).columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nPreprocessing pipeline created successfully!")
# Step 9: Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

dt_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\nDecision Tree trained successfully!")
# Step 10: Random Forest Model

from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest trained successfully!")
# Step 11: Model Evaluation

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Performance")
    print("-" * 35)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions), 4))
    print("Recall:", round(recall_score(y_test, predictions), 4))
    print("F1-Score:", round(f1_score(y_test, predictions), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

evaluate_model("Decision Tree", dt_model, X_test, y_test)
evaluate_model("Random Forest", rf_model, X_test, y_test)
# Step 12: Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, model, name in [
    (axes[0], dt_model, "Decision Tree"),
    (axes[1], rf_model, "Random Forest")
]:
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stay", "Churn"]
    ).plot(ax=ax, cmap="Greens", colorbar=False)

    ax.set_title(name)

plt.tight_layout()
plt.show()
# Step 13: ROC Curve Comparison

# Step 13: ROC Curve Comparison

from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_estimator(
    dt_model, X_test, y_test,
    name="Decision Tree",
    ax=ax
)

RocCurveDisplay.from_estimator(
    rf_model, X_test, y_test,
    name="Random Forest",
    ax=ax
)

ax.set_title("ROC Curve - Model Comparison")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    "outputs/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)
# Step 14: Feature Importance

rf_classifier = rf_model.named_steps["classifier"]
rf_preprocessor = rf_model.named_steps["preprocessor"]

feature_names = rf_preprocessor.get_feature_names_out()

importance = pd.Series(
    rf_classifier.feature_importances_,
    index=feature_names
).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
importance.sort_values().plot(kind="barh")

plt.title("Top 10 Factors Influencing Customer Churn")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig("outputs/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
# Step 15: Save the trained model

import joblib

joblib.dump(rf_model, "customer_churn_model.pkl")

print("\nRandom Forest model saved successfully!")
# Step 16: Model Comparison Summary

results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, dt_model.predict(X_test)),
        accuracy_score(y_test, rf_model.predict(X_test))
    ],
    "Precision": [
        precision_score(y_test, dt_model.predict(X_test)),
        precision_score(y_test, rf_model.predict(X_test))
    ],
    "Recall": [
        recall_score(y_test, dt_model.predict(X_test)),
        recall_score(y_test, rf_model.predict(X_test))
    ],
    "F1-Score": [
        f1_score(y_test, dt_model.predict(X_test)),
        f1_score(y_test, rf_model.predict(X_test))
    ]
})

print("\nMODEL COMPARISON")
print(results.round(4).to_string(index=False))
# Step 17: Classification Report

from sklearn.metrics import classification_report

print("\nRANDOM FOREST CLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        rf_model.predict(X_test),
        target_names=["Stay", "Churn"]
    )
)
# Step 18: Sample Customer Prediction

sample_customer = X_test.iloc[[0]]

prediction = rf_model.predict(sample_customer)
probability = rf_model.predict_proba(sample_customer)[0][1]

print("\nSAMPLE CUSTOMER PREDICTION")

if prediction[0] == 1:
    print("Prediction: Customer is likely to CHURN")
else:
    print("Prediction: Customer is likely to STAY")

print(f"Churn Probability: {probability:.2%}")
# Verify saved model

import joblib

loaded_model = joblib.load("customer_churn_model.pkl")

print("\nSaved model loaded successfully!")
print("Model type:", type(loaded_model).__name__)