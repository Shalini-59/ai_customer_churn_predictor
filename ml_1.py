# =====================================================
# CUSTOMER CHURN PREDICTION PROJECT - FIXED VERSION
# =====================================================

# Step 1: Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load dataset

file_path = r"C:\Users\shali\ML Project\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!\n")


# Step 3: Preview dataset

print(df.head())


# Step 4: Dataset info

print("\nDataset info:")
print(df.info())


# Step 5: Drop customerID column

df = df.drop("customerID", axis=1)


# Step 6: Convert TotalCharges to numeric safely

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# Step 7: Fill missing values correctly (FIXED)

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


# Step 8: Convert categorical columns properly (FIXED)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for column in df.select_dtypes(include=["object"]).columns:
    df[column] = le.fit_transform(df[column])


print("\nEncoding completed successfully!")


# Step 9: Define features and target

X = df.drop("Churn", axis=1)
y = df["Churn"]


# Step 10: Train-test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nDataset split completed!")


# Step 11: Train Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

print("\nDecision Tree trained successfully!")


# Step 12: Decision Tree prediction

dt_pred = dt_model.predict(X_test)


# Step 13: Accuracy check

from sklearn.metrics import accuracy_score

dt_acc = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", dt_acc)


# Step 14: Train Random Forest model

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

print("\nRandom Forest trained successfully!")


# Step 15: Random Forest prediction

rf_pred = rf_model.predict(X_test)


# Step 16: Random Forest accuracy

rf_acc = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_acc)


# Step 17: Classification report

from sklearn.metrics import classification_report

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))


# Step 18: Confusion matrix

from sklearn.metrics import confusion_matrix

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, rf_pred))


# Step 19: Feature importance graph

importance = pd.Series(rf_model.feature_importances_, index=X.columns)

importance.sort_values(ascending=False).head(10).plot(
    kind="bar",
    title="Top 10 Important Features Affecting Churn"
)

plt.show()


# =====================================================
# Step 20: SAMPLE TESTING
# =====================================================

sample_customer = X_test.iloc[0].values.reshape(1, -1)

prediction = rf_model.predict(sample_customer)

print("\nSample Customer Prediction:")

if prediction[0] == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is likely to STAY")
  
rf_model.fit(X_train, y_train)

# Save trained model
import joblib
joblib.dump(rf_model, "customer_churn_model.pkl")

print("Model saved successfully!")