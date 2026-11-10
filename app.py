
from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "customer_churn_model.pkl")
DATASET_PATH = os.path.join(
    BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATASET_PATH)


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("predict.html")

    try:
        customer_data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        customer_df = pd.DataFrame([customer_data])

        prediction = model.predict(customer_df)[0]
        probabilities = model.predict_proba(customer_df)[0]

        churn_index = list(model.classes_).index(1)
        churn_probability = probabilities[churn_index] * 100

        if churn_probability >= 70:
            risk = "High"
            action = (
                "Prioritize this customer for retention support "
                "and review their service experience."
            )
        elif churn_probability >= 40:
            risk = "Medium"
            action = (
                "Monitor customer satisfaction and consider "
                "a suitable retention offer."
            )
        else:
            risk = "Low"
            action = (
                "Continue regular customer engagement "
                "and service monitoring."
            )

        result = {
            "prediction": (
                "Likely to Churn" if prediction == 1
                else "Likely to Stay"
            ),
            "probability": round(churn_probability, 2),
            "risk": risk,
            "action": action
        }

        return render_template("result.html", result=result)

    except (ValueError, KeyError) as error:
        return f"Invalid customer information: {error}", 400


@app.route("/customers")
def customers():

    total_customers = len(df)
    churned_customers = int((df["Churn"] == "Yes").sum())
    retained_customers = int((df["Churn"] == "No").sum())

    customer_records = df.head(100).to_dict(orient="records")

    return render_template(
        "customers.html",
        customers=customer_records,
        total_customers=total_customers,
        churned_customers=churned_customers,
        retained_customers=retained_customers
    )
@app.route("/performance")
def performance():
    return render_template("performance.html")


if __name__ == "__main__":
    app.run(debug=True)
