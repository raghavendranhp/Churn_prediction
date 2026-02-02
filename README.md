---

# Customer Churn Intelligence – Streamlit App

## Project Overview

This project focuses on analyzing customer churn and predicting the likelihood of customers leaving the service using machine learning.
It combines **data analysis, churn prediction, and business insights** into an interactive **Streamlit application**.

Due to system constraints with Power BI, the dashboard and model insights are implemented using **Streamlit**, while maintaining the same business logic and KPIs.

---

## Objectives

* Understand customer behavior and churn patterns
* Identify key factors contributing to churn
* Predict churn probability using a machine learning model
* Provide actionable business recommendations

---

##  Project Structure

```
Churn_prediction/
│
├── streamlit_app.py
├── customer_churn_cleaned.csv
├── report
sources/
│
├── customer_churn.xlsx
model/
│
├── churn_model_rf.pkl
├── scaler.pkl
├── model_features.pkl
│
├── pages/
│   ├── business_overview.py
│   ├── model_prediction.py
│   └── action_plan.py
│
└── README.md
```

---

## Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Plotly
* Joblib

---

## How to Run the Application

### 1️⃣ Install Dependencies

```bash
pip install streamlit streamlit-option-menu pandas plotly scikit-learn joblib
```

### 2️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📊 Application Pages

### 🔹 Business Overview

* Business health KPIs
* Churn distribution
* Key churn drivers
* Identification of at-risk customer segments

### 🔹 Model Prediction

* Churn probability prediction using a trained ML model
* Single-customer prediction using manual inputs
* Bulk prediction using CSV upload

### 🔹 Action Plan

* Data-driven churn reduction strategies
* Business recommendations
* Expected business impact

---

## Machine Learning Model

* Model: Random Forest Classifier
* Features include customer usage, tenure, support interactions, and subscription details
* Model output is **churn probability**, not just churn/no-churn

---

## Key Insights

* Low usage frequency and high support tickets strongly contribute to churn
* Basic subscription users show higher churn risk
* Early-stage customers require proactive engagement

---

## Notes

* Streamlit is used as a lightweight alternative to Power BI for dashboarding
* The solution is modular, scalable, and can be migrated to Power BI or Tableau later
* This project demonstrates an end-to-end churn analytics workflow

---

## 👤 Author

**Raghavendran**

---

