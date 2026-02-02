import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("churn_model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features


model, scaler, model_features = load_model()


# --------------------------------------------------
# COMMON FEATURE ENGINEERING FUNCTION
# --------------------------------------------------
def preprocess(df):
    # Feature engineering
    ref_date = pd.to_datetime(df['Last_Login_Date']).max()
    df['Recency_Days'] = (ref_date - pd.to_datetime(df['Last_Login_Date'])).dt.days
    df['Avg_Monthly_Spend'] = df['Total_Revenue'] / df['Tenure_Months']
    df['Engagement_Score'] = df['Usage_Frequency'] - df['Support_Tickets']

    # Drop non-features
    df_model = df.drop(columns=['Customer_ID', 'Signup_Date', 'Last_Login_Date'])

    # Encoding
    df_model = pd.get_dummies(
        df_model,
        columns=['Gender', 'City', 'Subscription_Type'],
        drop_first=True
    )

    # Align with training features
    df_model = df_model.reindex(columns=model_features, fill_value=0)

    # Scaling
    scale_cols = [
        'Age','Monthly_Charges','Tenure_Months','Total_Revenue',
        'Usage_Frequency','Support_Tickets',
        'Recency_Days','Avg_Monthly_Spend','Engagement_Score'
    ]
    df_model[scale_cols] = scaler.transform(df_model[scale_cols])

    return df_model


# --------------------------------------------------
# PAGE UI
# --------------------------------------------------
def show():
    st.title(" Churn Prediction (ML Model)")
    st.markdown("Predict churn using **manual input** or **CSV upload**.")

    tab1, tab2 = st.tabs(["🧍 Single Customer", "📂 Bulk Prediction"])

    # ==================================================
    # TAB 1: MANUAL INPUT
    # ==================================================
    with tab1:
        st.subheader("🔹 Single Customer Prediction")

        with st.form("manual_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.slider("Age", 18, 70, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                city = st.selectbox(
                    "City",
                    ["Bangalore","Hyderabad","Delhi","Chennai",
                     "Pune","Mumbai","Kolkata"]
                )
                subscription = st.selectbox(
                    "Subscription Type",
                    ["Basic","Standard","Premium"]
                )

            with col2:
                monthly = st.number_input("Monthly Charges", 0, 2000, 499)
                tenure = st.slider("Tenure (Months)", 1, 72, 12)
                usage = st.slider("Usage Frequency", 0, 30, 5)
                tickets = st.slider("Support Tickets", 0, 20, 2)

            submit = st.form_submit_button("Predict Churn")

        if submit:
            total_revenue = monthly * tenure

            sample = pd.DataFrame([{
                'Customer_ID': 'NEW_CUSTOMER',
                'Age': age,
                'Gender': gender,
                'City': city,
                'Signup_Date': pd.Timestamp.today(),
                'Last_Login_Date': pd.Timestamp.today(),
                'Subscription_Type': subscription,
                'Monthly_Charges': monthly,
                'Tenure_Months': tenure,
                'Total_Revenue': total_revenue,
                'Usage_Frequency': usage,
                'Support_Tickets': tickets,
                'Churn_Flag': 0
            }])

            X_sample = preprocess(sample)
            prob = model.predict_proba(X_sample)[0][1]

            st.metric("Churn Probability", f"{prob:.2%}")

            if prob >= 0.7:
                st.error("🚨 High Risk of Churn")
            elif prob >= 0.4:
                st.warning("⚠️ Medium Risk of Churn")
            else:
                st.success("✅ Low Risk of Churn")

    # ==================================================
    # TAB 2: CSV UPLOAD
    # ==================================================
    with tab2:
        st.subheader("🔹 Bulk Prediction via CSV Upload")

        st.info("""
        Upload a CSV with the same columns as the cleaned dataset:
        Age, Gender, City, Signup_Date, Last_Login_Date,
        Subscription_Type, Monthly_Charges, Tenure_Months,
        Total_Revenue, Usage_Frequency, Support_Tickets
        """)

        file = st.file_uploader("Upload CSV File", type=["csv"])

        if file:
            df_upload = pd.read_csv(file)

            try:
                X_bulk = preprocess(df_upload)
                df_upload['Churn_Probability'] = model.predict_proba(X_bulk)[:, 1]

                st.success("Prediction completed successfully!")

                st.dataframe(
                    df_upload[['Customer_ID','Churn_Probability',
                               'Subscription_Type','City',
                               'Usage_Frequency','Support_Tickets']],
                    use_container_width=True
                )

                st.download_button(
                    "⬇️ Download Results",
                    df_upload.to_csv(index=False),
                    "churn_predictions.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error("Error processing file. Please check column names and data types.")
                st.exception(e)
