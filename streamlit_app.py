import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard (ML Powered)",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL ARTIFACTS
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("churn_model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features


model, scaler, model_features = load_model()

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn_cleaned.csv")
    return df


df = load_data()

# ---------------------------------------------------
# FEATURE ENGINEERING (SAME AS TRAINING)
# ---------------------------------------------------
reference_date = pd.to_datetime(df['Last_Login_Date']).max()
df['Recency_Days'] = (
    reference_date - pd.to_datetime(df['Last_Login_Date'])
).dt.days

df['Avg_Monthly_Spend'] = df['Total_Revenue'] / df['Tenure_Months']
df['Engagement_Score'] = df['Usage_Frequency'] - df['Support_Tickets']

df['Customer_Value'] = pd.cut(
    df['Total_Revenue'],
    bins=[0, 10000, 30000, df['Total_Revenue'].max()],
    labels=['Low', 'Medium', 'High']
)

# ---------------------------------------------------
# PREPARE MODEL DATA
# ---------------------------------------------------
df_model = df.drop(
    columns=['Customer_ID', 'Signup_Date', 'Last_Login_Date']
)

df_model = pd.get_dummies(
    df_model,
    columns=['Gender', 'City', 'Subscription_Type', 'Customer_Value'],
    drop_first=True
)

# Align columns with training
df_model = df_model.reindex(columns=model_features, fill_value=0)

# Scale numerical columns
scale_cols = [
    'Age', 'Monthly_Charges', 'Tenure_Months',
    'Total_Revenue', 'Usage_Frequency', 'Support_Tickets',
    'Recency_Days', 'Avg_Monthly_Spend', 'Engagement_Score'
]

df_model[scale_cols] = scaler.transform(df_model[scale_cols])

# ---------------------------------------------------
# PREDICT CHURN PROBABILITY
# ---------------------------------------------------
df['Churn_Probability'] = model.predict_proba(df_model)[:, 1]

# Risk segmentation
def risk_label(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"


df['Risk_Category'] = df['Churn_Probability'].apply(risk_label)

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.title("🔍 Filters")

city_filter = st.sidebar.multiselect(
    "City",
    df['City'].unique(),
    default=df['City'].unique()
)

plan_filter = st.sidebar.multiselect(
    "Subscription Type",
    df['Subscription_Type'].unique(),
    default=df['Subscription_Type'].unique()
)

risk_filter = st.sidebar.multiselect(
    "Risk Category",
    ["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)

df_filt = df[
    (df['City'].isin(city_filter)) &
    (df['Subscription_Type'].isin(plan_filter)) &
    (df['Risk_Category'].isin(risk_filter))
]

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("📊 ML-Powered Customer Churn Dashboard")
st.markdown("### Real-time churn prediction using Machine Learning")

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------
st.markdown("## 🔑 Key Business KPIs")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Total Customers", df_filt.shape[0])
col2.metric("Active Customers", df_filt[df_filt['Churn_Flag']==0].shape[0])
col3.metric("Churned Customers", df_filt[df_filt['Churn_Flag']==1].shape[0])
col4.metric("Churn Rate (%)", f"{df_filt['Churn_Flag'].mean()*100:.2f}%")
col5.metric(
    "Revenue Lost",
    f"₹{df_filt[df_filt['Churn_Flag']==1]['Total_Revenue'].sum():,}"
)
col6.metric(
    "Revenue at Risk",
    f"₹{df_filt[df_filt['Churn_Probability']>=0.7]['Total_Revenue'].sum():,}"
)

# ---------------------------------------------------
# CHURN DISTRIBUTION
# ---------------------------------------------------
st.markdown("---")
st.markdown("## 📉 Churn Distribution")

fig = px.pie(
    df_filt,
    names='Churn_Flag',
    title="Active vs Churned Customers",
    color_discrete_sequence=["#2ECC71", "#E74C3C"]
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# CHURN DRIVERS
# ---------------------------------------------------
st.markdown("---")
st.markdown("## 🧠 Churn Drivers")

c1, c2, c3 = st.columns(3)

c1.plotly_chart(
    px.box(df_filt, x='Churn_Flag', y='Usage_Frequency',
           title="Usage Frequency vs Churn"),
    use_container_width=True
)

c2.plotly_chart(
    px.box(df_filt, x='Churn_Flag', y='Support_Tickets',
           title="Support Tickets vs Churn"),
    use_container_width=True
)

c3.plotly_chart(
    px.box(df_filt, x='Churn_Flag', y='Tenure_Months',
           title="Tenure vs Churn"),
    use_container_width=True
)

# ---------------------------------------------------
# HIGH RISK CUSTOMERS
# ---------------------------------------------------
st.markdown("---")
st.markdown("## 🚨 High-Risk Customers (ML Identified)")

st.dataframe(
    df_filt[df_filt['Risk_Category']=="High"][
        ['Customer_ID', 'City', 'Subscription_Type',
         'Churn_Probability', 'Usage_Frequency',
         'Support_Tickets', 'Total_Revenue']
    ],
    use_container_width=True
)

# ---------------------------------------------------
# BUSINESS INSIGHTS
# ---------------------------------------------------
st.markdown("---")
st.markdown("## 📌 Key Business Insights")

st.markdown("""
- ML model identifies **inactivity, low engagement, and high support tickets** as top churn drivers  
- **Basic plan customers** show consistently higher churn risk  
- **Revenue at risk** enables prioritization of retention campaigns  
- This dashboard allows **proactive churn prevention**, not reactive reporting  
""")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("✅ ML-powered churn prediction dashboard – ready for business decision-making")
