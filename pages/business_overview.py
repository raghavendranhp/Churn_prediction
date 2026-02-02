import streamlit as st
import pandas as pd
import plotly.express as px

def show():
    st.title(" Business Health & Churn Overview")
    st.markdown("### Business Health → Churn Problem → Why It Happens → Who Is at Risk")

    df = pd.read_csv("customer_churn_cleaned.csv")

    #KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", df.shape[0])
    col2.metric("Churn Rate (%)", f"{df['Churn_Flag'].mean()*100:.2f}%")
    col3.metric(
        "Revenue Lost",
        f"₹{df[df['Churn_Flag']==1]['Total_Revenue'].sum():,}"
    )

    #Churn distribution
    fig = px.pie(
        df,
        names="Churn_Flag",
        title="Active vs Churned Customers",
        color_discrete_sequence=["#2ECC71", "#E74C3C"]
    )
    st.plotly_chart(fig, use_container_width=True)

    #Why churn happens
    c1, c2, c3 = st.columns(3)

    c1.plotly_chart(
        px.box(df, x="Churn_Flag", y="Usage_Frequency",
               title="Usage Frequency vs Churn"),
        use_container_width=True
    )

    c2.plotly_chart(
        px.box(df, x="Churn_Flag", y="Support_Tickets",
               title="Support Tickets vs Churn"),
        use_container_width=True
    )

    c3.plotly_chart(
        px.box(df, x="Churn_Flag", y="Tenure_Months",
               title="Tenure vs Churn"),
        use_container_width=True
    )

    st.subheader("🔹 Who Is at Risk")
    st.markdown("""
    - Low usage customers  
    - High support ticket customers  
    - New users with short tenure  
    - Basic subscription users  
    """)
