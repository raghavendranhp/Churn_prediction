import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


selected = option_menu(
        menu_title="Churn Intelligence",
        options=[
            "Business Overview",
            "Model Prediction",
            "Action Plan"
        ],
        icons=["bar-chart", "cpu", "lightbulb"],
        menu_icon="menu-button-wide",
        default_index=0,orientation="horizontal",
        styles={"nav-link": {"font-size": "18px", "text-align": "left", "margin": "-2px", "--hover-color": "#FF5A5F"},
                                   "nav-link-selected": {"background-color": "#6495ED"}}
    )

#ROUTING
if selected == "Business Overview":
    from pages.business_overview import show
    show()

elif selected == "Model Prediction":
    from pages.model_prediction import show
    show()

elif selected == "Action Plan":
    from pages.action_plan import show
    show()
