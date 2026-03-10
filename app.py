import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("purchase_model.pkl")

st.set_page_config(page_title="Purchase Prediction App", layout="centered")
st.title("Online Purchase Intent Prediction")
st.write("Enter session details to predict whether a customer will make a purchase.")

# ---------------------- User Inputs ----------------------

admin_visits = st.number_input("Administrative Visits", min_value=0, value=0)
admin_time = st.number_input("Time on Administrative Pages", min_value=0.0, value=0.0)

info_visits = st.number_input("Informational Visits", min_value=0, value=0)
info_time = st.number_input("Time on Informational Pages", min_value=0.0, value=0.0)

product_visits = st.number_input("Product Page Visits", min_value=0, value=0)
product_time = st.number_input("Time on Product Pages", min_value=0.0, value=0.0)

bounce_rate = st.number_input("Bounce Rate", min_value=0.0, max_value=1.0, value=0.0)
exit_rate = st.number_input("Exit Rate", min_value=0.0, max_value=1.0, value=0.0)
page_value = st.number_input("Page Value Score", min_value=0.0, value=0.0)
special_day_score = st.number_input(
    "Special Day Score", min_value=0.0, max_value=1.0, value=0.0
)

month = st.selectbox("Month (1-12)", list(range(1, 13)))
operating_system = st.selectbox("Operating System", list(range(1, 9)))
browser = st.selectbox("Browser Type", list(range(1, 12)))
region = st.selectbox("Region", list(range(1, 10)))
traffic_type = st.selectbox("Traffic Type", list(range(1, 21)))

visitor_type = st.selectbox(
    "Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"]
)
weekend_flag = st.selectbox("Weekend Visit (0 = No, 1 = Yes)", [0, 1])

# Derived feature (must match training)
total_time_spent = admin_time + info_time + product_time

# ---------------------- DataFrame for Model ----------------------

input_record = pd.DataFrame(
    {
        "Administrative": [admin_visits],
        "Administrative_Duration": [admin_time],
        "Informational": [info_visits],
        "Informational_Duration": [info_time],
        "ProductRelated": [product_visits],
        "ProductRelated_Duration": [product_time],
        "BounceRates": [bounce_rate],
        "ExitRates": [exit_rate],
        "PageValues": [page_value],
        "SpecialDay": [special_day_score],
        "Month": [month],
        "OperatingSystems": [operating_system],
        "Browser": [browser],
        "Region": [region],
        "TrafficType": [traffic_type],
        "VisitorType": [visitor_type],
        "Weekend": [weekend_flag],
        "TotalTimeSpent": [total_time_spent],
    }
)

# ---------------------- Prediction ----------------------

if st.button("Predict"):
    result = model.predict(input_record)[0]

    if result == 1:
        st.success("The customer is likely to make a purchase.")
    else:
        st.info("The customer is unlikely to make a purchase.")
