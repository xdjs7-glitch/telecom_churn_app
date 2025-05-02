import streamlit as st 
import pickle
import numpy as np

# Load model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("📞 Telecom Churn Predictor")
st.write("Fill in the customer info to predict if they will churn.")

# User Inputs
account_weeks = st.slider("Account Weeks", 1, 300, 100)
contract_renewal = st.selectbox("Contract Renewal", ["Yes", "No"])
data_plan = st.selectbox("Has Data Plan", ["Yes", "No"])
data_usage = st.number_input("Data Usage (GB)", 0.0, 10.0, 2.0)
custserv_calls = st.slider("Customer Service Calls", 0, 10, 1)
day_mins = st.number_input("Daytime Minutes", 0.0, 500.0, 200.0)
day_calls = st.slider("Daytime Calls", 0, 200, 100)
monthly_charge = st.number_input("Monthly Charge ($)", 0.0, 200.0, 70.0)
overage_fee = st.number_input("Overage Fee ($)", 0.0, 20.0, 5.0)
roam_mins = st.number_input("Roaming Minutes", 0.0, 20.0, 5.0)

# Convert inputs to numerical values
contract_renewal = 1 if contract_renewal == "Yes" else 0
data_plan = 1 if data_plan == "Yes" else 0



# Combine into input array
input_data = np.array([[account_weeks, contract_renewal, data_plan, data_usage,
                        custserv_calls, day_mins, day_calls, monthly_charge,
                        overage_fee, roam_mins]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "⚠️ Customer is likely to churn." if prediction[0] == 1 else "✅ Customer is not likely to churn."
    st.success(result)
