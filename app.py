import streamlit as st
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found. Make sure xgb_model.pkl is in the app directory.")
        return None

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("telecom_churn.csv")

df = load_data()

# Streamlit page config
st.set_page_config(page_title="DropAlertAI", layout="wide")

# Styling
st.markdown("""
    <style>
    .stApp { background-color: #1e1e1e; color: white; }
    label, .css-1cpxqw2, .css-1p05t8e, .css-1y4p8pa { color: white !important; }
    .stTabs [role="tab"] {
        background-color: #333; color: white;
        padding: 10px; border-radius: 8px 8px 0 0;
    }
    .stTabs [role="tab"]:hover { background-color: #444; }
    .stTabs [role="tab"][aria-selected="true"] { background-color: #555; }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 DropAlertAI - Telecom Churn Predictor")
tab1, tab2 = st.tabs(["📊 Dashboard", "🔍 Predict"])

# Dashboard
with tab1:
    st.header("Churn Overview")
    churn_counts = df['Churn'].value_counts()
    labels = ['No Churn', 'Churn']
    values = [churn_counts[0], churn_counts[1]]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig)

# Prediction
with tab2:
    st.subheader("Predict Churn")

    account_weeks = st.slider("Account Weeks", 0, 300, 100)
    contract_renewal = st.selectbox("Contract Renewal", ["Yes", "No"])
    data_plan = st.selectbox("Data Plan", ["Yes", "No"])
    data_usage = st.number_input("Data Usage (GB)", min_value=0.0, format="%.2f")
    cust_serv_calls = st.number_input("Customer Service Calls", min_value=0)
    day_mins = st.number_input("Day Minutes", min_value=0.0, format="%.2f")
    day_calls = st.number_input("Day Calls", min_value=0)
    monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, format="%.2f")
    overage_fee = st.number_input("Overage Fee ($)", min_value=0.0, format="%.2f")
    roam_mins = st.number_input("Roaming Minutes", min_value=0.0, format="%.2f")

    if st.button("Predict") and model:
        input_data = [[
            account_weeks,
            1 if contract_renewal == "Yes" else 0,
            1 if data_plan == "Yes" else 0,
            data_usage,
            cust_serv_calls,
            day_mins,
            day_calls,
            monthly_charge,
            overage_fee,
            roam_mins
        ]]
        result = model.predict(input_data)[0]
        st.success("❌ Customer will churn" if result == 1 else "✅ Customer will not churn")
