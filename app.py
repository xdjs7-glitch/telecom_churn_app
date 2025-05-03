import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load your saved model
model = joblib.load('rf_model.pkl')

# Configure Streamlit page
st.set_page_config(page_title="DropAlertAI", layout="wide")

# Apply custom dark theme styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #1e1e1e;
            color: white;
        }
        .css-1d391kg { color: white; }
        .css-18ni7ap { color: white; }
        .stButton > button {
            background-color: #444444;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("DropAlertAI")
page = st.sidebar.radio("Go to", ["Predict", "Dashboard"])

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("telecom_churn.csv")  # make sure this is in the same directory or repo

df = load_data()

# ========== PAGE: PREDICT ==========
if page == "Predict":
    st.title("📉 Churn Prediction")
    st.markdown("Enter customer details to predict churn.")

    # Input fields (basic example — adapt to your model's features)
    account_weeks = st.number_input("Account Weeks", min_value=0)
    contract_renewal = st.selectbox("Contract Renewal", [0, 1])
    data_plan = st.selectbox("Data Plan", [0, 1])
    data_usage = st.number_input("Data Usage (GB)", min_value=0.0)
    cust_serv_calls = st.number_input("Customer Service Calls", min_value=0)
    day_mins = st.number_input("Day Minutes", min_value=0.0)
    day_calls = st.number_input("Day Calls", min_value=0)
    monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0)
    overage_fee = st.number_input("Overage Fee ($)", min_value=0.0)
    roam_mins = st.number_input("Roaming Minutes", min_value=0.0)

    if st.button("Predict Churn"):
        features = [[
            account_weeks, contract_renewal, data_plan, data_usage,
            cust_serv_calls, day_mins, day_calls, monthly_charge,
            overage_fee, roam_mins
        ]]
        prediction = model.predict(features)
        result = "❌ Customer will churn" if prediction[0] == 1 else "✅ Customer will not churn"
        st.success(result)

# ========== PAGE: DASHBOARD ==========
elif page == "Dashboard":
    st.title("📊 Churn Insights Dashboard")

    # Churn distribution pie chart
    churn_counts = df['Churn'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(
        churn_counts,
        labels=["No Churn", "Churn"],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff', '#ff6666']
    )
    ax1.set_title("Churn Distribution")
    st.pyplot(fig1)

    # Customer Service Calls vs Churn
    st.subheader("📞 Customer Service Calls vs Churn")
    fig2, ax2 = plt.subplots()
    sns.barplot(x="CustServCalls", y="Churn", data=df, palette="rocket", ax=ax2)
    st.pyplot(fig2)

    # Data Usage by Churn
    st.subheader("📶 Data Usage by Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Churn", y="DataUsage", data=df, palette="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # Monthly Charge Distribution
    st.subheader("💳 Monthly Charges by Churn")
    fig4, ax4 = plt.subplots()
    sns.histplot(data=df, x="MonthlyCharge", hue="Churn", multiple="stack", kde=True, ax=ax4)
    st.pyplot(fig4)

    # Hide Streamlit default menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
