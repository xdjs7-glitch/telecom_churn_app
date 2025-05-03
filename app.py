import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("rf_model.pkl")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("telecom_churn.csv")

df = load_data()

# Page config
st.set_page_config(page_title="DropAlertAI", layout="wide")

# Custom dark theme styling
st.markdown("""
    <style>
        .stApp { background-color: #1e1e1e; color: white; }
        label, .css-1cpxqw2, .css-1p05t8e, .css-1y4p8pa { color: white !important; }
        .stTabs [role="tab"] {
            background-color: #333;
            color: white;
            padding: 10px;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [role="tab"]:hover {
            background-color: #444;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #555;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("🚨 DropAlertAI - Telecom Churn Predictor")

# Create tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🔍 Predict"])

# ========== Dashboard ==========
with tab1:
    st.subheader("📊 Churn Insights Dashboard")

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

    st.subheader("📞 Customer Service Calls vs Churn")
    fig2, ax2 = plt.subplots()
    sns.barplot(x="CustServCalls", y="Churn", data=df, palette="rocket", ax=ax2)
    st.pyplot(fig2)

    st.subheader("📶 Data Usage by Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Churn", y="DataUsage", data=df, palette="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.subheader("💳 Monthly Charges by Churn")
    fig4, ax4 = plt.subplots()
    sns.histplot(data=df, x="MonthlyCharge", hue="Churn", multiple="stack", kde=True, ax=ax4)
    st.pyplot(fig4)

# ========== Predict ==========
with tab2:
    st.subheader("🔍 Predict Churn")

    # Input fields
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

    if st.button("Predict"):
        features = [[
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

        prediction = model.predict(features)
        st.success("❌ Customer will churn" if prediction[0] == 1 else "✅ Customer will not churn")

    # Hide Streamlit default menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
