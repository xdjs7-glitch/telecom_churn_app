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

    # PIE CHART - Churn Distribution
    churn_counts = df['Churn'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(2.5, 2.5), dpi=100)
    wedges, texts, autotexts = ax1.pie(
        churn_counts,
        labels=["No Churn", "Churn"],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.3),
        textprops=dict(color="white", fontsize=8),
        colors=['#2ca02c', '#d62728']
    )
    ax1.set(aspect="equal")
    fig1.patch.set_alpha(0.0)
    st.pyplot(fig1, clear_figure=True)

    # BARPLOT - Customer Service Calls vs Churn
    st.markdown("#### 📞 Customer Service Calls vs Churn")
    fig2, ax2 = plt.subplots(figsize=(3, 2), dpi=100)
    sns.barplot(x="CustServCalls", y="Churn", data=df, palette="viridis", ax=ax2)
    ax2.set_xlabel("CustServCalls", fontsize=8)
    ax2.set_ylabel("Churn", fontsize=8)
    ax2.tick_params(axis='both', labelsize=7)
    fig2.tight_layout()
    fig2.patch.set_alpha(0.0)
    st.pyplot(fig2, clear_figure=True)

    # BOXPLOT - Data Usage by Churn
    st.markdown("#### 📶 Data Usage by Churn")
    fig3, ax3 = plt.subplots(figsize=(3, 2), dpi=100)
    sns.boxplot(x="Churn", y="DataUsage", data=df, palette="Set2", ax=ax3)
    ax3.set_xlabel("Churn", fontsize=8)
    ax3.set_ylabel("Data Usage", fontsize=8)
    ax3.tick_params(axis='both', labelsize=7)
    fig3.tight_layout()
    fig3.patch.set_alpha(0.0)
    st.pyplot(fig3, clear_figure=True)

    # HISTOGRAM - Monthly Charges by Churn
    st.markdown("#### 💳 Monthly Charges by Churn")
    fig4, ax4 = plt.subplots(figsize=(3, 2), dpi=100)
    sns.histplot(data=df, x="MonthlyCharge", hue="Churn", multiple="stack", kde=True, palette="coolwarm", ax=ax4)
    ax4.set_xlabel("Monthly Charge", fontsize=8)
    ax4.set_ylabel("Count", fontsize=8)
    ax4.tick_params(axis='both', labelsize=7)
    fig4.tight_layout()
    fig4.patch.set_alpha(0.0)
    st.pyplot(fig4, clear_figure=True)


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
