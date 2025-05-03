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
import plotly.graph_objects as go

with tab1:
    st.subheader("📊 Churn Insights Dashboard")

    # Plotly Donut Chart
    churn_counts = df['Churn'].value_counts()
    labels = ["No Churn", "Churn"]
    values = [churn_counts[0], churn_counts[1]]

    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker=dict(colors=["green", "red"]),
            textinfo="percent+label",
            insidetextorientation="radial"
        )]
    )

    fig.update_layout(
        width=250,
        height=250,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(color="white", size=12)
    )

    st.plotly_chart(fig, use_container_width=False)


    # Bar plot with Plotly


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
