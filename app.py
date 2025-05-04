import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# Load model
model = joblib.load("xgb_model.pkl")

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
# ========== Dashboard ==========
with tab1:
    st.title("📊 Dashboard - DropAlertAI")

    churn_counts = df['Churn'].value_counts()
    churn_labels = ['No Churn', 'Churn']
    churn_values = [churn_counts[0], churn_counts[1]]
    churn_colors = ['purple', 'pink']

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=churn_labels,
            values=churn_values,
            hole=0.5,
            marker=dict(colors=churn_colors),
            textinfo='label+percent',
            insidetextfont=dict(color='white', size=14),
            outsidetextfont=dict(color='white', size=14)
        )])

        fig_pie.update_layout(
            title=dict(
                text="Churn Distribution",
                font=dict(color='white', size=20)
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            width=400,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(
                font=dict(color='white')
            )
        )

        st.plotly_chart(fig_pie, use_container_width=False)

    with col2:
        st.markdown(
            """
            <div style="color: white; font-size: 16px; padding-top: 50px;">
                <p><strong>Insight:</strong></p>
                <p>According to our dataset, <strong>85.5%</strong> (<strong>2,850</strong>) of customers have continued their subscription, indicating a strong level of customer retention.</p>
                <p>In contrast, <strong>14.5%</strong> (<strong>483</strong>) of customers have churned, highlighting a significant portion that opted to discontinue the service.</p>
                <p>This distribution underscores the importance of identifying key factors contributing to churn and developing targeted strategies to enhance customer satisfaction and loyalty.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


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

    # Style the Predict button
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            color: black !important;
            background-color: #f0f0f0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hide Streamlit default menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
