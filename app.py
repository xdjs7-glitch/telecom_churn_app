import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="DropAlertAI", layout="wide")

# Load model
model = joblib.load('model_telecom/rf_model.pkl')

# Load data
df = pd.read_csv('telecom_churn.csv')

# Custom CSS for white labels and dark background
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
    }
    .stTextInput > div > div > input, .stSelectbox > div > div {
        color: white !important;
    }
    label, .stSlider, .stRadio label {
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Tabs for navigation
tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Predict"])

# ----------------- Dashboard -----------------
with tab1:
    st.title("📊 Dashboard - DropAlertAI")
    
    churn_counts = df['Churn'].value_counts()
    churn_labels = ['No Churn', 'Churn']
    churn_values = [churn_counts[0], churn_counts[1]]
    churn_colors = ['green', 'red']

    # Pie Chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=churn_labels,
        values=churn_values,
        hole=0.5,
        marker=dict(colors=churn_colors),
        textinfo='label+percent',
        insidetextfont=dict(color='white', size=12),
    )])
    fig_pie.update_layout(
        title="Churn Distribution",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350, width=350,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_pie, use_container_width=False)

    # Bar Chart: Mean account weeks per churn status
    avg_weeks = df.groupby('Churn')['Account Weeks'].mean()
    fig_bar = go.Figure(data=[
        go.Bar(x=['No Churn', 'Churn'], y=avg_weeks, marker_color=['green', 'red'])
    ])
    fig_bar.update_layout(
        title="Average Account Weeks by Churn",
        xaxis_title="Churn Status",
        yaxis_title="Avg Account Weeks",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ----------------- Predict -----------------
with tab2:
    st.title("🧠 Predict Churn")

    st.markdown("### Enter customer information:")

    # Inputs
    account_weeks = st.slider("Account Weeks", min_value=1, max_value=300, value=100)
    contract_renewal = st.selectbox("Contract Renewal", ['Yes', 'No'])
    data_plan = st.selectbox("Data Plan", ['Yes', 'No'])
    data_usage = st.number_input("Data Usage (GB)", min_value=0.0, value=1.0)
    cust_care_calls = st.number_input("Customer Care Calls", min_value=0, value=1)
    day_mins = st.number_input("Day Minutes", min_value=0.0, value=100.0)
    day_calls = st.number_input("Day Calls", min_value=0, value=50)
    monthly_charge = st.number_input("Monthly Charge", min_value=0.0, value=50.0)
    overage_fee = st.number_input("Overage Fee", min_value=0.0, value=5.0)
    roaming_mins = st.number_input("Roaming Minutes", min_value=0.0, value=0.0)

    # Convert Yes/No to 1/0
    contract_renewal = 1 if contract_renewal == 'Yes' else 0
    data_plan = 1 if data_plan == 'Yes' else 0

    if st.button("Predict"):
        input_data = [[
            account_weeks, contract_renewal, data_plan, data_usage,
            cust_care_calls, day_mins, day_calls, monthly_charge,
            overage_fee, roaming_mins
        ]]
        prediction = model.predict(input_data)[0]

        st.subheader("🔍 Prediction:")
        if prediction == 1:
            st.error("❌ This customer is **likely to churn**.")
        else:
            st.success("✅ This customer is **not likely to churn**.")

    # Hide Streamlit default menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
