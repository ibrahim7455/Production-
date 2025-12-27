from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ================================
# Load dataset and preprocess
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")

    if 'customerid' in df.columns:
        df.drop(columns=['customerid'], inplace=True)

    # FIX: churn to numeric
    if df['churn'].dtype == 'object':
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

    le = LabelEncoder()
    categorical_cols = [
        'gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streamingtv', 'streamingmovies', 'paperlessbilling'
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    num_cols = ["tenure", "monthlycharges", "totalcharges"]
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop('churn', axis=1)
    y = df['churn']

    return df, X, y


# ================================
# Load model
# ================================
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

# ================================
# Analysis Page
# ================================
def analysis_page(df):
    st.title("📊 Telecom Churn Analysis Dashboard")

    # ---------- KPIs ----------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{df['churn'].mean()*100:.2f}%")
    col3.metric("Avg Monthly Charges", f"{df['monthlycharges'].mean():.2f}")

    st.divider()

    # ---------- Charts ----------
    st.subheader("Payment Method Distribution")
    payment_counts = df['paymentmethod'].value_counts().reset_index()
    payment_counts.columns = ['Payment Method', 'Count']  # rename columns
    fig = px.bar(
        payment_counts,
        x='Payment Method',
        y='Count',
        title="Payment Method Count"
)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Contract Types")
    fig = px.pie(
        df,
        names='contract',
        title="Contract Types Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn Distribution")
    fig = px.pie(
        df,
        names='churn',
        title="Churn Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Contract vs Churn")
    fig = px.histogram(
        df,
        x='contract',
        color='churn',
        barmode='group',
        title="Contract vs Churn"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Charges Distribution")
    fig = px.histogram(
        df,
        x='monthlycharges',
        nbins=30,
        title="Monthly Charges Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    cols = [
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'paperlessbilling',
        'paymentmethod'
    ]

    for col in cols:
        st.subheader(f"{col} vs Churn")
        fig = px.histogram(
            df,
            x=col,
            color='churn',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

# ================================
# Prediction Page
# ================================
def prediction_page():
    st.title("📉 Customer Churn Prediction")

    df, X, y = load_data()
    model = load_model()

    gender = st.selectbox("Gender", ["Male", "Female"])
    seniorcitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", 0, 100, 10)
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    multiplelines = st.selectbox("Multiple Lines", ["Yes", "No"])
    onlinesecurity = st.selectbox("Online Security", ["Yes", "No"])
    onlinebackup = st.selectbox("Online Backup", ["Yes", "No"])
    deviceprotection = st.selectbox("Device Protection", ["Yes", "No"])
    techsupport = st.selectbox("Tech Support", ["Yes", "No"])
    streamingtv = st.selectbox("Streaming TV", ["Yes", "No"])
    streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No"])
    paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthlycharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    totalcharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
    internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paymentmethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Credit card (automatic)",
            "Bank transfer (automatic)",
            "Manual"
        ]
    )

    input_data = {
        "gender": gender,
        "seniorcitizen": 1 if seniorcitizen == "Yes" else 0,
        "partner": 1 if partner == "Yes" else 0,
        "dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "phoneservice": 1 if phoneservice == "Yes" else 0,
        "multiplelines": 1 if multiplelines == "Yes" else 0,
        "onlinesecurity": 1 if onlinesecurity == "Yes" else 0,
        "onlinebackup": 1 if onlinebackup == "Yes" else 0,
        "deviceprotection": 1 if deviceprotection == "Yes" else 0,
        "techsupport": 1 if techsupport == "Yes" else 0,
        "streamingtv": 1 if streamingtv == "Yes" else 0,
        "streamingmovies": 1 if streamingmovies == "Yes" else 0,
        "paperlessbilling": 1 if paperlessbilling == "Yes" else 0,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,
        "internetservice": internetservice,
        "contract": contract,
        "paymentmethod": paymentmethod,
    }

    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=X.columns, fill_value=0)

    if st.button("Predict Churn"):
        pred = model.predict(df_input)[0]
        if pred == 1:
            st.error("❌ Customer is likely to churn")
        else:
            st.success("✅ Customer is not likely to churn")

# ================================
# Main
# ================================
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Analysis", "Prediction"])

    df, _, _ = load_data()

    if page == "Analysis":
        analysis_page(df)
    else:
        prediction_page()

if __name__ == "__main__":
    main()
