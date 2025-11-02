import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# --------------------------
# Custom CSS (Professional UI)
# --------------------------
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 8px 20px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        padding: 18px;
        background-color: #262730;
        border-radius: 10px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

st.title("❤️ Heart Disease Prediction App")
st.write("A machine learning model that estimates heart disease probability from medical variables.")

# --------------------------
# Load Model + Scaler
# --------------------------
MODEL_PATH = "random_forest_heart_disease_model.joblib"
SCALER_PATH = "standard_scaler.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("📝 Enter Patient Information")

input_data = {}

# Numeric inputs
input_data['age'] = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
input_data['trestbps'] = st.sidebar.number_input("Resting BP", min_value=80, max_value=200, value=120)
input_data['chol'] = st.sidebar.number_input("Cholesterol", min_value=100, max_value=600, value=200)
input_data['thalach'] = st.sidebar.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
input_data['oldpeak'] = st.sidebar.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Categorical inputs
input_data['sex'] = st.sidebar.selectbox(
    "Sex", options=[0,1],
    format_func=lambda x: "Female" if x==0 else "Male"
)
input_data['cp'] = st.sidebar.selectbox(
    "Chest Pain Type", options=[0,1,2,3],
    format_func=lambda x: ["Typical angina","Atypical angina","Non-anginal pain","Asymptomatic"][x]
)
input_data['fbs'] = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0,1])
input_data['restecg'] = st.sidebar.selectbox("Resting ECG", options=[0,1,2])
input_data['exang'] = st.sidebar.selectbox("Exercise Induced Angina", options=[0,1])
input_data['slope'] = st.sidebar.selectbox("ST Slope", options=[0,1,2])
input_data['ca'] = st.sidebar.selectbox("Number of Major Vessels (0-3)", options=[0,1,2,3])
input_data['thal'] = st.sidebar.selectbox(
    "Thalassemia", options=[0,1,2],
    format_func=lambda x: ["Normal","Fixed Defect","Reversible Defect"][x]
)

# Ensure proper order for model
feature_order = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# --------------------------
# Prediction Logic
# --------------------------
def generate_pdf(pred, proba, input_dict):
    filename = f"heart_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join("/tmp", filename)

    c = canvas.Canvas(filepath)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Heart Disease Prediction Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
    c.drawString(50, 750, f"Probability: {proba:.4f}")
    c.drawString(50, 730, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawString(50, 700, "Input Data:")
    y = 680
    for k, v in input_dict.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 20

    c.save()
    return filepath


def log_prediction(input_dict, pred, proba):
    log_row = input_dict.copy()
    log_row["prediction"] = pred
    log_row["probability"] = proba
    log_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_file = "prediction_logs.csv"
    df_row = pd.DataFrame([log_row])

    if os.path.exists(log_file):
        df_row.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_row.to_csv(log_file, index=False)

# --------------------------
# Predict Button
# --------------------------
if st.sidebar.button("🔍 Predict"):
    df = pd.DataFrame([input_data], columns=feature_order)

    scaled = scaler.transform(df)
    proba = model.predict_proba(scaled)[0][1]
    pred = int(proba > 0.5)

    # Log prediction
    log_prediction(input_data, pred, proba)

    # Show results
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.subheader("✅ Prediction Result")
    st.write(f"**Prediction:** {'❤️ Heart Disease' if pred == 1 else '✅ No Heart Disease'}")
    st.write(f"**Probability:** {proba:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Generate PDF
    pdf_path = generate_pdf(pred, proba, input_data)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download PDF Report",
            data=f,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf"
        )
