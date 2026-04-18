import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load hospital dataset
hospitals = pd.read_csv("hospital dataset.csv")

# Clean column names (IMPORTANT FIX)
hospitals.columns = hospitals.columns.str.strip().str.lower().str.replace(" ", "_")

# Page UI
st.set_page_config(page_title="Medical Cost Estimator", layout="wide")

# Grey Theme
st.markdown("""
<style>
.stApp {
    background-color: #2c2f33;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("💊 AI Medical Cost Estimator + Hospital Suggestion")

# ---------------- INPUT SECTION ---------------- #

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 25)
    bmi = st.slider("BMI", 10.0, 50.0, 22.0)
    children = st.slider("Children", 0, 5, 0)

with col2:
    sex = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["No", "Yes"])

# Convert inputs
sex_val = 1 if sex == "Female" else 0
smoker_val = 1 if smoker == "Yes" else 0

# Final input (same as train.py)
input_data = np.array([[age, sex_val, bmi, children, smoker_val]])

# ---------------- PREDICTION ---------------- #

if st.button("💰 Predict Cost"):

    result = model.predict(input_data)[0]

    st.success(f"Estimated Cost: ₹ {result:,.2f}")

    # Risk level
    if result < 10000:
        st.info("🟢 Low Cost")
    elif result < 30000:
        st.warning("🟠 Medium Cost")
    else:
        st.error("🔴 High Cost")

    # ---------------- HOSPITAL SUGGESTION ---------------- #

    st.subheader("🏥 Suggested Hospitals")

    # Detect columns dynamically
    min_col = None
    max_col = None

    for col in hospitals.columns:
        if "min" in col:
            min_col = col
        if "max" in col:
            max_col = col

    if min_col is None or max_col is None:
        st.error("❌ Hospital dataset column issue")
    else:
        filtered = hospitals[
            (hospitals[min_col] <= result) &
            (hospitals[max_col] >= result)
        ]

        if len(filtered) > 0:
            for i, row in filtered.iterrows():
                st.write(f"""
                🏥 {row.iloc[0]}  
                💰 ₹{row[min_col]} - ₹{row[max_col]}
                """)
        else:
            st.write("No hospitals found in your budget")

    # ---------------- HEALTH INSIGHTS ---------------- #

    st.subheader("📊 Health Insights")

    if bmi < 18.5:
        st.write("⚠ Underweight")
    elif bmi < 25:
        st.write("✅ Normal weight")
    elif bmi < 30:
        st.write("⚠ Overweight")
    else:
        st.write("🚨 Obese")

    if smoker_val:
        st.write("🚬 Smoking increases medical cost")
    else:
        st.write("👍 Non-smoker")

# ---------------- CHATBOT ---------------- #

st.write("---")
st.subheader("💬 Chat Assistant")

msg = st.text_input("Ask something...")

if msg:
    if "cost" in msg.lower():
        st.write("Medical cost depends on age, BMI, and smoking habits.")
    elif "bmi" in msg.lower():
        st.write("BMI indicates body fat. Higher BMI increases risk.")
    elif "smoking" in msg.lower():
        st.write("Smoking significantly increases medical expenses.")
    else:
        st.write("I'm here to help with your health queries!")