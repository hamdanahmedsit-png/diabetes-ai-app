import streamlit as st
from model_predict import predict_diabetes

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Diabetes AI Predictor", page_icon="🧠", layout="centered")

# ---------------------------
# HEADER
# ---------------------------
st.title("🧠 Diabetes Risk AI Predictor")
st.markdown("### Predict your diabetes risk using AI")
st.markdown("Enter your health details below 👇")

st.divider()

# ---------------------------
# INPUT UI (2 COLUMNS)
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.slider("Glucose Level", 0, 200)
    blood_pressure = st.slider("Blood Pressure", 0, 150)
    skin_thickness = st.slider("Skin Thickness", 0, 100)

with col2:
    insulin = st.slider("Insulin Level", 0, 900)
    bmi = st.slider("BMI", 0.0, 70.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.slider("Age", 1, 120)

st.divider()

# ---------------------------
# BUTTON
# ---------------------------
if st.button("🔍 Predict Now"):

    input_data = [
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]

    result, prob = predict_diabetes(input_data)

    st.divider()

    # ---------------------------
    # OUTPUT UI
    # ---------------------------
    if result == "High Risk":
        st.error(f"🚨 {result} ({prob*100:.2f}%)")
        st.progress(int(prob * 100))

        st.subheader("⚠️ Health Recommendations")
        st.write("• Reduce sugar & processed food")
        st.write("• Exercise at least 30 mins daily")
        st.write("• Maintain healthy weight")
        st.write("• Monitor blood sugar regularly")
        st.write("• Consult a doctor")

    else:
        st.success(f"✅ {result} ({(1-prob)*100:.2f}%)")
        st.progress(int((1 - prob) * 100))

        st.subheader("💪 Stay Healthy")
        st.write("• Maintain balanced diet")
        st.write("• Stay physically active")
        st.write("• Regular health checkups")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("Built with ❤️ using AI & Streamlit")