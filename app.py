import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image

# =======================
# Load Models
# =======================
heart_model = pickle.load(open("model/heart_model.pkl", "rb"))  #model\heart_model.pkl
diabetes_model = pickle.load(open("model/diabetes_model.pkl", "rb"))
kidney_model = pickle.load(open("model/kidney_model.pkl", "rb"))
liver_model = pickle.load(open("model/liver_model.pkl", "rb"))
brain_tumor_model = tf.keras.models.load_model('model/brain_tumor_model.h5')

# Load scalers
heart_scaler = pickle.load(open("model/heart_scaler.pkl", "rb"))
diabetes_scaler = pickle.load(open("model/diabetes_scaler.pkl", "rb"))
kidney_scaler = pickle.load(open("model/kidney_scaler.pkl", "rb"))
liver_scaler = pickle.load(open("model/liver_scaler.pkl", "rb"))

# =======================
# Brain Tumor Classes
# =======================
# Direct mapping (since you already used this in training)
brain_tumor_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# =======================
# Streamlit UI
# =======================
st.title("🩺 AI Health Disease Prediction System")

menu = ["Heart Disease", "Diabetes", "Kidney Disease", "Liver Disease", "Brain Tumor"]
choice = st.sidebar.selectbox("Select Disease to Predict", menu)

# -----------------------
# Heart Disease Prediction
# -----------------------
if choice == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
    cp = st.number_input("Chest Pain Type")
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Serum Cholestoral")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.number_input("Resting ECG results")
    thalach = st.number_input("Maximum Heart Rate Achieved")
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST depression induced by exercise")
    slope = st.number_input("Slope of the peak exercise ST segment")
    ca = st.number_input("Number of major vessels (0-3)")
    thal = st.number_input("Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)")

    if st.button("Predict Heart Disease"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs,
                              restecg, thalach, exang, oldpeak, slope, ca, thal]])
        features = heart_scaler.transform(features)
        prediction = heart_model.predict(features)[0]
        st.success("Disease Detected" if prediction == 1 else "No Disease Detected")

# -----------------------
# Diabetes Prediction
# -----------------------
elif choice == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose")
    blood_pressure = st.number_input("Blood Pressure")
    skin_thickness = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes"):
        features = np.array([[pregnancies, glucose, blood_pressure,
                              skin_thickness, insulin, bmi, dpf, age]])
        features = diabetes_scaler.transform(features)
        prediction = diabetes_model.predict(features)[0]
        st.success("Disease Detected" if prediction == 1 else "No Disease Detected")

# -----------------------
# Kidney Disease Prediction
# -----------------------
# 
# -----------------------
# Kidney Disease Prediction
# -----------------------
elif choice == "Kidney Disease":
    st.header("🧪 Kidney Disease Prediction")

    # ✅ Define encoding maps (must match training preprocessing)
    rbc_map = {"normal": 0, "abnormal": 1}
    pc_map = {"normal": 0, "abnormal": 1}
    pcc_map = {"notpresent": 0, "present": 1}
    ba_map = {"notpresent": 0, "present": 1}
    htn_map = {"no": 0, "yes": 1}
    dm_map = {"no": 0, "yes": 1}
    cad_map = {"no": 0, "yes": 1}
    appet_map = {"good": 0, "poor": 1}
    pe_map = {"no": 0, "yes": 1}
    ane_map = {"no": 0, "yes": 1}

    # ✅ Inputs (25 features)
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    bp = st.number_input("Blood Pressure", min_value=50, max_value=180, value=80)
    sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.number_input("Albumin", min_value=0, max_value=5, value=0)
    su = st.number_input("Sugar", min_value=0, max_value=5, value=0)
    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
    ba = st.selectbox("Bacteria", ["notpresent", "present"])
    bgr = st.number_input("Blood Glucose Random", min_value=50, max_value=500, value=100)
    bu = st.number_input("Blood Urea", min_value=1, max_value=300, value=50)
    sc = st.number_input("Serum Creatinine", min_value=0.1, max_value=20.0, value=1.2)
    sod = st.number_input("Sodium", min_value=100, max_value=200, value=140)
    pot = st.number_input("Potassium", min_value=2.5, max_value=7.5, value=4.5)
    hemo = st.number_input("Hemoglobin", min_value=3, max_value=17, value=13)
    pcv = st.number_input("Packed Cell Volume", min_value=10, max_value=60, value=40)
    wc = st.number_input("White Blood Cell Count", min_value=2000, max_value=30000, value=8000)
    rc = st.number_input("Red Blood Cell Count", min_value=2.5, max_value=8.0, value=5.0)
    htn = st.selectbox("Hypertension", ["no", "yes"])
    dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
    cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
    appet = st.selectbox("Appetite", ["good", "poor"])
    pe = st.selectbox("Pedal Edema", ["no", "yes"])
    ane = st.selectbox("Anemia", ["no", "yes"])

    if st.button("Predict Kidney Disease"):
        # ✅ Encode categorical values
        features = np.array([[age, bp, sg, al, su,
                              rbc_map[rbc], pc_map[pc], pcc_map[pcc], ba_map[ba],
                              bgr, bu, sc, sod, pot, hemo, pcv,
                              wc, rc, htn_map[htn], dm_map[dm], cad_map[cad],
                              appet_map[appet], pe_map[pe], ane_map[ane], 0]])

        # ✅ Apply scaler
        features = kidney_scaler.transform(features)

        # ✅ Predict
        prediction = kidney_model.predict(features)[0]
        st.success("Disease Detected" if prediction == 1 else "No Disease Detected")

# -----------------------
# Liver Disease Prediction
# -----------------------
elif choice == "Liver Disease":
    st.header("🧬 Liver Disease Prediction")

    # ✅ Full 10 features
    age = st.number_input("Age")
    gender = st.selectbox("Gender", [0, 1])  # 0 = Female, 1 = Male
    total_bilirubin = st.number_input("Total Bilirubin")
    direct_bilirubin = st.number_input("Direct Bilirubin")
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase")
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase")
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase")
    total_proteins = st.number_input("Total Proteins")
    albumin = st.number_input("Albumin")
    ag_ratio = st.number_input("Albumin and Globulin Ratio")

    if st.button("Predict Liver Disease"):
        features = np.array([[age, gender, total_bilirubin, direct_bilirubin,
                              alkaline_phosphotase, alamine_aminotransferase,
                              aspartate_aminotransferase, total_proteins,
                              albumin, ag_ratio]])
        features = liver_scaler.transform(features)
        prediction = liver_model.predict(features)[0]
        st.success("Disease Detected" if prediction == 1 else "No Disease Detected")


# -----------------------
# Brain Tumor Prediction
# -----------------------
elif choice == "Brain Tumor":
    st.header("🧠 Brain Tumor Prediction")

    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded MRI Image", use_container_width=True)

        # Preprocess image
        img = img.convert("RGB")  # handles grayscale images
        img = img.resize((150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = brain_tumor_model.predict(img_array)
        predicted_class = brain_tumor_classes[np.argmax(prediction)]

        # Display
        if predicted_class == "notumor":
            st.success("✅ No Brain Tumor Detected")
        else:
            st.error(f"⚠️ Brain Tumor Detected: *{predicted_class.upper()}*")

