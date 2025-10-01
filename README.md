# 🧬 AI-Powered Multi-Disease Prediction System

## 📌 Overview
This project is an **AI-powered healthcare assistant** that predicts multiple diseases using a combination of **Machine Learning (ML)** and **Deep Learning (DL)** models.  

The goal is to provide a **simple and interactive Streamlit application** that allows users to input medical parameters (like blood pressure, glucose level, hemoglobin, etc.) or upload brain MRI scans, and get quick AI-based predictions.  

⚠️ **Note:** This project is for **educational and research purposes only** and should **not** be used for real medical diagnosis.  

---

## 🧑‍⚕️ Diseases Covered
- 💓 **Heart Disease** – Logistic Regression / Random Forest on tabular data  
- 🩸 **Diabetes** – ML model trained on PIMA dataset  
- 🧪 **Kidney Disease** – ML model with encoded categorical + numerical features  
- 🍷 **Liver Disease** – ML model for liver function dataset  
- 🧠 **Brain Tumor** – CNN-based Deep Learning model trained on MRI images  

---

## 🚀 Features
✅ Predicts 5 major diseases from health parameters / images  
✅ User-friendly **Streamlit interface**  
✅ Uses **ML models** (pickle format) for structured datasets  
✅ Uses **CNN model** (`.h5`) for brain tumor detection  
✅ Extendable to **more diseases**, **Blockchain integration**, or **Federated Learning** in the future  

---

## ⚙️ Installation & Setup

Clone the repository:
```bash
git clone https://github.com/ProxyCode1010/AI-Health-Disease-Prediction-System.git
cd AI-Health-Disease-Prediction-System

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py


---

🏗️ Tech Stack

Python 3.8+

Streamlit – for UI

Scikit-learn – for ML disease prediction models

TensorFlow / Keras – for CNN brain tumor detection

Pandas, NumPy – for data preprocessing

Pillow (PIL) – for handling MRI images



---

⚠️ Disclaimer

This application is developed for educational and research purposes only.

Predictions are based on trained AI/ML models and may not be 100% accurate.

Do not use this tool for real medical treatment or decision-making.

Always consult with a certified medical professional for health concerns.

Developers hold no responsibility for risks, harm, or consequences arising from the use of this project.



---

📌 Future Scope

Add more diseases (e.g., Lung Cancer, Alzheimer’s, Parkinson’s)

Enable real-time IoT sensor integration (wearables & medical devices)

Secure data sharing using Blockchain for tamper-proof medical records

Implement Federated Learning for privacy-preserving healthcare AI

Integrate Transformer-based models (e.g., BERT, Vision Transformers, Med-BERT) for advanced medical predictions such as disease progression forecasting, personalized treatment recommendations, and multi-modal diagnosis combining text + images



---

👨‍💻 Developed by: ProxyCode1010
🔗 GitHub: https://github.com/ProxyCode1010/AI-Health-Disease-Prediction-System
