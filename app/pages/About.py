import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About CancerAI")

st.write(
    "CancerAI is an Explainable Artificial Intelligence (XAI) platform developed to classify cancer types using gene expression data."
)

st.divider()

st.header("🎯 Project Overview")

st.markdown("""
CancerAI is an end-to-end machine learning application that predicts cancer type
from gene expression data using a tuned Random Forest model.

The platform not only predicts cancer type but also explains each prediction
using SHAP (SHapley Additive exPlanations), making the model transparent and interpretable.

The application supports:

- Gene expression CSV upload
- Dataset validation
- Cancer prediction
- Prediction confidence
- Explainable AI (SHAP)
- Interactive analytics dashboard
""")

st.header("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Samples",
        "801"
    )

with col2:
    st.metric(
        "Gene Features",
        "20,531"
    )

with col3:
    st.metric(
        "Cancer Types",
        "5"
    )

st.subheader("Supported Cancer Types")

st.markdown("""
🩷 **BRCA** — Breast Cancer

🟤 **COAD** — Colon Adenocarcinoma

🟢 **KIRC** — Kidney Renal Clear Cell Carcinoma

🔵 **LUAD** — Lung Adenocarcinoma

🟣 **PRAD** — Prostate Adenocarcinoma
""")

st.header("⚙ Machine Learning Pipeline")

st.markdown("""
1. 📥 Data Collection
2. 🧹 Data Preprocessing
3. 📊 Feature Scaling
4. 🤖 Model Training
5. 🔍 Hyperparameter Tuning
6. 📈 Model Evaluation
7. 🧬 SHAP Explainability
8. 🚀 Deployment using Streamlit
""")

st.header("🛠 Technologies Used")

st.markdown("""
- Python
- Streamlit
- Scikit-learn
- Random Forest
- XGBoost
- SHAP
- Pandas
- NumPy
- Plotly
- Matplotlib
- Joblib
""")

st.header("📈 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Best Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Accuracy",
        "98.76%"
    )

with col3:
    st.metric(
        "CV Accuracy",
        "99.06%"
    )

st.header("🚀 Future Scope")

st.success("""
• Support additional cancer types

• Integrate real-time genomic databases

• Deep Learning based classification

• Clinical Decision Support System

• Multi-omics analysis

• Cloud deployment for healthcare applications
""")

st.header("👨‍💻 Developer")

st.info("""
**Kunal Sah**

B.Tech Computer Science with Bioinformatics

Vellore Institute of Technology (VIT)

Developed as a Machine Learning and Explainable AI project for cancer classification using gene expression data.
""")

st.divider()

st.caption(
    "CancerAI © 2026 | Explainable AI for Cancer Classification"
)