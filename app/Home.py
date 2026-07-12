import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CancerAI",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🧬 CancerAI")
st.sidebar.markdown("### Version 1.0")

st.sidebar.info(
    """
**Developed By**

Kunal Sah

B.Tech CSE (Bioinformatics)

VIT
"""
)

st.sidebar.success("Select a page from the navigation above.")

# -----------------------------
# Hero Section
# -----------------------------
st.title("🧬 CancerAI")

st.subheader(
    "AI-Powered Cancer Classification using Gene Expression Data"
)

st.markdown(
"""
Predict **five cancer types** using an optimized **Random Forest**
machine learning model with **SHAP Explainability**.

✔ Random Forest Classifier

✔ Explainable AI (SHAP)

✔ Hyperparameter Tuned Model

✔ 98.76% Prediction Accuracy
"""
)

st.divider()

# -----------------------------
# Performance Cards
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🏆 Best Model",
        value="Random Forest"
    )

with col2:
    st.metric(
        label="🎯 Accuracy",
        value="98.76%"
    )

with col3:
    st.metric(
        label="🧬 Cancer Types",
        value="5"
    )

st.divider()

# -----------------------------
# Two Column Layout
# -----------------------------
left, right = st.columns([2,1])

# -----------------------------
# Left Column
# -----------------------------
with left:

    st.header("📌 About CancerAI")

    st.write(
        """
CancerAI is an AI-powered machine learning platform developed to
classify cancer using **gene expression profiles**.

The system predicts the cancer type from uploaded gene expression
data and provides:

- Accurate cancer prediction
- Prediction confidence
- SHAP Explainability
- Feature Importance
- Top Influential Genes
"""
    )

    st.header("⚙ Technologies")

    st.markdown("""
Python • Pandas • NumPy • Scikit-learn

Random Forest • XGBoost • SHAP

Streamlit • Machine Learning
""")

    st.header("🔄 Workflow")

    st.markdown("""
📂 **Upload Gene Expression CSV**

⬇

✅ **Validate Dataset**

⬇

⚙ **Preprocess Data**

⬇

🤖 **Predict Cancer Type**

⬇

📊 **Prediction Confidence**

⬇

💡 **SHAP Explainability**
""")

# -----------------------------
# Right Column
# -----------------------------
with right:

    st.header("🧬 Supported Cancers")

    st.markdown("""
🩷 **BRCA**  
Breast Cancer

---

🟤 **COAD**  
Colon Adenocarcinoma

---

🟢 **KIRC**  
Kidney Renal Clear Cell Carcinoma

---

🔵 **LUAD**  
Lung Adenocarcinoma

---

🟣 **PRAD**  
Prostate Adenocarcinoma
""")

st.divider()

# -----------------------------
# Footer
# -----------------------------
st.markdown(
"""
<center>

### 🚀 CancerAI

AI-Powered Cancer Classification using Gene Expression Data

Developed by **Kunal Sah**

Computer Science with Bioinformatics • VIT

© 2026 CancerAI

</center>
""",
unsafe_allow_html=True
)