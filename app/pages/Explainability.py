import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]

EXPLAINABILITY_DIR = (
    ROOT_DIR
    / "reports"
    / "explainability"
)

st.set_page_config(
    page_title="Explainability",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Model Explainability")

st.write(
    "Understand how the Random Forest model makes predictions using SHAP."
)

st.divider()


st.subheader("📖 What is SHAP?")

st.info("""
SHAP (SHapley Additive exPlanations) is an Explainable AI technique
that measures the contribution of each gene to a model's prediction.

Positive SHAP values increase the likelihood of a cancer type,
while negative values decrease it.
""")


summary = Image.open(
    EXPLAINABILITY_DIR /
    "shap_summary_BRCA.png"
)

st.subheader("📊 SHAP Summary Plot")

st.image(
    summary,
    width="stretch"
)

bar = Image.open(
    EXPLAINABILITY_DIR /
    "shap_bar_BRCA.png"
)

st.subheader("📈 Global Feature Importance")

st.image(
    bar,
    width="stretch"
)

genes = pd.read_csv(
    EXPLAINABILITY_DIR /
    "top_genes_BRCA.csv"
)

st.subheader("🧬 Top 20 Most Influential Genes for BRCA Prediction")

st.dataframe(
    genes,
    width="stretch"
)


st.subheader("📌 Key Insights")

st.success("""
✅ Random Forest identified a small subset of genes that strongly influence cancer classification.

✅ SHAP values provide transparent explanations, making the model more interpretable for biomedical applications.

✅ These genes contribute the most to distinguishing BRCA samples from other cancer types.
""")

with st.expander("📖 Learn More About SHAP"):

    st.markdown("""
### What does SHAP do?

SHAP (SHapley Additive exPlanations) explains how much each feature contributes to a prediction.

- 🔴 Red = High Gene Expression
- 🔵 Blue = Low Gene Expression
- ➡ Positive SHAP Value = Pushes prediction towards BRCA
- ⬅ Negative SHAP Value = Pushes prediction away from BRCA

Higher absolute SHAP values indicate greater influence on the model's decision.
""")
    
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Cancer Type",
        "BRCA"
    )

with col2:
    st.metric(
        "Important Genes",
        len(genes)
    )



st.divider()

st.caption(
    "CancerAI © 2026 | Developed by Kunal Sah"
)



