import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add src folder to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src"))

from predict import (
    load_artifacts,
    predict_sample
)

st.set_page_config(
    page_title="Prediction",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Cancer Prediction")

st.write(
    "Upload a gene expression CSV file to predict the cancer type."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Gene Expression CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    model, scaler, encoder, feature_names = load_artifacts()

    uploaded_columns = df.columns.tolist()

    ignore_columns = [
        "Sample_ID",
        "Cancer_Type"
    ]

    uploaded_features = [
        col
        for col in uploaded_columns
        if col not in ignore_columns
    ]

    missing_features = list(
        set(feature_names) - set(uploaded_features)
    )

    extra_features = list(
        set(uploaded_features) - set(feature_names)
    )

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Samples",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Features",
            df.shape[1]
        )

    st.subheader("📋 Dataset Validation")

    if missing_features:

        st.error(
            f"Missing Features: {len(missing_features)}"
        )

        st.write(missing_features[:10])

    if extra_features:

        st.warning(
            f"Extra Features: {len(extra_features)}"
        )

        st.write(extra_features[:10])

    if (
        not missing_features
        and
        not extra_features
    ):

        st.success("Dataset validated successfully!")
    if (
        not missing_features
        and
        not extra_features
    ):

        if st.button("🚀 Predict"):

            st.success("Prediction Started")