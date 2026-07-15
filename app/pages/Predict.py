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

    st.dataframe(
        df.head(),
        width="stretch",
        height=250
    )

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

            with st.spinner("🧬 Analyzing gene expression..."):

                X = df.copy()

                if "Sample_ID" in X.columns:
                    X = X.drop(columns=["Sample_ID"])

                if "Cancer_Type" in X.columns:
                    X = X.drop(columns=["Cancer_Type"])

                X_scaled = scaler.transform(X)

                predictions = model.predict(X_scaled)

                probabilities = model.predict_proba(X_scaled)

                predicted_labels = encoder.inverse_transform(
                    predictions
                )



                results = pd.DataFrame({
                    "Sample_ID": df["Sample_ID"],
                    "Predicted_Cancer": predicted_labels,
                    "Confidence (%)": (
                        probabilities.max(axis=1) * 100
                    ).round(2)
                })

                st.balloons()

                st.success(
                    "🎉 Prediction Completed Successfully!"
                )

                st.subheader("📊 Prediction Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                    "Samples Predicted",
                    len(results)
                )

                with col2:
                    st.metric(
                        "Cancer Types Found",
                        results["Predicted_Cancer"].nunique()
                    )

                with col3:
                    st.metric(
                        "Average Confidence",
                        f"{results['Confidence (%)'].mean():.2f}%"
                )


                st.subheader("Prediction Results")

                st.dataframe(
                    results,
                    width="stretch",
                    height=350
                )

                distribution = (
                    results["Predicted_Cancer"]
                    .value_counts()
                )

                st.subheader("📈 Predicted Cancer Distribution")

                st.bar_chart(distribution)

                st.download_button(
                    label="📥 Download Predictions",
                    data=results.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )

st.divider()

st.caption(
    "CancerAI © 2026 | Developed by Kunal Sah"
)