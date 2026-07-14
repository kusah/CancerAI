import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
from train import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]

REPORTS_DIR = ROOT_DIR / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"
TUNING_DIR = REPORTS_DIR / "tuning"

st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analytics Dashboard")

st.write(
    "Explore model performance, dataset statistics and tuning results."
)

st.divider()


comparison = pd.read_csv(
    RESULTS_DIR / "model_comparison.csv"
)


best = comparison.sort_values(
    "Accuracy",
    ascending=False
).iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🏆 Best Model",
        best["Model"]
    )

with col2:
    st.metric(
        "🎯 Best Accuracy",
        f"{best['Accuracy']:.2f}%"
    )

with col3:
    st.metric(
        "🤖 Models Compared",
        len(comparison)
    )

with col4:
    st.metric(
        "🧬 Cancer Classes",
        "5"
    )

st.subheader("📋 Model Comparison")

st.dataframe(
    comparison,
    width="stretch",
    height=300
)


st.subheader("🏆 Model Leaderboard")

leaderboard = comparison.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

leaderboard.index = leaderboard.index + 1

st.dataframe(
    leaderboard,
    width="stretch"
)


st.subheader("📈 Model Accuracy Comparison")

fig = px.bar(
    leaderboard,
    x="Model",
    y="Accuracy",
    color="Accuracy",
    text="Accuracy",
    title="Model Accuracy Comparison"
)

fig.update_layout(
    xaxis_title="Model",
    yaxis_title="Accuracy (%)"
)

st.plotly_chart(
    fig,
    width="stretch"
)



df, encoder = load_dataset()

st.subheader("🧬 Dataset Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Samples", df.shape[0])

with col2:
    st.metric("Features", df.shape[1]-2)

with col3:
    st.metric("Cancer Types", df["Cancer_Type"].nunique())

with col4:
    st.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )

with col5:
    st.metric(
        "Duplicates",
        df.duplicated().sum()
    )


st.subheader("🧬 Cancer Type Distribution")

df["Cancer_Type"] = encoder.inverse_transform(
    df["Cancer_Type"]
)

distribution = (
    df["Cancer_Type"]
    .value_counts()
    .reset_index()
)

distribution.columns = [
    "Cancer Type",
    "Count"
]

fig = px.pie(
    distribution,
    names="Cancer Type",
    values="Count",
    title="Cancer Type Distribution"
)

st.plotly_chart(
    fig,
    width="stretch"
)



st.subheader("🖼 Model Confusion Matrices")

images = sorted(FIGURES_DIR.glob("*confusion_matrix*.png"))

col1, col2 = st.columns(2)

for i, image in enumerate(images):
    caption = image.stem.replace("_", " ").title()

    if i % 2 == 0:
        with col1:
            st.image(image, caption=caption, width="stretch")
    else:
        with col2:
            st.image(image, caption=caption, width="stretch")
st.subheader("⚙ Hyperparameter Tuning")

rf = pd.read_csv(
    TUNING_DIR / "random_forest_best_params.csv"
)

st.write("### 🌲 Random Forest")

st.dataframe(
    rf,
    width="stretch"
)
xgb = pd.read_csv(
    TUNING_DIR / "xgboost_best_params.csv"
)

st.write("### 🚀 XGBoost")

st.dataframe(
    xgb,
    width="stretch"
)

rf_grid = pd.read_csv(
    TUNING_DIR / "random_forest_gridsearch_results.csv"
)

st.write("### Top Random Forest Grid Search Results")

st.dataframe(
    rf_grid.head(10),
    width="stretch"
)