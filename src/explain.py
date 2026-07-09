import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import config
from train import (
    load_dataset,
    prepare_data,
    split_data,
    scale_data
)

from config import MODEL_DIR,EXPLAINABILITY_DIR



def main():

    # Load dataset
    df, encoder = load_dataset()

    # Prepare data
    X, y = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale data
    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train,
        X_test
    )
    # Load Tuned Random Forest Model
    model = joblib.load(
        MODEL_DIR / "random_forest_tuned.pkl"
    )

    print("Tuned Random Forest model loaded successfully.")
    # Create SHAP Explainer
    explainer = shap.TreeExplainer(model)

    print("SHAP Explainer created successfully.")

    print("Generating SHAP values...")

    shap_values = explainer.shap_values(X_test_scaled)

    print("SHAP values generated successfully.")
    print(type(shap_values))
    print(shap_values.shape)
    # SHAP Summary Plot (Class 0)

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values[:, :, 0],
        X_test,
        show=False
    )

    plt.tight_layout()

    plt.savefig(EXPLAINABILITY_DIR / "shap_summary_class0.png",dpi=600,bbox_inches="tight")

    plt.close()

    print("SHAP Summary Plot saved successfully.")

    # SHAP Feature Importance Bar Plot

    plt.figure(figsize=(10, 8))

    shap.plots.bar(shap.Explanation(values=shap_values[:, :, 0],data=X_test.values,feature_names=X_test.columns),show=False)

    plt.savefig(EXPLAINABILITY_DIR / "shap_bar_class0.png",dpi=600,bbox_inches="tight")

    plt.close()

    print("SHAP Bar Plot saved successfully.")

if __name__ == "__main__":
    main()