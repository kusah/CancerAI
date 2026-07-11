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
CLASS_NAMES = {
    0: "BRCA",
    1: "COAD",
    2: "KIRC",
    3: "LUAD",
    4: "PRAD"
}
import numpy as np


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

    class_index = 0

    class_name = CLASS_NAMES[class_index]

    # SHAP Summary Plot (Class 0)

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values[:, :, 0],
        X_test,
        show=False
    )

    plt.tight_layout()

    plt.savefig(EXPLAINABILITY_DIR / f"shap_summary_{class_name}.png", dpi=600, bbox_inches="tight")

    plt.close()

    print("SHAP Summary Plot saved successfully.")
#     SHAP Summary Plot 

# This is exactly what we want.

# It tells us:

# Top 20 most influential genes for Class 0.
# Red = high gene expression.
# Blue = low gene expression.
# Points farther from zero have a stronger impact on the prediction.

# This is a standard explainability visualization used in many ML projects.

    # SHAP Feature Importance Bar Plot

    plt.figure(figsize=(10, 8))

    shap.plots.bar(shap.Explanation(values=shap_values[:, :, 0],data=X_test.values,feature_names=X_test.columns),show=False)

    plt.savefig(EXPLAINABILITY_DIR / f"shap_bar_{class_name}.png", dpi=600, bbox_inches="tight")

    plt.close()

    print("SHAP Bar Plot saved successfully.")


    
    importance = np.abs(
        shap_values[:, :, class_index]
    ).mean(axis=0)

    top_genes = (
        pd.DataFrame({
            "Gene": X.columns,
            "Importance": importance
        })
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(20)
    )

    top_genes.to_csv(
        EXPLAINABILITY_DIR /
        f"top_genes_{class_name}.csv",
        index=False
    )

    print("Top important genes saved successfully.")

    

if __name__ == "__main__":
    main()