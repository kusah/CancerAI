import pandas as pd
from config import RESULTS_DIR


def update_model_comparison(
    model_name,
    accuracy,
    report,
    training_time
):
    display_name = model_name.replace("_", " ").title()

    if model_name == "knn":
        display_name = "KNN"

    new_row = pd.DataFrame([{
        "Model": display_name,
        "Accuracy": round(accuracy * 100, 2),
        "Macro Precision": round(report["macro avg"]["precision"], 4),
        "Macro Recall": round(report["macro avg"]["recall"], 4),
        "Macro F1": round(report["macro avg"]["f1-score"], 4),
        "Weighted F1": round(report["weighted avg"]["f1-score"], 4),
        "Training Time (s)": round(training_time, 4)
    }])
    
    comparison_file = RESULTS_DIR / "model_comparison.csv"

    if comparison_file.exists():

        old_df = pd.read_csv(comparison_file)

        old_df = old_df[
            old_df["Model"] != display_name
        ]

        final_df = pd.concat(
            [old_df, new_row],
            ignore_index=True
        )

    else:

        final_df = new_row

    final_df = final_df.sort_values(
        by="Accuracy",
        ascending=False
    ).reset_index(drop=True)

    final_df.to_csv(
        comparison_file,
        index=False
    )

    print("Model comparison updated successfully.")