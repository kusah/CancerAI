from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import joblib
import pandas as pd
from config import MODEL_DIR, TUNING_DIR
from xgboost import XGBClassifier

from train import (
    load_dataset,
    prepare_data,
    split_data,
    scale_data
)

TUNING_MODELS = {
        "random_forest": {
            "model": RandomForestClassifier(
                random_state=42
            ),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },

        "xgboost": {
            "model": XGBClassifier(
            random_state=42,
            eval_metric="mlogloss"
        ),
            "param_grid": {
            "n_estimators": [100, 200],
            "learning_rate": [0.1, 0.2],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0]
            }
        }
    }



def main():
    df, encoder = load_dataset()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train,
        X_test
    )
    
    for model_name, model_info in TUNING_MODELS.items():
        model = model_info["model"]

        param_grid = model_info["param_grid"]
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        print("Starting Grid Search...")

        print(f"\nTuning {model_name.replace('_', ' ').title()}...")

        grid_search.fit(
            X_train_scaled,
            y_train
        )

        print("\nBest Parameters:")
        print(grid_search.best_params_)

        print("\nBest CV Accuracy:")
        print(grid_search.best_score_)


        # -------------------------
        # Save tuned model
        # -------------------------

        best_model = grid_search.best_estimator_

        joblib.dump(
            best_model,
               MODEL_DIR / f"{model_name}_tuned.pkl"
        )

        print("Tuned model saved successfully.")


        best_params = pd.DataFrame(
            [grid_search.best_params_]
        )

        best_params["Best CV Accuracy"] = grid_search.best_score_

        best_params.to_csv(
            TUNING_DIR / f"{model_name}_best_params.csv",
            index=False
        )

        print("Best parameters saved successfully.")

        results_df = pd.DataFrame(grid_search.cv_results_)

        results_df.to_csv(
            TUNING_DIR / f"{model_name}_gridsearch_results.csv",
            index=False
        )

        print("Grid search results saved successfully.")

if __name__ == "__main__":
    main()