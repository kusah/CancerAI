from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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


if __name__ == "__main__":
    main()