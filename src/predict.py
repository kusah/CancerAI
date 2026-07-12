from config import MODEL_DIR
import pandas as pd
import joblib
from train import (
    load_dataset,
    prepare_data,
    split_data,
    scale_data
)


def load_artifacts():

    model = joblib.load(
        MODEL_DIR / "random_forest_tuned.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "scaler_v1.pkl"
    )

    encoder = joblib.load(
        MODEL_DIR / "label_encoder.pkl"
    )

    return model, scaler, encoder

def prepare_prediction_data():

    df, _ = load_dataset()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train,
        X_test
    )

    return X_test, X_test_scaled, y_test


def predict_sample(
    model,
    encoder,
    sample
):

    prediction = model.predict(sample)

    probability = model.predict_proba(sample)

    cancer = encoder.inverse_transform(
        prediction
    )[0]

    return cancer, probability[0]


def main():

    model, scaler, encoder = load_artifacts()

    X_test, X_test_scaled, y_test = prepare_prediction_data()

    sample = X_test_scaled[0].reshape(1, -1)

    cancer, probabilities = predict_sample(
        model,
        encoder,
        sample
    )
    actual = encoder.inverse_transform(
        [y_test.iloc[0]]
    )[0]

    print("=" * 50)
    print("CANCER PREDICTION")
    print("=" * 50)
    print(f"Actual Cancer Type    : {actual}")
    print(f"Predicted Cancer Type : {cancer}")
    print(f"\nPredicted Cancer Status : {cancer}\n")

    print("Prediction Probabilities\n")

    results = sorted(
        zip(encoder.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    for label, prob in results:
        print(f"{label:<5} : {prob:.2%}")

if __name__ == "__main__":
    main()