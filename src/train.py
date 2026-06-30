from data_loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)
import joblib
from config import MODEL_DIR


def main():

    loader = DataLoader()

    df = loader.load_csv(
        "cancer_dataset.csv",
        folder="processed"
    )

    print("=" * 50)
    print("DATASET LOADED")
    print("=" * 50)

    print(df.shape)

    # Features
    X = df.drop(columns=["Sample_ID", "Cancer_Type"])

    # Target
    y = df["Cancer_Type"]

    # print("\nFeature Shape:", X.shape)
    # print("Target Shape:", y.shape)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain-Test Split Completed!")
    print(f"Training Samples : {X_train.shape}")
    print(f"Testing Samples  : {X_test.shape}")


    # Feature Scaling
    scaler=StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature Scaling Completed!")

    print(f"Training Shape : {X_train_scaled.shape}")
    print(f"Testing Shape  : {X_test_scaled.shape}")


    # Model Training

    model = LogisticRegression(max_iter=1000,random_state=42)
    model.fit(X_train_scaled, y_train)
    print("\nLogistic Regression Model Trained Successfully!")


    # Prediction

    y_pred = model.predict(X_test_scaled)

    print("\nPredictions Generated!")

    # Evaluation

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 50)

    print(f"\nAccuracy : {accuracy:.4f}")

    print("\nClassification Report")

    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix")

    print(confusion_matrix(y_test, y_pred))



    # Save Model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "logistic_regression_v1.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler_v1.pkl")

    print("\nModel saved successfully!")
    print("Scaler saved successfully!")





if __name__ == "__main__":
    main()