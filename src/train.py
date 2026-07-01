from data_loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)
import joblib
from config import MODEL_DIR, FIGURES_DIR, RESULTS_DIR
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
import time

from report_utils import update_model_comparison
from sklearn.ensemble import RandomForestClassifier


def load_dataset():
    loader = DataLoader()

    encoder = joblib.load(
        MODEL_DIR / "label_encoder.pkl"
    )

    df = loader.load_csv(
        "cancer_dataset.csv",
        folder="processed"
    )

    return df, encoder

def prepare_data(df):
    X = df.drop(columns=["Sample_ID", "Cancer_Type"])

    y = df["Cancer_Type"]

    return X, y

def split_data(X, y):
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

    return X_train, X_test, y_train, y_test

def scale_data(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature Scaling Completed!")

    print(f"Training Shape : {X_train_scaled.shape}")
    print(f"Testing Shape  : {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, scaler


def train_and_evaluate(model,model_name,X_train,X_test,y_train,y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"\n{model_name} Model Trained Successfully!")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy : {accuracy:.4f}")

    

    return y_pred, accuracy, training_time

def generate_reports(
    y_test,
    y_pred,
    accuracy,
    encoder,
    model_name,
    training_time
):
    report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        RESULTS_DIR / f"{model_name}_classification_report.csv"
    )   

    print("Classification report saved.")

    
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm,index=encoder.classes_,columns=encoder.classes_)

    cm_df.to_csv(
        RESULTS_DIR / f"{model_name}_confusion_matrix.csv"
    )

    print("Confusion matrix saved.")
    

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ],
        "Value": [
            accuracy,
            report["macro avg"]["precision"],
            report["macro avg"]["recall"],
            report["macro avg"]["f1-score"]
        ]
    })

    metrics.to_csv(RESULTS_DIR /  f"{model_name}_metrics.csv",index=False)

    print("Metrics saved.")

    update_model_comparison(
        model_name,
        accuracy,
        report,
        training_time
    )
      # ==========================
    # Save Confusion Matrix Figure
    # ==========================

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=encoder.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False
    )
    plt.xticks(fontsize=11)

    plt.yticks(fontsize=11)

    plt.title(
        f"{model_name.replace('_', ' ').title()}\nAccuracy = {accuracy:.2%}",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{model_name}_confusion_matrix.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix figure saved.")

    


def save_model(
    model,
    scaler,
    model_name
):
    joblib.dump(
        model,
        MODEL_DIR / f"{model_name}_v1.pkl"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler_v1.pkl"
    )
    print(f"{model_name} model saved successfully.")
    print("Scaler saved successfully.")

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "decision_tree": DecisionTreeClassifier(
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

def main():


    df, encoder = load_dataset()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    

    # Feature Scaling
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train,X_test)


    # Model Training

    for model_name, model in MODELS.items():
        
        print(f"\nTraining {model_name.replace('_', ' ').title()} Model...")

        y_pred, accuracy, training_time = train_and_evaluate(
            model,
            model_name,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )

        generate_reports(
            y_test,
            y_pred,
            accuracy,
            encoder,
            model_name,
            training_time
        )

        save_model(
            model,
            scaler,
            model_name
        )

if __name__ == "__main__":
    main()