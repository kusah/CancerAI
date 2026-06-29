import pandas as pd
from sklearn.preprocessing import LabelEncoder
from data_loader import DataLoader
from config import PROCESSED_DATA_DIR
import joblib
from config import MODEL_DIR
def main():
    loader=DataLoader()

    data=loader.load_csv("data.csv")
    labels=loader.load_csv("labels.csv")

    #print(data.columns)
    data = data.rename(columns={"Unnamed: 0": "Sample_ID"})
    labels = labels.rename(columns={"Unnamed: 0": "Sample_ID","Class": "Cancer_Type"})
    # print(data.columns[:5])      # Show first few columns
    # print(labels.columns)

    merge_data=pd.merge(data,labels,on="Sample_ID")
    # print(merge_data.sample(5))
    # print(merge_data.shape)

    encoder = LabelEncoder()

    merge_data["Cancer_Type"] = encoder.fit_transform(merge_data["Cancer_Type"])
    print("\nClass Mapping:")

    # for index, label in enumerate(encoder.classes_):
    #     print(f"{label} -> {index}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DATA_DIR / "cancer_dataset.csv"

    merge_data.to_csv(output_file, index=False)

    print(f"\nProcessed dataset saved to:\n{output_file}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

    print("Label encoder saved successfully.")
if __name__ == "__main__":
    main()