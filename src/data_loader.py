import pandas as pd
from pathlib import Path
from config import RAW_DATA_DIR


class DataLoader:
    """
    Loads datasets from data/raw directory.
    """

    def __init__(self):
        self.raw_dir = RAW_DATA_DIR

    def load_csv(self, filename: str) -> pd.DataFrame:
        file_path = self.raw_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"{filename} not found.")

        return pd.read_csv(file_path)


if __name__ == "__main__":
    loader = DataLoader()

    data = loader.load_csv("data.csv")
    labels = loader.load_csv("labels.csv") 

    print("=" * 50)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 50)

    print("\nData Shape:")
    print(data.shape)

    print("\nLabels Shape:")
    print(labels.shape)

    print("\nFirst 5 Rows of Data:")
    print(data.head())

    print("\nFirst 5 Labels:")
    print(labels.head())