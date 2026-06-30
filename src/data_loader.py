import pandas as pd
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataLoader:
    """
    Utility class for loading datasets from the project.
    """

    def __init__(self):
        self.directories = {
            "raw": RAW_DATA_DIR,
            "processed": PROCESSED_DATA_DIR
        }

    def load_csv(self, filename: str, folder: str = "raw") -> pd.DataFrame:
        """
        Load a CSV file from the specified data folder.

        Parameters:
            filename (str): Name of the CSV file.
            folder (str): "raw" or "processed".

        Returns:
            pd.DataFrame
        """

        if folder not in self.directories:
            raise ValueError(f"Invalid folder: {folder}")

        file_path = self.directories[folder] / filename

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found.")

        return pd.read_csv(file_path)
    
if __name__ == "__main__":

    loader = DataLoader()

    raw = loader.load_csv("data.csv")

    processed = loader.load_csv(
        "cancer_dataset.csv",
        folder="processed"
    )

    print(raw.shape)
    print(processed.shape)