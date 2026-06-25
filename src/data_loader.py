from pathlib import Path
import pandas as pd
from config import RAW_DATA_DIR


class DataLoader:
    """
    Responsible for loading datasets from the raw data directory.
    """

    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load a CSV file from the raw data directory.
        """
        file_path = self.raw_data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset '{filename}' not found in {self.raw_data_dir}"
            )

        df = pd.read_csv(file_path)
        return df


if __name__ == "__main__":
    loader = DataLoader()

    print("Data Loader Ready")