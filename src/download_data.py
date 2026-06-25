from pathlib import Path
from config import RAW_DATA_DIR

class DataDownloader:
    """
    Handles downloading and verifying datasets.
    """

    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR

    def dataset_exists(self, filename: str) -> bool:
        """
        Check if the dataset already exists.
        """
        file_path = self.raw_data_dir / filename
        return file_path.exists()

    def show_status(self, filename: str):
        if self.dataset_exists(filename):
            print(f"Dataset found: {filename}")
        else:
            print(f"Dataset not found: {filename}")


if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.show_status("tcga_gene_expression.csv")