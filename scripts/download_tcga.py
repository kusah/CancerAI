from pathlib import Path
import requests
from tqdm import tqdm

from src.config import RAW_DATA_DIR

# TCGA-derived expression matrix (Toil RNA-seq)
DATA_URL = (
    "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TcgaTargetGtex_gene_expected_count.gz"
)

OUTPUT_FILE = RAW_DATA_DIR / "expression_matrix.tsv.gz"


def download_file(url: str, destination: Path):
    """
    Download a file with a progress bar.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"File already exists: {destination.name}")
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(destination, "wb") as file,
        tqdm(
            desc=destination.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress.update(len(chunk))

    print("Download completed!")


if __name__ == "__main__":
    download_file(DATA_URL, OUTPUT_FILE)