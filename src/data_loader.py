import pandas as pd
from pathlib import Path

RAW_DATA = Path("data/raw")
PROCESSED_DATA = Path("data/processed")

print("Raw data folder:", RAW_DATA)
print("Processed data folder:", PROCESSED_DATA)

RAW_DATA.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

print("Folders verified successfully!")