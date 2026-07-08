from pathlib import Path

# Root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
INTERIM_DATA_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Other directories
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"
RESULTS_DIR = REPORT_DIR / "results"
TUNING_DIR = REPORT_DIR / "tuning"
# Create directories automatically
for directory in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    TUNING_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

