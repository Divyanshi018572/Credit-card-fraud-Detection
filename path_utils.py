from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"


def ensure_project_dirs() -> None:
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PIPELINE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def raw_data_path() -> Path:
    return RAW_DATA_DIR / "creditcard.csv"
