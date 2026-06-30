from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
CLASSICAL_MODEL_DIR = MODELS_DIR / "classical"
BERTURK_MODEL_DIR = MODELS_DIR / "berturk"

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"

DATASET_ID = "anilguven/turkish_tweet_emotion_dataset"

RAW_DATA_PATH = RAW_DATA_DIR / "turkish_tweet_emotion_raw.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "turkish_tweet_emotion_processed.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
RAW_LABEL_COLUMN = "raw_label"

RANDOM_STATE = 42
TEST_SIZE = 0.2

for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CLASSICAL_MODEL_DIR,
    BERTURK_MODEL_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    METRICS_DIR
]:
    path.mkdir(parents=True, exist_ok=True)
