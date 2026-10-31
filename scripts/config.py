from pathlib import Path

# project root = one level above the scripts folder
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx"   # use your real path
TRAIN_CSV = BASE_DIR / "data" / "train.csv"
TEST_CSV = BASE_DIR / "data" / "test_data.csv"
MODEL_DIR = BASE_DIR / "model"