from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

TEMPLATES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_TEMPLATE = TEMPLATES_DIR / "certificate_template.png"
DATABASE_URL = f"sqlite:///{BASE_DIR}/certificates.db"

# Placement model path (trained once, cached)
PLACEMENT_MODEL_PATH = MODEL_DIR / "placement_model.joblib"