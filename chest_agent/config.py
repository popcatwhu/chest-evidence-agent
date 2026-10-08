import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("CHEST_DATA_DIR", str(ROOT / "data")))
MODEL = Path(os.getenv("CHEST_MODEL_PATH", str(ROOT.parent / "models/Lingshu-32B-NF4")))
BACKEND = os.getenv("CHEST_BACKEND", "local")
API_MODEL = os.getenv("CHEST_API_MODEL", "")
CXR_FINDINGS_MODEL = ROOT.parent / 'models/CXR-Findings'
CXR_EXPERT_ENABLED = os.getenv('CHEST_CXR_EXPERT','1') == '1'
CXR_READER = os.getenv('CHEST_CXR_READER','iamjb')
NVREASON_MODEL = ROOT.parent / 'models/NV-Reason-CXR-3B'
RECHECK_IMAGE = os.getenv('CHEST_RECHECK_IMAGE','1') == '1'
CONSTRAINED_JSON = os.getenv('CHEST_CONSTRAINED_JSON','1') == '1'
MAX_INPUT_TOKENS = int(os.getenv('CHEST_MAX_INPUT_TOKENS','8192'))
LOAD_NF4 = os.getenv('CHEST_LOAD_NF4','0') == '1'
INFERENCE_PAUSED = os.getenv('CHEST_INFERENCE_PAUSED','0') == '1'

for directory in (DATA, DATA / "uploads", DATA / "artifacts", DATA / "knowledge"):
    directory.mkdir(parents=True, exist_ok=True)
