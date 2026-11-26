# Lets the scripts import neuer_perception without `pip install -e .`
import sys
from pathlib import Path

PERCEPTION_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PERCEPTION_DIR / "data"
CLIPS_DIR = DATA_DIR / "clips"
RESULTS_DIR = PERCEPTION_DIR / "results"
CALIB_FILE = PERCEPTION_DIR / "configs" / "camera_macbook.yaml"
if str(PERCEPTION_DIR) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_DIR))
