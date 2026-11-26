"""A recorded test clip: frames on disk + timestamps + test conditions + hand labels.

Layout of one clip on disk (written by scripts/record.py):

    data/clips/<name>/
        frames/00000.jpg, 00001.jpg, ...   # one JPEG per camera frame
        timestamps.csv                      # frame index, seconds since first frame
        meta.json                           # test conditions (distance, lighting, ...)
        labels.json                         # ground truth you click in scripts/label.py

Every method is run on the *same* recorded frames, so the comparison is fair:
the ball did exactly the same thing for every model.

The frame naming (<index>.jpg) is also the format SAM 2 / EfficientTAM expect,
so the trackers can read a clip directory directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Label:
    """Ground truth for one frame, clicked by hand."""

    visible: bool
    cx: float = float("nan")  # ball centre, pixels
    cy: float = float("nan")
    r: float = float("nan")  # ball radius, pixels
    occluded: bool = False  # partly hidden (still counts as visible)


@dataclass
class Detection:
    """What a method reports for one frame (its single best guess)."""

    cx: float
    cy: float
    r: float
    score: float = 1.0


class Clip:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(f"{self.frames_dir} does not exist - is this a clip folder?")
        self.frame_paths = sorted(self.frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        meta_path = self.root / "meta.json"
        self.meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        ts_path = self.root / "timestamps.csv"
        if ts_path.exists():
            data = np.loadtxt(ts_path, delimiter=",", skiprows=1, ndmin=2)
            self.timestamps = data[:, 1]
        else:  # fall back to nominal fps
            fps = float(self.meta.get("fps", 30.0))
            self.timestamps = np.arange(len(self.frame_paths)) / fps

    @property
    def name(self) -> str:
        return self.root.name

    def __len__(self) -> int:
        return len(self.frame_paths)

    def read(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.frame_paths[i]))
        if img is None:
            raise IOError(f"could not read {self.frame_paths[i]}")
        return img

    # ---- labels -------------------------------------------------------------
    @property
    def labels_path(self) -> Path:
        return self.root / "labels.json"

    def labels(self) -> dict[int, Label]:
        if not self.labels_path.exists():
            return {}
        raw = json.loads(self.labels_path.read_text())
        return {int(k): Label(**v) for k, v in raw.items()}

    def save_labels(self, labels: dict[int, Label]) -> None:
        out = {str(k): asdict(v) for k, v in sorted(labels.items())}
        self.labels_path.write_text(json.dumps(out, indent=1))
