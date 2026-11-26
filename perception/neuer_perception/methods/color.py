"""Classical baseline: find the ball by its colour. No learning involved.

Pipeline (Blake's board: IM -> colour -> fit circle to px -> done):

1. Blur slightly to remove sensor noise.
2. Convert BGR -> HSV. In HSV, "which colour" (Hue) is separated from
   "how saturated" (S) and "how bright" (V), so a yellow ball stays roughly the
   same Hue in shade and in light. In BGR all three channels change together.
3. Threshold: keep pixels whose H, S, V fall inside a box -> binary mask.
4. Morphology: "open" removes isolated speckles, "close" fills small holes
   (e.g. the white seam of a tennis ball).
5. Contours: outline every blob in the mask.
6. Fit a circle to each blob and score it by how ball-like it is
   (fill ratio = blob area / circle area; 1.0 for a perfect disc).

Strengths: ~1 ms per frame, fully interpretable, no GPU.
Weaknesses: anything of the same colour is a ball; the thresholds drift with
lighting and camera auto-exposure; a classic white/black football has no
distinctive colour at all. These are exactly what the test matrix probes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from ..clip import Detection
from .base import FrameDetector

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "color_tennis.yaml"


class ColorDetector(FrameDetector):
    name = "opencv-color"

    def __init__(
        self,
        hsv_lo=(25, 80, 70),
        hsv_hi=(45, 255, 255),
        blur_ksize: int = 5,
        morph_ksize: int = 5,
        min_radius_px: float = 2.0,
        min_fill: float = 0.45,
    ):
        self.hsv_lo = np.array(hsv_lo, dtype=np.uint8)
        self.hsv_hi = np.array(hsv_hi, dtype=np.uint8)
        self.blur_ksize = blur_ksize
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        self.min_radius_px = min_radius_px
        self.min_fill = min_fill

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG) -> "ColorDetector":
        cfg = yaml.safe_load(Path(path).read_text()) if Path(path).exists() else {}
        return cls(**cfg)

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.GaussianBlur(frame_bgr, (self.blur_ksize, self.blur_ksize), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel)
        return m

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        m = self.mask(frame_bgr)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if r < self.min_radius_px:
                continue
            fill = cv2.contourArea(c) / (np.pi * r * r)
            if fill < self.min_fill:
                continue
            # Prefer big, round blobs. Score in [0, 1]-ish; area term breaks ties.
            dets.append(Detection(cx, cy, r, score=float(min(fill, 1.0))))
        dets.sort(key=lambda d: d.score * d.r, reverse=True)
        return dets
