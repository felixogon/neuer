"""Learned detector: Ultralytics YOLO, pretrained on COCO.

COCO has 80 object classes; class 32 is "sports ball", which covers tennis
balls and footballs. So this runs out of the box, without any training on our
data. YOLO is a single-stage detector: one forward pass of a CNN predicts boxes,
class scores and confidences for the whole image at once ("You Only Look Once").

Knobs worth testing:
* model size: n (nano) < s < m < l < x. Bigger = more accurate, slower.
* imgsz: the image is resized so its long side is imgsz before the network
  sees it. A 9-pixel ball in a 1280-wide frame becomes ~4.5 px at imgsz=640,
  which is close to invisible. imgsz=1280 helps small (far) balls, at ~4x cost.
* conf: the confidence threshold. Lower = more recall, more false positives.
"""

from __future__ import annotations

from ..clip import Detection
from .base import FrameDetector, best_device

COCO_SPORTS_BALL = 32


class YoloDetector(FrameDetector):
    def __init__(self, weights: str = "yolo26n.pt", imgsz: int = 640, conf: float = 0.15, device: str | None = None):
        from ultralytics import YOLO  # imported here so the colour baseline works without it

        from pathlib import Path

        ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        local = ckpt_dir / weights
        self.model = YOLO(str(local) if local.exists() or "/" not in weights else weights)  # downloads on first use
        self.imgsz = imgsz
        self.conf = conf
        self.device = device or best_device()
        stem = weights.replace(".pt", "")
        self.name = stem if imgsz == 640 else f"{stem}@{imgsz}"

    def detect(self, frame_bgr):
        res = self.model.predict(
            frame_bgr,
            classes=[COCO_SPORTS_BALL],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        dets = []
        for (x1, y1, x2, y2), score in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            r = max(x2 - x1, y2 - y1) / 2.0  # largest extent: robust to partial occlusion
            dets.append(Detection((x1 + x2) / 2.0, (y1 + y2) / 2.0, r, float(score)))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets
