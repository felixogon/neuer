"""Method registry: build any benchmark method from a short name.

    color                      OpenCV HSV threshold + circle fit (configs/color_tennis.yaml)
    yolo26n, yolo11n, yolo26s  Ultralytics YOLO, COCO "sports ball" class, imgsz 640
    yolo26n@1280               same, run at a larger input size (helps far/small balls)
    sam2.1-tiny, sam2.1-small  SAM 2.1 video predictor, prompted with ground truth
    efficienttam-ti[-512], efficienttam-s-512
    yolo26n+sam2.1-tiny        tracker prompted by a detector instead of ground truth
"""

from __future__ import annotations

from .base import FrameDetector, FrameResult, Method


def build(name: str, device: str | None = None, yolo_conf: float | None = None) -> Method:
    if "+" in name:  # detector-prompted tracker
        det_name, trk_name = name.split("+", 1)
        detector = build(det_name, device, yolo_conf)
        assert isinstance(detector, FrameDetector), f"{det_name} cannot be used as a prompt"
        from .sam_family import PromptedTracker

        return PromptedTracker(trk_name, prompt=detector, device=device)
    if name == "color" or name.startswith("color:"):
        from .color import DEFAULT_CONFIG, ColorDetector

        cfg = name.split(":", 1)[1] if ":" in name else DEFAULT_CONFIG
        return ColorDetector.from_yaml(cfg)
    if name.startswith("yolo"):
        from .yolo import YoloDetector

        stem, _, imgsz = name.partition("@")
        kw = {} if yolo_conf is None else {"conf": yolo_conf}
        return YoloDetector(f"{stem}.pt", imgsz=int(imgsz) if imgsz else 640, device=device, **kw)
    from .sam_family import MODELS, PromptedTracker

    if name in MODELS:
        return PromptedTracker(name, prompt="gt", device=device)
    raise ValueError(f"unknown method {name!r}")


__all__ = ["build", "Method", "FrameDetector", "FrameResult"]
