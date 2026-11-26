"""Common interface so every method can be benchmarked the same way.

There are two kinds of methods, and the difference matters for the table:

* Detectors (OpenCV colour, YOLO) look at one frame at a time and answer
  "where is a ball in this image?" from scratch. They need no help to start.
* Prompted trackers (SAM 2, EfficientTAM) answer "where is *this* object now?".
  Someone has to point at the ball once (a click or a box: the "prompt"),
  then they follow it through the video using a memory of past frames.
  In deployment the prompt has to come from a detector, which is why the
  benchmark can prompt them from ground truth (best case) or from a detector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from ..clip import Clip, Detection


@dataclass
class FrameResult:
    frame: int
    det: Detection | None  # best detection, or None if nothing found
    latency_ms: float  # compute time for this frame (excludes reading the JPEG)
    n_dets: int = 0  # how many candidates passed the method's threshold


class Method:
    name: str = "method"

    def run(self, clip: Clip) -> Iterator[FrameResult]:
        raise NotImplementedError


class FrameDetector(Method):
    """A method that processes each frame independently."""

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Return candidates sorted best-first."""
        raise NotImplementedError

    def warmup(self, frame_bgr: np.ndarray, n: int = 3) -> None:
        # The first calls of a neural net are slow (memory allocation, kernel
        # compilation). Run a few before timing so latency numbers are honest.
        for _ in range(n):
            self.detect(frame_bgr)

    def run(self, clip: Clip) -> Iterator[FrameResult]:
        self.warmup(clip.read(0))
        for i in range(len(clip)):
            frame = clip.read(i)
            t0 = time.perf_counter()
            dets = self.detect(frame)
            dt = (time.perf_counter() - t0) * 1000.0
            yield FrameResult(i, dets[0] if dets else None, dt, len(dets))


def mask_to_detection(mask: np.ndarray, score: float = 1.0) -> Detection | None:
    """Turn a binary mask into a ball estimate: largest blob -> enclosing circle.

    The minimum enclosing circle is used (rather than area) because a partly
    occluded ball still spans its full width in at least one direction.
    """
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max() == 0:
        return None
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    if r < 1.0:
        return None
    return Detection(float(cx), float(cy), float(r), float(score))


def best_device() -> str:
    """'cuda' on NVIDIA (e.g. the Jetson), 'mps' on Apple Silicon, else 'cpu'."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
