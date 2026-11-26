"""Camera geometry: from pixels to metres.

Pinhole model. A 3D point (X, Y, Z) in the camera frame lands on pixel

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

fx, fy (focal length in pixels) and cx, cy (principal point) are the camera
*intrinsics*; scripts/calibrate.py measures them. A ball of known diameter D
that appears with radius r pixels is therefore at depth

    Z = fx * D / (2 r)

which is how a single RGB camera can estimate distance (monocular depth from
known size). Note the error: dZ/Z = -dr/r, so a 1 px radius error on a ball
that is 10 px wide is a 10 % depth error. Far balls are small, so depth gets
worse fast with distance. The G1's depth camera is meant to fix exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

TENNIS_BALL_DIAMETER_M = 0.067  # ITF spec: 6.54-6.86 cm
SOCCER_BALL_DIAMETER_M = 0.22  # size 5: 68-70 cm circumference


@dataclass
class Intrinsics:
    K: np.ndarray  # 3x3 camera matrix
    dist: np.ndarray  # distortion coefficients (k1, k2, p1, p2, k3)
    width: int
    height: int
    rms_px: float = float("nan")  # reprojection error of the calibration

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """Intrinsics for the same camera at a different resolution (same aspect ratio)."""
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return Intrinsics(K, self.dist.copy(), width, height, self.rms_px)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "dist": self.dist.ravel().tolist(),
            "rms_px": float(self.rms_px),
        }
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @staticmethod
    def load(path: str | Path) -> "Intrinsics":
        d = yaml.safe_load(Path(path).read_text())
        return Intrinsics(
            K=np.array(d["K"], dtype=float),
            dist=np.array(d["dist"], dtype=float),
            width=int(d["width"]),
            height=int(d["height"]),
            rms_px=float(d.get("rms_px", float("nan"))),
        )


def depth_from_radius(r_px: float, fx: float, diameter_m: float = TENNIS_BALL_DIAMETER_M) -> float:
    """Distance along the optical axis from the ball's apparent radius."""
    if not np.isfinite(r_px) or r_px <= 0:
        return float("nan")
    return fx * diameter_m / (2.0 * r_px)


def backproject(u: float, v: float, Z: float, intr: Intrinsics) -> np.ndarray:
    """Pixel (u, v) at depth Z -> 3D point (X, Y, Z) in the camera frame, metres."""
    X = (u - intr.cx) * Z / intr.fx
    Y = (v - intr.cy) * Z / intr.fy
    return np.array([X, Y, Z])
