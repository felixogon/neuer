"""Opening the laptop webcam with OpenCV."""

from __future__ import annotations

import platform

import cv2


def open_camera(index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    """Open a camera and request a resolution/fps (the driver may pick something else).

    On macOS we ask for the AVFoundation backend explicitly. If index 0 turns out
    to be your iPhone (Continuity Camera), try index 1.
    """
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. On macOS, give your terminal camera access in "
            "System Settings > Privacy & Security > Camera, then restart the terminal."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Read one frame so the settings take effect, then report what we actually got.
    cap.read()
    return cap


def describe(cap: cv2.VideoCapture) -> dict:
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps_reported": float(cap.get(cv2.CAP_PROP_FPS)),
    }
