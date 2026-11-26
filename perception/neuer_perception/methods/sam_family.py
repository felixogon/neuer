"""Promptable video segmentation: SAM 2 (Meta) and EfficientTAM (Meta).

Both share one design (EfficientTAM is built on the SAM 2 code base):

    frame -> image encoder (big ViT) -> features
    features + memory of past frames (memory attention) -> mask decoder -> mask
    the new mask is written back into the memory bank

You give a prompt (here: a box around the ball) on one frame, and the model
propagates the mask through the rest of the video. EfficientTAM swaps SAM 2's
hierarchical Hiera encoder for a plain lightweight ViT and makes the memory
attention cheaper, trading a little accuracy for a lot of speed.

Important for interpreting the benchmark:
* The image encoder runs on every frame -> these are much slower than YOLO.
* They never "find" a ball by themselves: the prompt does. With prompt="gt"
  you measure the best case (perfect initialisation). With a detector prompt
  ("yolo26n+sam2.1-tiny") you measure a realistic detect-then-track pipeline.
* The official video predictor processes a recorded clip (it loads all frames
  up front), so here we measure per-frame compute, not live streaming.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np

from ..clip import Clip
from .base import FrameDetector, FrameResult, Method, best_device, mask_to_detection

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

# name -> (python package, hydra config, checkpoint file)
MODELS = {
    "sam2.1-tiny": ("sam2", "configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "sam2.1-small": ("sam2", "configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "efficienttam-ti": ("efficient_track_anything", "configs/efficienttam/efficienttam_ti.yaml", "efficienttam_ti.pt"),
    "efficienttam-ti-512": (
        "efficient_track_anything",
        "configs/efficienttam/efficienttam_ti_512x512.yaml",
        "efficienttam_ti_512x512.pt",
    ),
    "efficienttam-s-512": (
        "efficient_track_anything",
        "configs/efficienttam/efficienttam_s_512x512.yaml",
        "efficienttam_s_512x512.pt",
    ),
}


def _reset_hydra(package: str) -> None:
    # Both packages register their configs with Hydra's global state on import.
    # Only one can own it at a time, so re-point Hydra at the one we need.
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    initialize_config_module(package, version_base="1.2")


def build_predictor(model: str, device: str, checkpoint: str | Path | None = None):
    package, config, ckpt_name = MODELS[model]
    ckpt = Path(checkpoint) if checkpoint else CHECKPOINT_DIR / ckpt_name
    if checkpoint is not False and not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found. Run perception/setup_models.sh first.")
    ckpt_arg = None if checkpoint is False else str(ckpt)  # False = random weights (tests only)
    _reset_hydra(package)
    if package == "sam2":
        from sam2.build_sam import build_sam2_video_predictor

        return build_sam2_video_predictor(config, ckpt_arg, device=device)
    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

    return build_efficienttam_video_predictor(config, ckpt_arg, device=device)


class NoPrompt(Exception):
    """A ground-truth-prompted tracker cannot start: the ball is never labelled visible."""


class PromptedTracker(Method):
    def __init__(
        self,
        model: str = "sam2.1-tiny",
        prompt: str | FrameDetector = "gt",
        device: str | None = None,
        checkpoint: str | Path | None = None,
    ):
        self.model_name = model
        self.device = device or best_device()
        self.predictor = build_predictor(model, self.device, checkpoint)
        self.prompt = prompt
        self.name = model if prompt == "gt" else f"{prompt.name}+{model}"

    def _find_prompt(self, clip: Clip) -> tuple[int, list[float]] | None:
        """Return (frame index, [x1, y1, x2, y2]) of the first frame we can prompt on."""
        if self.prompt == "gt":
            for f, lab in sorted(clip.labels().items()):
                if lab.visible:
                    return f, [lab.cx - lab.r, lab.cy - lab.r, lab.cx + lab.r, lab.cy + lab.r]
            raise NoPrompt(f"{clip.name}: no labelled visible frame to prompt from")
        for i in range(len(clip)):
            dets = self.prompt.detect(clip.read(i))
            if dets:
                d = dets[0]
                return i, [d.cx - d.r, d.cy - d.r, d.cx + d.r, d.cy + d.r]
        return None

    def run(self, clip: Clip) -> Iterator[FrameResult]:
        import torch

        found = self._find_prompt(clip)
        if found is None:  # the prompting detector never saw the ball
            for i in range(len(clip)):
                yield FrameResult(i, None, float("nan"), 0)
            return
        f0, box = found
        for i in range(f0):  # nothing to report before the prompt
            yield FrameResult(i, None, float("nan"), 0)

        with torch.inference_mode():
            state = self.predictor.init_state(str(clip.frames_dir), offload_video_to_cpu=True)
            self.predictor.add_new_points_or_box(state, frame_idx=f0, obj_id=1, box=np.array(box, dtype=np.float32))
            t0 = time.perf_counter()
            for idx, _obj_ids, mask_logits in self.predictor.propagate_in_video(state, start_frame_idx=f0):
                mask = (mask_logits[0, 0] > 0.0).cpu().numpy()
                det = mask_to_detection(mask)
                dt = (time.perf_counter() - t0) * 1000.0
                yield FrameResult(int(idx), det, dt, int(det is not None))
                t0 = time.perf_counter()
            self.predictor.reset_state(state)
