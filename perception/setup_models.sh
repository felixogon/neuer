#!/usr/bin/env bash
# Install SAM 2 and EfficientTAM (not on PyPI in usable form) and download their weights.
# YOLO weights download themselves on first use.
#   bash setup_models.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p checkpoints third_party

# SAM2_BUILD_CUDA=0: skip the optional CUDA extension (there is no CUDA on a Mac).
[ -d third_party/sam2 ] || git clone --depth 1 https://github.com/facebookresearch/sam2 third_party/sam2
SAM2_BUILD_CUDA=0 pip install -e third_party/sam2

[ -d third_party/EfficientTAM ] || git clone --depth 1 https://github.com/yformer/EfficientTAM third_party/EfficientTAM
SAM2_BUILD_CUDA=0 pip install -e third_party/EfficientTAM

fetch() { [ -f "checkpoints/$2" ] || curl -L --fail -o "checkpoints/$2" "$1/$2"; }
SAM=https://dl.fbaipublicfiles.com/segment_anything_2/092824
fetch $SAM sam2.1_hiera_tiny.pt
fetch $SAM sam2.1_hiera_small.pt
ETAM=https://huggingface.co/yunyangx/efficient-track-anything/resolve/main
fetch $ETAM efficienttam_ti.pt
fetch $ETAM efficienttam_ti_512x512.pt
fetch $ETAM efficienttam_s_512x512.pt
ls -lh checkpoints
