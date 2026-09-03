#!/usr/bin/env bash
set -Eeuo pipefail

PINNED_UNLANEDET_COMMIT="03921844220adb2e65c840de2d9759478d5c3d4c"
UNLANEDET_URL="https://github.com/zkyntu/UnLanedet.git"
PYTHON_BIN="${HARDLANE_PYTHON:-python}"

require_absolute_dir_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "ERROR: $name must be set to an absolute AutoDL path" >&2
    exit 2
  fi
}

require_absolute_dir_var HARDLANE_PROJECT_ROOT
require_absolute_dir_var HARDLANE_DATA_ROOT
require_absolute_dir_var UNLANEDET_ROOT
require_absolute_dir_var HARDLANE_WEIGHTS_ROOT

if [[ ! -f "$HARDLANE_PROJECT_ROOT/TASKS.md" ]]; then
  echo "ERROR: HARDLANE_PROJECT_ROOT is not this project checkout" >&2
  exit 2
fi
for manifest in \
  "$HARDLANE_PROJECT_ROOT/data/processed/manifest_train_v1_seed42.jsonl" \
  "$HARDLANE_PROJECT_ROOT/data/processed/manifest_val_v1_seed42.jsonl"; do
  if [[ ! -f "$manifest" ]]; then
    echo "ERROR: missing frozen manifest: $manifest" >&2
    exit 2
  fi
done
for directory in JPEGImages anno_txt Annotations; do
  if [[ ! -d "$HARDLANE_DATA_ROOT/$directory" ]]; then
    echo "ERROR: HARDLANE_DATA_ROOT missing $directory/" >&2
    exit 2
  fi
done
mkdir -p "$HARDLANE_WEIGHTS_ROOT"

if [[ ! -e "$UNLANEDET_ROOT" ]]; then
  git clone "$UNLANEDET_URL" "$UNLANEDET_ROOT"
elif [[ ! -d "$UNLANEDET_ROOT/.git" ]]; then
  echo "ERROR: UNLANEDET_ROOT exists but is not a git checkout" >&2
  exit 2
fi

actual_commit="$(git -C "$UNLANEDET_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$PINNED_UNLANEDET_COMMIT" ]]; then
  if ! git -C "$UNLANEDET_ROOT" diff --quiet || \
     ! git -C "$UNLANEDET_ROOT" diff --cached --quiet; then
    echo "ERROR: refusing to switch a dirty UnLanedet checkout" >&2
    exit 2
  fi
  git -C "$UNLANEDET_ROOT" fetch origin "$PINNED_UNLANEDET_COMMIT"
  git -C "$UNLANEDET_ROOT" checkout --detach "$PINNED_UNLANEDET_COMMIT"
fi

patch_file="$HARDLANE_PROJECT_ROOT/patches/unlanedet_hardlane.patch"
if git -C "$UNLANEDET_ROOT" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
  echo "HardLane UnLanedet patch already applied"
elif git -C "$UNLANEDET_ROOT" diff --quiet && \
     git -C "$UNLANEDET_ROOT" apply --check "$patch_file"; then
  git -C "$UNLANEDET_ROOT" apply "$patch_file"
else
  echo "ERROR: UnLanedet checkout is dirty or patch does not match pinned commit" >&2
  exit 2
fi

"$PYTHON_BIN" -m pip install --upgrade pip "setuptools<81" wheel ninja
# Pin the exact PyTorch/TorchVision build the project is validated against. AutoDL official
# images only ship 2.2.1 / 2.0.0 / 2.5.1 etc. (no 2.1.2), so reinstall here regardless of the
# base image. cu118 runtime is backward compatible with cu121 drivers on the 3090.
"$PYTHON_BIN" -m pip install \
  "torch==2.1.2+cu118" "torchvision==0.16.2+cu118" \
  --index-url https://download.pytorch.org/whl/cu118
# albumentations==0.4.6 is an sdist whose setup.py imports pkg_resources, removed from setuptools
# >=81. Pin setuptools<81 (above) and disable build isolation so the build uses the env's
# setuptools (still provides pkg_resources) instead of pip's isolated latest build of setuptools.
"$PYTHON_BIN" -m pip install --no-build-isolation -r "$UNLANEDET_ROOT/requirements.txt"
"$PYTHON_BIN" -m pip install \
  "numpy==1.26.4" "scipy==1.11.4" "opencv-python-headless==4.9.0.80" \
  "Pillow>=10,<12"
# imgaug==0.4.0 pulls the GUI build opencv-python, which shares the cv2/ dir with headless and
# shadows our pinned headless build. Drop the GUI variant and force-reinstall headless so the
# pinned 4.9.0.80 is the only cv2 provider.
"$PYTHON_BIN" -m pip uninstall -y opencv-python >/dev/null 2>&1 || true
"$PYTHON_BIN" -m pip install --force-reinstall --no-deps "opencv-python-headless==4.9.0.80"

if ! "$PYTHON_BIN" -c '
import sys
import torch
import torchvision
assert sys.version_info[:2] in {(3, 8), (3, 9), (3, 10), (3, 11)}, f"expected Python 3.8/3.9/3.10/3.11, got {sys.version}"
assert torch.__version__.startswith("2.1.2"), f"expected torch 2.1.2, got {torch.__version__}"
assert torchvision.__version__.startswith("0.16.2"), f"expected torchvision 0.16.2, got {torchvision.__version__}"
assert torch.version.cuda == "11.8", f"expected torch CUDA 11.8, got {torch.version.cuda}"
assert torch.cuda.is_available(), "CUDA unavailable"
print(sys.version)
print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
'; then
  echo "ERROR: use an AutoDL PyTorch image with working CUDA before running setup" >&2
  exit 2
fi

# Auto-detect the CUDA arch from the actual GPU compute capability (3090 -> 8.6, 4090 -> 8.9).
# Honor an explicit TORCH_CUDA_ARCH_LIST from the caller; only default-detect when unset.
# Fallback 8.6 targets the RTX 3090 (sm_86) used for this challenge's baseline runs.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$("$PYTHON_BIN" -c 'import torch; c = torch.cuda.get_device_capability(0); print(f"{c[0]}.{c[1]}")' 2>/dev/null || echo 8.6)"
  echo "setup: TORCH_CUDA_ARCH_LIST auto-detected = $TORCH_CUDA_ARCH_LIST (fallback 8.6 for 3090)"
fi
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS="${MAX_JOBS:-4}"
"$PYTHON_BIN" -m pip install --no-build-isolation -v -e "$UNLANEDET_ROOT"

"$PYTHON_BIN" -c \
  'import unlanedet; from unlanedet.layers.ops import nms, nms_ad_; print("UnLanedet imports and CUDA ops: OK")'
echo "Pinned UnLanedet AutoDL setup complete: $PINNED_UNLANEDET_COMMIT"
