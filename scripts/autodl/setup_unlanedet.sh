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

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel ninja
"$PYTHON_BIN" -m pip install -r "$UNLANEDET_ROOT/requirements.txt"
"$PYTHON_BIN" -m pip install \
  "numpy==1.26.4" "scipy==1.11.4" "opencv-python-headless==4.9.0.80" \
  "Pillow>=10,<12"

if ! "$PYTHON_BIN" -c '
import sys
import torch
import torchvision
assert sys.version_info[:2] == (3, 10), f"expected Python 3.10, got {sys.version}"
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

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-4}"
"$PYTHON_BIN" -m pip install --no-build-isolation -v -e "$UNLANEDET_ROOT"

"$PYTHON_BIN" -c \
  'import unlanedet; from unlanedet.layers.ops import nms, nms_ad_; print("UnLanedet imports and CUDA ops: OK")'
echo "Pinned UnLanedet AutoDL setup complete: $PINNED_UNLANEDET_COMMIT"
