#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
STEP_DEVICE="${1:-auto}"
python examples/wt_limb.py --device "$STEP_DEVICE"
python examples/perturb_mouse.py --device "$STEP_DEVICE"
