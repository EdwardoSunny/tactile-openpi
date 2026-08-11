#!/bin/bash
# Copy the final checkpoint of each points9_arrow_len0 run to the canonical
# transfer location (/workspace-vast/$USER/exp/models/), ready to scp to the
# robot workstation for deployment.
#
# Usage: ./scripts/transfer_arrowlen0_checkpoints.sh
set -euo pipefail

CK=/workspace-vast/edwardosunny/tactile-openpi/checkpoints
DEST=/workspace-vast/edwardosunny/exp/models/tactile-arrowlen0
mkdir -p "$DEST"

for task in cube tube charger dishwasher; do
  cfg="pi05_xarm_${task}_points9_arrow_len0_lora"
  # latest experiment dir, then its highest-numbered step dir
  exp=$(ls -1d "$CK/$cfg"/*/ 2>/dev/null | sort | tail -1) || true
  if [ -z "${exp:-}" ]; then echo "SKIP $cfg: no experiment dir"; continue; fi
  step=$(ls -1 "$exp" | grep -E '^[0-9]+$' | sort -n | tail -1) || true
  if [ -z "${step:-}" ]; then echo "SKIP $cfg: no step dir in $exp"; continue; fi
  out="$DEST/$cfg/$(basename "$exp")/$step"
  echo "copy $cfg  step=$step  ->  $out"
  mkdir -p "$out"
  # params/ + assets/ + metadata are all inference needs; skip train_state/ (optimizer)
  rsync -a --exclude 'train_state' "$exp$step/" "$out/"
done
echo "done. transfer dir: $DEST"
du -sh "$DEST"/* 2>/dev/null || true
