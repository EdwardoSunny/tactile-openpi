#!/bin/bash
# Render + convert + norm-stats for the points9_arrow_len0 ablation, one array
# task per xarm task. CPU-only.
#
# Prereq: source zarrs copied from the robot workstation into
#   /workspace-vast/edwardosunny/data/tactile/teleop_data_<task>.zarr        (raw)
#   or .../teleop_data_<task>_overlay.zarr                                   (overlay)
# Either works — the renderer only needs img_{i}, joint_angles, grip_pos, tactile.
#
# Submit:  sbatch scripts/prep_arrowlen0.sbatch

#SBATCH --job-name=arrowlen0_prep
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --chdir=/workspace-vast/edwardosunny/tactile-openpi
#SBATCH --output=/workspace-vast/edwardosunny/tactile-openpi/logs/slurm-%x-%A_%a.out
#SBATCH --error=/workspace-vast/edwardosunny/tactile-openpi/logs/slurm-%x-%A_%a.err

set -euo pipefail

export HF_HOME=/workspace-vast/edwardosunny/hf_cache
export HF_LEROBOT_HOME=/workspace-vast/edwardosunny/lerobot-cache
export OPENPI_DATA_HOME=/workspace-vast/edwardosunny/openpi-cache
export PYTHONUNBUFFERED=1

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

TASKS=(cube tube charger dishwasher)
PROMPTS=(
  "pick up the cube"
  "pick up the tube and put it in the nearest slot"
  "pick up the charger and plug it into the nearest plug"
  "pull the basket outside of the dishwasher and pick up the mug and put it into the basket"
)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
PROMPT="${PROMPTS[$SLURM_ARRAY_TASK_ID]}"

DATA_DIR=/workspace-vast/edwardosunny/tactile_data
DC_REPO=/workspace-vast/edwardosunny/tactile-data-collection
DC_PY=/workspace-vast/edwardosunny/envs/tactile-dc/bin/python
OPENPI=/workspace-vast/edwardosunny/tactile-openpi

SRC="$DATA_DIR/teleop_data_${TASK}.zarr"
[ -d "$SRC" ] || SRC="$DATA_DIR/teleop_data_${TASK}_overlay.zarr"
if [ ! -d "$SRC" ]; then
  echo "FATAL: no source zarr for task '$TASK' under $DATA_DIR" >&2
  exit 1
fi
DST="$DATA_DIR/teleop_data_${TASK}_arrowlen0.zarr"

echo "==== $TASK  (job ${SLURM_JOB_ID:-?} array ${SLURM_ARRAY_TASK_ID:-?} on $(hostname)) ===="
echo "src: $SRC"

echo "---- [1/3] render points9_arrow_len0 ----"
srun --ntasks=1 bash -c "cd $DC_REPO && PYTHONPATH=$DC_REPO $DC_PY scripts/render_arrowlen0.py '$SRC' '$DST'"

echo "---- [2/3] convert to LeRobot ----"
srun --ntasks=1 bash -c "cd $OPENPI && ./.venv/bin/python examples/xarm/convert_zarr_to_lerobot.py \
  --zarr '$DST' \
  --mode points9_arrow_len0 \
  --repo-id local/xarm_${TASK}_points9_arrow_len0 \
  --language \"$PROMPT\""

echo "---- [3/3] norm stats ----"
srun --ntasks=1 bash -c "cd $OPENPI && JAX_PLATFORMS=cpu ./.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_xarm_${TASK}_points9_arrow_len0_lora"

echo "==== $TASK done $(date -Iseconds) ===="
