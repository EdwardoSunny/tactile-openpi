#!/bin/bash
# Submit with:
#   sbatch --job-name=<short> scripts/slurm_train.sbatch <config_name> [exp_name]
# IMPORTANT: compute nodes have separate /home per node; only /workspace-vast
# is shared. All repo+venv+cache paths below MUST be on /workspace-vast.

#SBATCH --qos=high
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --chdir=/workspace-vast/edwardosunny/tactile-openpi
#SBATCH --output=/workspace-vast/edwardosunny/tactile-openpi/logs/slurm-%x-%j.out
#SBATCH --error=/workspace-vast/edwardosunny/tactile-openpi/logs/slurm-%x-%j.err

set -euo pipefail

CONFIG_NAME="${1:?usage: sbatch scripts/slurm_train.sbatch <config_name> [exp_name]}"
EXP_NAME="${2:-${EXP_NAME:-auto_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-noid}}}"

REPO_DIR=/workspace-vast/edwardosunny/tactile-openpi
export OPENPI_DATA_HOME=/workspace-vast/edwardosunny/openpi-cache
export HF_LEROBOT_HOME=/workspace-vast/edwardosunny/lerobot-cache

cd "$REPO_DIR"
mkdir -p logs

echo "==== job ${SLURM_JOB_ID:-N/A} on ${SLURMD_NODENAME:-$(hostname)} ===="
echo "config:   $CONFIG_NAME"
echo "exp_name: $EXP_NAME"
echo "start:    $(date -Iseconds)"
echo "----"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo "----"
./.venv/bin/python -c "import jax, torch; print('jax devices:', jax.devices()); print('torch cuda:', torch.cuda.is_available(), torch.cuda.device_count())"
echo "----"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
PYTHONUNBUFFERED=1 \
  ./.venv/bin/python -u scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --overwrite

echo "==== finished $(date -Iseconds) ===="
