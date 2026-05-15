"""Pull a published openpi xArm checkpoint from the Hugging Face Hub.

Usage:
    /data/edward/openpi/.venv/bin/python examples/xarm/inference/pull_from_hub.py \\
        --repo-id <hf-username>/pi05-xarm-finetune-lora-droid-init-20k \\
        --local-dir ./checkpoints/pi05_xarm_finetune_lora_pulled

Then run inference pointed at the pulled directory:
    --checkpoint ./checkpoints/pi05_xarm_finetune_lora_pulled

Prereqs (same as push):
    /data/edward/openpi/.venv/bin/pip install huggingface_hub
    /data/edward/openpi/.venv/bin/huggingface-cli login    # only if the repo is private

The download uses `snapshot_download` so it's resumable and only fetches the
files we ask for (skip `train_state/` unless you want to resume training).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo-id", type=str, required=True,
                   help="Source HF repo, e.g. yourusername/pi05-xarm-finetune-lora")
    p.add_argument("--local-dir", type=Path, required=True,
                   help="Local directory to download into.")
    p.add_argument("--include-train-state", action="store_true", default=False,
                   help="Also download train_state/ (~2 GB). You only need it to resume training.")
    args = p.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)

    # Only what's needed for inference unless --include-train-state.
    allow = [
        "_CHECKPOINT_METADATA",
        "params/**",
        "assets/**",
    ]
    if args.include_train_state:
        allow.append("train_state/**")

    print(f"[hub] downloading {args.repo_id} -> {args.local_dir}")
    print(f"[hub] allow patterns: {allow}")
    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=str(args.local_dir),
        allow_patterns=allow,
    )
    print(f"[hub] done. checkpoint root: {local_path}")
    print()
    print("Run inference:")
    print(f"  --checkpoint {local_path}")


if __name__ == "__main__":
    main()
