"""Push a trained openpi xArm checkpoint dir to the Hugging Face Hub.

Usage:
    /data/edward/openpi/.venv/bin/python examples/xarm/inference/push_to_hub.py \\
        --checkpoint /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999 \\
        --repo-id <hf-username>/pi05-xarm-finetune-lora-droid-init-20k

Prereqs:
    /data/edward/openpi/.venv/bin/pip install huggingface_hub
    /data/edward/openpi/.venv/bin/huggingface-cli login

The push is a single `upload_folder` call. With Git LFS this handles the ~9 GB
checkpoint as multiple LFS objects automatically. The `params/` orbax shards are
the bulk of the size; everything else is small.

What ends up in the repo:
    repo-id/
        _CHECKPOINT_METADATA
        params/              # ~7 GB orbax shards (the model weights)
        train_state/         # ~2 GB optimizer state (only needed if resuming training)
        assets/
            local/xarm_teleop/norm_stats.json   # required for inference

Pull side: see pull_from_hub.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Local checkpoint directory to upload "
                        "(e.g. .../pi05_xarm_finetune_lora/droid_init_20k/19999)")
    p.add_argument("--repo-id", type=str, required=True,
                   help="Target HF repo, e.g. yourusername/pi05-xarm-finetune-lora")
    p.add_argument("--private", action="store_true", default=False,
                   help="Create as a private repo (default: public).")
    p.add_argument("--exclude-train-state", action="store_true", default=False,
                   help="Don't upload train_state/ (saves ~2 GB; you only need it to resume training).")
    p.add_argument("--commit-message", type=str, default="upload xarm fine-tune checkpoint")
    args = p.parse_args()

    if not args.checkpoint.is_dir():
        raise SystemExit(f"Not a directory: {args.checkpoint}")
    required = ["params", "assets"]
    missing = [r for r in required if not (args.checkpoint / r).exists()]
    if missing:
        raise SystemExit(f"Checkpoint missing required entries: {missing}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"[hub] target repo: {args.repo_id} (private={args.private})")

    ignore = ["*.lock", "*.tmp", "*.orbax-checkpoint-tmp-*"]
    if args.exclude_train_state:
        ignore.append("train_state/*")

    print(f"[hub] uploading {args.checkpoint} (ignore: {ignore})…")
    api.upload_folder(
        folder_path=str(args.checkpoint),
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=ignore,
        commit_message=args.commit_message,
    )
    print(f"[hub] done. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
