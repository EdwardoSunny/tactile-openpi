"""Offline policy probe — load a checkpoint, run one forward pass on a
synthetic observation at the documented xArm home pose, print the action chunk.

No cameras, no tactile, no arm needed. Used to spot-check that a checkpoint
loads cleanly and produces non-NaN, plausibly-scaled outputs before bringing
hardware online.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from openpi.policies import policy_config
from openpi.training import config as _config


HOME_TCP_MM_DEG = (475.79, -1.14, 244.72, 179.13, -0.01, 0.78)


def home_state_8dim() -> np.ndarray:
    xyz = np.array(HOME_TCP_MM_DEG[:3], dtype=np.float32) / 1000.0
    aa = Rot.from_euler("xyz", HOME_TCP_MM_DEG[3:], degrees=True).as_rotvec().astype(np.float32)
    return np.concatenate([xyz, aa, [0.0, 0.0]]).astype(np.float32)


def synth_image(kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "gray":
        return np.full((224, 224, 3), 128, dtype=np.uint8)
    if kind == "noise":
        return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    raise ValueError(kind)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-config-name", type=str, required=True)
    p.add_argument("--prompt", type=str, default="pick up the tube")
    p.add_argument("--image-kind", choices=["gray", "noise"], default="gray")
    p.add_argument("--n-trials", type=int, default=3,
                   help="Run N trials with different image seeds to see how stable the action chunk is.")
    args = p.parse_args()

    print(f"loading: {args.checkpoint}")
    print(f"config:  {args.train_config_name}")
    print(f"prompt:  {args.prompt}")
    print(f"image:   {args.image_kind}")
    print("=" * 78)

    train_cfg = _config.get_config(args.train_config_name)
    policy = policy_config.create_trained_policy(
        train_cfg, str(args.checkpoint), default_prompt=args.prompt
    )
    state = home_state_8dim()
    print(f"state (home, 8-dim): {np.array2string(state, precision=4, suppress_small=True)}")
    print()

    chunks = []
    for t in range(args.n_trials):
        agent = synth_image(args.image_kind, seed=1000 + t)
        wrist = synth_image(args.image_kind, seed=2000 + t)
        obs = {
            "observation/image": agent,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": args.prompt,
        }
        chunk = np.asarray(policy.infer(obs)["actions"])  # (10, 7)
        chunks.append(chunk)
        nan_count = int(np.isnan(chunk).sum())
        print(f"--- trial {t}  (shape={chunk.shape}, NaN={nan_count}) ---")
        print("        dx       dy       dz       drx      dry      drz      grasp")
        for i, a in enumerate(chunk):
            print(f"  t={i:2d}  {a[0]:+.4f}  {a[1]:+.4f}  {a[2]:+.4f}   "
                  f"{a[3]:+.4f}  {a[4]:+.4f}  {a[5]:+.4f}   {a[6]:+.3f}")
        cum = chunk[:, :3].sum(axis=0)
        print(f"  cum dxyz over 10 steps: [{cum[0]:+.4f} {cum[1]:+.4f} {cum[2]:+.4f}] m"
              f"   |xyz-step max|={np.max(np.linalg.norm(chunk[:, :3], axis=1)):.4f} m"
              f"   grasp[0,5,9]=[{chunk[0,6]:+.2f}, {chunk[5,6]:+.2f}, {chunk[9,6]:+.2f}]")
        print()

    stacked = np.stack(chunks)  # (n_trials, 10, 7)
    print("=" * 78)
    print("across-trial summary  (per-action-dim mean and std over trials):")
    print("              dx       dy       dz       drx      dry      drz      grasp")
    for i in range(stacked.shape[1]):
        mean = stacked[:, i, :].mean(axis=0)
        std = stacked[:, i, :].std(axis=0)
        print(f"  t={i:2d} mean  {mean[0]:+.4f}  {mean[1]:+.4f}  {mean[2]:+.4f}   "
              f"{mean[3]:+.4f}  {mean[4]:+.4f}  {mean[5]:+.4f}   {mean[6]:+.3f}")
        print(f"        std   {std[0]:.4f}   {std[1]:.4f}   {std[2]:.4f}    "
              f"{std[3]:.4f}   {std[4]:.4f}   {std[5]:.4f}    {std[6]:.3f}")


if __name__ == "__main__":
    main()
