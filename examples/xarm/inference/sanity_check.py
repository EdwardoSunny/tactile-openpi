"""Offline sanity check for a trained pi05_xarm_finetune_lora checkpoint.

Loads the policy via openpi, then for each of three training episodes pulls three
frames covering distinct phases of the demonstration:
  - early approach  (gripper open, far above object)
  - just before grasp closes
  - just after grasp closes (lift)
At each frame we run inference and compare the predicted 10-step action chunk to the
ground-truth `tcp[t+1] - tcp[t]` chunk built from the same demo. This confirms the
model output is in the right shape, has correct units, and that the descent/close/lift
phase dynamics are actually being predicted — without touching the robot or cameras.

Usage:
    /data/edward/openpi/.venv/bin/python examples/xarm/inference/sanity_check.py \\
        --checkpoint /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import zarr

from openpi.policies import policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    checkpoint: Path
    train_config_name: str = "pi05_xarm_finetune_lora"
    zarr_path: Path = Path("/data/edward/teleop_data.zarr")
    prompt: str = "pick up the red block"
    n_episodes: int = 3
    horizon: int = 10


def euler_deg_to_axis_angle(rpy_deg: np.ndarray) -> np.ndarray:
    return Rot.from_euler("xyz", rpy_deg, degrees=True).as_rotvec().astype(np.float32)


def build_obs(g: zarr.Group, frame_idx: int, prompt: str) -> dict:
    state_raw = np.asarray(g["data/state"][frame_idx], dtype=np.float32)
    img0 = np.clip(np.asarray(g["data/img_0"][frame_idx]) * 255.0, 0, 255).astype(np.uint8)
    img1 = np.clip(np.asarray(g["data/img_1"][frame_idx]) * 255.0, 0, 255).astype(np.uint8)
    state = np.concatenate([
        state_raw[:3] / 1000.0,
        euler_deg_to_axis_angle(state_raw[3:6]),
        [state_raw[6], state_raw[6]],
    ]).astype(np.float32)
    return {
        "observation/image": img0,
        "observation/wrist_image": img1,
        "observation/state": state,
        "prompt": prompt,
    }


def ground_truth_chunk(g: zarr.Group, frame_idx: int, horizon: int) -> np.ndarray | None:
    tcp = np.asarray(g["data/state"][frame_idx : frame_idx + horizon + 1], dtype=np.float32)
    if tcp.shape[0] < horizon + 1:
        return None
    xyz_m = tcp[:, :3] / 1000.0
    rot = Rot.from_euler("xyz", tcp[:, 3:6], degrees=True)
    dxyz = (xyz_m[1:] - xyz_m[:-1]).astype(np.float32)
    drot = (rot[1:] * rot[:-1].inv()).as_rotvec().astype(np.float32)
    grasp_pm1 = (2.0 * tcp[:-1, 6] - 1.0).astype(np.float32).reshape(-1, 1)
    return np.concatenate([dxyz, drot, grasp_pm1], axis=1)


def find_grasp_close(g: zarr.Group, ep_start: int, ep_end: int) -> int | None:
    grasp = np.asarray(g["data/state"][ep_start:ep_end, 6])
    trans = np.where(np.diff(grasp) > 0.5)[0]
    return ep_start + int(trans[0]) if len(trans) else None


def summarize(label: str, pred: np.ndarray, gt: np.ndarray) -> None:
    pdz_mm = pred[:, 2] * 1000.0
    gdz_mm = gt[:, 2] * 1000.0
    trend = "DOWN" if pdz_mm.mean() < -0.5 else ("UP" if pdz_mm.mean() > 0.5 else "flat")
    print(f"  {label}")
    print(f"    pred Δz/step (mm): {np.round(pdz_mm, 2)}")
    print(f"    gt   Δz/step (mm): {np.round(gdz_mm, 2)}")
    print(f"    pred grasp:        {np.round(pred[:, 6], 2)}")
    print(f"    gt   grasp:        {np.round(gt[:, 6], 2)}")
    err = np.abs(pred - gt)
    print(f"    |err| xyz mean: {err[:, :3].mean()*1000:.2f} mm   "
          f"|err| aa mean: {err[:, 3:6].mean():.4f} rad   "
          f"|err| grasp mean: {err[:, 6].mean():.3f}")
    print(f"    pred z-trend: {trend}   grasp transition: {pred[0, 6]:+.2f} -> {pred[-1, 6]:+.2f}")
    print()


def main(args: Args) -> None:
    print(f"[load] {args.checkpoint}")
    cfg = _config.get_config(args.train_config_name)
    policy = policy_config.create_trained_policy(cfg, str(args.checkpoint), default_prompt=args.prompt)
    print("[ok] policy loaded\n")

    g = zarr.open(str(args.zarr_path), mode="r")
    ep_ends = np.asarray(g["meta/episode_ends"][:])
    shown = 0
    for ep_idx in range(len(ep_ends)):
        if shown >= args.n_episodes:
            break
        ep_start = 0 if ep_idx == 0 else int(ep_ends[ep_idx - 1])
        ep_end = int(ep_ends[ep_idx])
        gclose = find_grasp_close(g, ep_start, ep_end)
        if gclose is None:
            continue
        print(f"=== episode {ep_idx}  (frames {ep_start}..{ep_end}, grasp-close at {gclose}) ===")
        phases = [
            ("approach", ep_start + min(15, (ep_end - ep_start) // 4)),
            ("pre-grasp", max(ep_start, gclose - 5)),
            ("lift",      min(ep_end - args.horizon - 1, gclose + 3)),
        ]
        for label, t in phases:
            if t + args.horizon + 1 > ep_end:
                print(f"  {label}: SKIP (frame {t} too close to end)\n")
                continue
            obs = build_obs(g, t, args.prompt)
            pred = np.asarray(policy.infer(obs)["actions"])
            gt = ground_truth_chunk(g, t, args.horizon)
            if gt is None:
                continue
            summarize(f"{label}  [frame={t}  state z={obs['observation/state'][2]:.3f}m  grasp={obs['observation/state'][6]:.0f}]", pred, gt)
        shown += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Trained step-N checkpoint dir")
    p.add_argument("--train-config-name", type=str, default="pi05_xarm_finetune_lora")
    p.add_argument("--zarr-path", type=Path, default=Path("/data/edward/teleop_data.zarr"))
    p.add_argument("--prompt", type=str, default="pick up the red block")
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--horizon", type=int, default=10)
    args = p.parse_args()
    main(Args(**vars(args)))
