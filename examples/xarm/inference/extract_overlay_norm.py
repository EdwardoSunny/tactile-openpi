"""
Extract the overlay-normalization stats from one or more training overlay
zarrs and pack them into a single portable npz that ships alongside the
trained model checkpoint.

The training renderer (phone_data_collection/scripts/render_overlays.py)
reads stats from each zarr's /meta/normalization, which was written by
phone_data_collection/scripts/compute_overlay_normalization.py. Those stats
contain:
  - raw_clip_low / raw_clip_high  (2, 9, 3)  per-task Hampel bounds
  - scale_xy / scale_z            (2,)        shared across all pooled tasks
  - deadband                      scalar      shared across all pooled tasks
  - n_baseline_frames, mad_k, percentile      scalars (the knobs used)
  - attrs.source_zarrs                        which zarrs were pooled

When the model was trained on a SINGLE task zarr, the npz is a 1:1 copy of
the loaded stats. When the model was trained on MULTIPLE task zarrs, this
script POOLS them:
  - raw_clip_low  -> elementwise min  (most permissive lower bound)
  - raw_clip_high -> elementwise max  (most permissive upper bound)
  - scale_xy, scale_z, deadband       must already be identical across the
                                       inputs (they were computed jointly by
                                       compute_overlay_normalization.py over
                                       all input zarrs together — if you ran
                                       that script with the same set of inputs,
                                       they match exactly).

The output npz also bundles the rendering parameters that determine arrow
size on screen, so inference is fully self-describing:
  - mode_key, arrow_length_scale, arrow_thickness, dot_size

Usage
-----
    python extract_overlay_norm.py \
        /home/u-ril/edward/phone_data_collection/teleop_data_cube_overlay.zarr \
        /home/u-ril/edward/phone_data_collection/teleop_data_charger_overlay.zarr \
        /home/u-ril/edward/phone_data_collection/teleop_data_dishwasher_overlay.zarr \
        /home/u-ril/edward/phone_data_collection/teleop_data_tube_overlay.zarr \
        --mode points1_arrow --out overlay_norm.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import zarr


# These mirror the per-mode defaults in
# phone_data_collection/environment/tactile_overlay.py:MODES. If the user
# tunes the per-mode arrow_length_scale upstream, update this table to match
# (or pass --arrow-length-scale on the CLI).
DEFAULT_ARROW_LENGTH_SCALE = {
    "points9_arrow": 0.06,
    "points1_arrow": 0.02,
    "points1_contact_spatial": 0.12,
    "points9_color_spatial": 0.12,
    "points1_contact_flat": 0.12,
    "points9_color_flat": 0.12,
}
# Match BOLD_ARROW_THICKNESS / BOLD_DOT_SIZE in
# phone_data_collection/environment/tactile_overlay.py used by the renderer.
DEFAULT_ARROW_THICKNESS = 8
DEFAULT_DOT_SIZE = 22


def _load_one(zarr_path: str) -> dict:
    """Read /meta/normalization out of one overlay zarr."""
    if not os.path.isdir(zarr_path):
        raise SystemExit(f"[error] zarr not found: {zarr_path}")
    z = zarr.open(zarr_path, mode="r")
    if "meta" not in z or "normalization" not in z["meta"]:
        raise SystemExit(
            f"[error] {zarr_path} has no /meta/normalization. Run "
            f"phone_data_collection/scripts/compute_overlay_normalization.py "
            f"against the source raw zarrs first."
        )
    n = z["meta/normalization"]
    return {
        "raw_clip_low":  np.asarray(n["raw_clip_low"][:],  dtype=np.float32),
        "raw_clip_high": np.asarray(n["raw_clip_high"][:], dtype=np.float32),
        "scale_xy":      np.asarray(n["scale_xy"][:],      dtype=np.float32),
        "scale_z":       np.asarray(n["scale_z"][:],       dtype=np.float32),
        "deadband":      float(np.asarray(n["deadband"])),
    }


def _pool(per_task: list[dict]) -> dict:
    """Pool stats across multiple task zarrs. See module docstring."""
    if not per_task:
        raise ValueError("need at least one input zarr")
    raw_clip_low = np.min(np.stack([t["raw_clip_low"]  for t in per_task]), axis=0)
    raw_clip_high = np.max(np.stack([t["raw_clip_high"] for t in per_task]), axis=0)

    # scales + deadband should already be identical if compute_overlay_normalization.py
    # was run with this same set of inputs (the pooled values are jointly
    # derived). Warn loudly if they're not.
    def _check_shared(key: str):
        vals = np.stack([np.asarray(t[key]).ravel() for t in per_task])
        if not np.allclose(vals, vals[0:1], rtol=1e-3, atol=1e-3):
            print(f"  [warn] {key} differs across inputs (max disagreement "
                  f"{float(np.abs(vals - vals[0:1]).max()):.4g}). Was "
                  f"compute_overlay_normalization.py run with the SAME "
                  f"--zarrs list for all of these? Falling back to mean.")
            return vals.mean(axis=0)
        return vals[0]

    scale_xy = _check_shared("scale_xy")
    scale_z = _check_shared("scale_z")
    deadband_arr = np.stack([np.asarray(t["deadband"]).reshape(()) for t in per_task])
    if not np.allclose(deadband_arr, deadband_arr[0], rtol=1e-3, atol=1e-3):
        print(f"  [warn] deadband differs across inputs; using mean")
    deadband = float(deadband_arr.mean())

    return {
        "raw_clip_low":  raw_clip_low.astype(np.float32),
        "raw_clip_high": raw_clip_high.astype(np.float32),
        "scale_xy":      np.asarray(scale_xy, dtype=np.float32),
        "scale_z":       np.asarray(scale_z, dtype=np.float32),
        "deadband":      np.float32(deadband),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zarrs", nargs="+",
                    help="One or more training overlay zarrs (must have "
                         "/meta/normalization; written by "
                         "compute_overlay_normalization.py).")
    ap.add_argument("--mode", required=True,
                    choices=list(DEFAULT_ARROW_LENGTH_SCALE.keys()),
                    help="Which overlay variant the trained model was trained "
                         "on (and which one inference should redraw).")
    ap.add_argument("--arrow-length-scale", type=float, default=None,
                    help="Override the per-mode default arrow length scale "
                         "(use this if you tuned environment/tactile_overlay.py:MODES).")
    ap.add_argument("--arrow-thickness", type=int, default=DEFAULT_ARROW_THICKNESS,
                    help=f"Arrow line thickness in px (default: {DEFAULT_ARROW_THICKNESS}, "
                         f"matches BOLD_ARROW_THICKNESS).")
    ap.add_argument("--dot-size", type=int, default=DEFAULT_DOT_SIZE,
                    help=f"Sensor dot size in px (default: {DEFAULT_DOT_SIZE}, "
                         f"matches BOLD_DOT_SIZE).")
    ap.add_argument("--out", required=True, help="Output .npz path.")
    args = ap.parse_args()

    print(f"Reading /meta/normalization from {len(args.zarrs)} zarr(s):")
    per_task = []
    for p in args.zarrs:
        print(f"  - {p}")
        per_task.append(_load_one(p))

    pooled = _pool(per_task)
    arrow_length_scale = (args.arrow_length_scale
                           if args.arrow_length_scale is not None
                           else DEFAULT_ARROW_LENGTH_SCALE[args.mode])

    print()
    print(f"  raw_clip_low  range : [{pooled['raw_clip_low'].min():.1f}, "
          f"{pooled['raw_clip_low'].max():.1f}]")
    print(f"  raw_clip_high range : [{pooled['raw_clip_high'].min():.1f}, "
          f"{pooled['raw_clip_high'].max():.1f}]")
    print(f"  scale_xy            : LEFT={pooled['scale_xy'][0]:.1f}  "
          f"RIGHT={pooled['scale_xy'][1]:.1f}")
    print(f"  scale_z             : LEFT={pooled['scale_z'][0]:.1f}  "
          f"RIGHT={pooled['scale_z'][1]:.1f}")
    print(f"  deadband            : {float(pooled['deadband']):.4f}")
    print(f"  mode_key            : {args.mode}")
    print(f"  arrow_length_scale  : {arrow_length_scale}")
    print(f"  arrow_thickness     : {args.arrow_thickness}")
    print(f"  dot_size            : {args.dot_size}")

    np.savez(
        args.out,
        raw_clip_low=pooled["raw_clip_low"],
        raw_clip_high=pooled["raw_clip_high"],
        scale_xy=pooled["scale_xy"],
        scale_z=pooled["scale_z"],
        deadband=pooled["deadband"],
        # Fixed-length unicode (NOT object dtype) so allow_pickle=False loading works.
        mode_key=np.array(args.mode, dtype="<U64"),
        arrow_length_scale=np.float32(arrow_length_scale),
        arrow_thickness=np.int32(args.arrow_thickness),
        dot_size=np.int32(args.dot_size),
    )
    print(f"\nWrote {args.out}")
    print(f"\nUse with run_xarm_inference.py via:")
    print(f"  --overlay-stats {args.out}")


if __name__ == "__main__":
    main()
