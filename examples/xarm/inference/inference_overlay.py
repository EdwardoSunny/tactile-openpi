"""
Inference-time tactile overlay — matches the training-time renderer exactly.

The training data was rendered by phone_data_collection/scripts/render_overlays.py
with normalization stats from /meta/normalization (written by
scripts/compute_overlay_normalization.py). Those stats encode:

  - raw_clip_low / raw_clip_high  per-cell Hampel bounds applied to raw tactile
  - episode_offsets               per-EPISODE mean of first ~15 idle frames
                                   (at inference, replaced by the per-rollout
                                    baseline captured ~1.5s after homing —
                                    semantically equivalent)
  - scale_xy / scale_z            cross-task pooled per-finger scales (median
                                   of per-episode p95 of |centered|)
  - deadband                      adaptive noise gate

This module loads those stats (from an npz produced by
extract_overlay_norm.py — pooled across the training task zarrs) and reapplies
the SAME pipeline to live camera frames before they're sent to the policy.
Without this, the policy sees raw camera frames at inference while it was
trained on overlay-augmented frames — a guaranteed distribution shift.

Dependencies: the phone_data_collection repo provides the underlying
SensorOverlay / SensorDrawer / SensorNormalizer + per-finger calibrations.
We import them lazily (only when the InferenceOverlay is actually
instantiated) so the rest of run_xarm_inference.py works in dry-run mode
without that repo on PYTHONPATH.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np


NATIVE_W, NATIVE_H = 640, 480


class InferenceOverlay:
    """Draws the tactile overlay onto a single camera frame at inference time.

    Construction:
        ov = InferenceOverlay(
            stats_path="/path/to/overlay_norm.npz",
            phone_data_collection_repo="/home/u-ril/edward/phone_data_collection",
            baseline=baseline_2_9_3,  # captured by _capture_tactile_baseline
        )

    Per frame:
        out_bgr = ov.apply(
            bgr_uint8_HWC,         # 224x224x3 BGR uint8 from CameraManager
            role,                  # "side" (agent cam) or "wrist"
            joint_angles_deg,      # iterable of 7 servo angles
            grip_pos_raw,          # float, 0..850 from arm.get_gripper_position
            raw_L, raw_R,          # (9, 3) tactile xyz per finger from tactile.get_latest()
        )
        # out_bgr has the SAME shape/dtype as bgr_uint8_HWC, with the overlay
        # variant drawn on top. Ready to be packed into obs["observation/image"].

    Color order: BGR in, BGR out — matches the training-side renderer
    (phone_data_collection/scripts/render_overlays.py), which operates on
    BGR bytes from the raw zarr and writes BGR bytes back into img_{i}_<key>.
    Those BGR bytes are later relabeled as "RGB" by PIL inside
    LeRobotDataset.add_frame (image_writer.image_array_to_pil_image), so the
    model is trained on BGR-byte-content under the name `observation/image`.
    Sending true RGB at inference would produce an R/B-channel-swapped
    distribution shift; do NOT pre-convert to RGB.

    The baseline replaces the training-time per-episode offset. Update it
    whenever you re-home + recapture (call set_baseline). Without a baseline
    the overlay falls back to "raw - 0" which puts the arrows in raw-count
    units and gates everything below the deadband — i.e. blank arrows.
    """

    def __init__(
        self,
        stats_path: str,
        phone_data_collection_repo: str,
        baseline: Optional[np.ndarray] = None,
        mode_key_override: Optional[str] = None,
    ) -> None:
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"overlay stats not found: {stats_path}")

        if phone_data_collection_repo and phone_data_collection_repo not in sys.path:
            sys.path.insert(0, phone_data_collection_repo)
        try:
            from environment.tactile_overlay import SensorOverlay, apply_deadband  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"Could not import environment.tactile_overlay from "
                f"{phone_data_collection_repo!r}. Set the right path via "
                f"--overlay-repo. Underlying error: {e}"
            ) from e
        self._apply_deadband = apply_deadband

        s = np.load(stats_path, allow_pickle=False)
        self.raw_clip_low = np.asarray(s["raw_clip_low"], dtype=np.float64)   # (2, 9, 3)
        self.raw_clip_high = np.asarray(s["raw_clip_high"], dtype=np.float64) # (2, 9, 3)
        self.scale_xy = np.asarray(s["scale_xy"], dtype=np.float32)           # (2,)
        self.scale_z = np.asarray(s["scale_z"], dtype=np.float32)             # (2,)
        self.deadband = float(s["deadband"])
        self.mode_key = mode_key_override or str(s["mode_key"])
        self.arrow_length_scale = float(s["arrow_length_scale"])
        self.arrow_thickness = int(s["arrow_thickness"])
        self.dot_size = int(s["dot_size"])

        # SensorOverlay holds one SensorDrawer per role + one SensorNormalizer
        # per finger. We construct with baseline=None and patch the normalizer
        # attributes directly so it applies the scales we loaded (with offset=0
        # since we subtract the per-rollout baseline upstream of normalize()).
        self.overlay = SensorOverlay(baseline=None)
        self.overlay.norm_L.offset = np.zeros((9, 3), dtype=np.float32)
        self.overlay.norm_L.global_scale_xy = float(self.scale_xy[0])
        self.overlay.norm_L.global_scale_z = float(self.scale_z[0])
        self.overlay.norm_R.offset = np.zeros((9, 3), dtype=np.float32)
        self.overlay.norm_R.global_scale_xy = float(self.scale_xy[1])
        self.overlay.norm_R.global_scale_z = float(self.scale_z[1])

        self._baseline = None
        if baseline is not None:
            self.set_baseline(baseline)

    def set_baseline(self, baseline: np.ndarray) -> None:
        """Install the per-rollout tactile baseline (mean of idle frames after
        homing). Equivalent to the per-episode offset used at training time."""
        arr = np.asarray(baseline, dtype=np.float32)
        if arr.shape != (2, 9, 3):
            raise ValueError(f"baseline must be (2, 9, 3); got {arr.shape}")
        self._baseline = arr

    def has_baseline(self) -> bool:
        return self._baseline is not None

    def apply(
        self,
        bgr_uint8_hwc: np.ndarray,
        role: str,
        joint_angles_deg,
        grip_pos_raw: float,
        raw_L: Optional[np.ndarray],
        raw_R: Optional[np.ndarray],
    ) -> np.ndarray:
        """Render the overlay onto `bgr_uint8_hwc` and return the result with
        identical shape + dtype.

        Color order is BGR throughout — see class docstring. The training-time
        renderer (render_overlays.py) draws into a BGR buffer and stores BGR
        bytes; we do the same here. No RGB<->BGR conversion at either end.

        If raw_L / raw_R are None or the baseline is missing, returns the
        input frame unchanged — same fail-open behavior as the training-time
        renderer when normalization fields are missing.
        """
        if raw_L is None or raw_R is None or self._baseline is None:
            return bgr_uint8_hwc

        H, W = bgr_uint8_hwc.shape[:2]

        # Stage 1: Hampel clip on raw, per-cell.
        L = np.clip(np.asarray(raw_L, dtype=np.float64),
                    self.raw_clip_low[0], self.raw_clip_high[0])
        R = np.clip(np.asarray(raw_R, dtype=np.float64),
                    self.raw_clip_low[1], self.raw_clip_high[1])

        # Stage 2: subtract per-rollout baseline (= the per-episode offset at training).
        cL = L - self._baseline[0]
        cR = R - self._baseline[1]

        # Stages 3/5: normalizer applies the patched scales; then deadband.
        nL, nR = self.overlay.normalize(cL, cR)
        nL = self._apply_deadband(nL, self.deadband)
        nR = self._apply_deadband(nR, self.deadband)

        # Stage 4: upscale to 640x480 (where sensordrawing's K and T_rc are
        # calibrated), draw, then downsize back to the input HxW. Buffer is
        # BGR throughout — same as render_overlays.py.
        bgr_native = cv2.resize(bgr_uint8_hwc, (NATIVE_W, NATIVE_H))
        drawn_native = self.overlay.draw(
            role, bgr_native, joint_angles_deg, float(grip_pos_raw),
            nL, nR,
            mode_key=self.mode_key,
            arrow_length_scale=self.arrow_length_scale,
            arrow_thickness=self.arrow_thickness,
            dot_size=self.dot_size,
        )
        return cv2.resize(drawn_native, (W, H))
