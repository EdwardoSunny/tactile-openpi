"""Real-robot closed-loop inference for a pi0.5 LoRA fine-tune on xArm.

Loads a checkpoint produced by `scripts/train.py pi05_xarm_finetune_lora`, captures images
from two cameras, reads the xArm end-effector pose + gripper state, queries the policy for
an action chunk, and executes the chunk on the robot at 10 Hz with re-planning.

This is the single-process equivalent of `scripts/serve_policy.py` + an external client:
inference and control run in the same Python process, no websocket. Use this when the
robot host and the GPU host are the same machine. If they're not, run `serve_policy.py`
on the GPU host and talk to it via `openpi_client.websocket_client_policy` (see
docs/remote_inference.md).

Hardware:
    - xArm 6/7 robot (reachable via IP)
    - CUDA-capable GPU (model needs ~12 GB)
    - 2 cameras: agent-view + wrist-view. Either RealSense (default, matches the
      phone-teleop collection setup) or USB webcams.

Setup (run once):
    /data/edward/openpi/.venv/bin/pip install xArm-Python-SDK pyrealsense2 opencv-python

Usage:
    # Real run on the robot:
    /data/edward/openpi/.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
        --checkpoint /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999 \
        --xarm-ip 192.168.1.223

    # Dry run (no robot connection — just captures cams, runs inference, prints actions):
    /data/edward/openpi/.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
        --checkpoint /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999 \
        --dry-run

Safety:
    - Ctrl+C triggers emergency stop and homes the robot before exit.
    - --max-action-norm clips per-step Cartesian motion (default 0.05 m = 5 cm).
    - --action-scale globally scales delta actions (default 1.0; set 0.5 for half-speed).
    - --dry-run runs the full inference loop with no robot servo commands.
    - The first servo command is blocked behind a manual ENTER prompt unless --auto-start.

Tactile gripper safety (mandatory):
    Two A31301 ESP32 boards (one per finger) stream a 3x3 Hall grid each over USB
    serial. We mirror the gripper-safety wrapper from
    /data/edward/tactile-data-collection (XArm._apply_tactile_safety): every control
    tick, evaluate the per-cell delta-from-idle metric (default sum_abs_z over the
    18 connected taxels); if it exceeds --safety-threshold OR the readings are stale,
    a CLOSING command is clamped to the previous grasp value — the gripper freezes
    in place. Opening is always allowed. This is the same logic the data was
    collected under, so the deployment-time behavior matches training.

    Tactile is REQUIRED. If both boards aren't streaming live data within
    --tactile-init-timeout seconds, the script hard-fails before homing the
    robot. The continuous-grasp mapping (action[6] in [-1,+1] -> gripper
    position in [850,0]) replaces the previous binary 0/850 step so the
    safety clamp actually corresponds to a physical "stop here" rather than
    "next tick try to slam to 0 again".

Tactile arrow overlay (matches training-time `img_{0,1}_points1_arrow` etc.):
    Easy path — `--overlay <mode>`:
        Use the bundled pooled npz for that mode. Resolves to
        `examples/xarm/inference/overlay_norm_<mode>.npz`. ONE file works for
        all 4 tasks because scale_xy/scale_z/deadband are identical across
        tasks (they were computed jointly by phone_data_collection's
        compute_overlay_normalization.py over all task zarrs); only raw_clip
        bounds differ, and the pooled file uses the union (elementwise
        min/max) which is the most permissive correct choice.
        Example: --overlay points9_arrow

    Power-user path — `--overlay-stats <path>` (+ optional `--overlay-mode-key`):
        Explicit path to any overlay_norm.npz (e.g. a per-task variant
        you produced yourself with extract_overlay_norm.py). Use this if
        you have task-specific clip bounds you want to honor exactly.

    Without either flag, raw frames are sent to the policy — only correct
    for checkpoints trained on raw img_0/img_1 (the *_baseline_lora
    configs).

Manifeel inference path (`pi05_xarm_<task>_manifeel_baseline_lora`):
    Pass `--manifeel`. Camera frames are NOT modified; instead a synthetic
    tactile-only image (sensordrawing "third_image" mode — two 3x3 arrow
    grids on black, color=z normal force, direction=xy shear) is rendered
    per tick from live tactile readings and packed as
    `observation/tactile_image`. LiberoManifeelInputs slots that into
    right_wrist_0_rgb with image_mask=True, so the model sees:
        - agent camera (raw)
        - wrist camera (raw)
        - tactile third_image (synthetic, this tick's tactile)
    matching exactly what the training pipeline produced. Video recording
    switches to 3-up (agent | wrist | manifeel). Mutually exclusive with
    --overlay / --overlay-stats.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Allow `tactile_safety` (sibling file in this directory) to be importable when
# the script is run as `python examples/xarm/inference/run_xarm_inference.py ...`
# rather than as a package. Has to happen before the `from tactile_safety` import.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# openpi imports (must be importable; the .venv created by `uv sync` has them).
from openpi.policies import policy_config
from openpi.training import config as _config
from tactile_safety import TactileConfig, TactileSensors


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
log = logging.getLogger("xarm-infer")
log.setLevel(logging.INFO)


# ───────── config dataclass ─────────


@dataclasses.dataclass
class XArmInferenceConfig:
    checkpoint: Path
    train_config_name: str = "pi05_xarm_finetune_lora"

    # Robot
    xarm_ip: str = "192.168.1.223"
    control_hz: int = 10
    # Joint-space home pose (7 joint angles in degrees) — the rig's known-good rest
    # configuration from ril_env.xarm_controller.XArmConfig.home_pos. Homing in
    # joint space (set_servo_angle) is more robust than Cartesian (set_position)
    # because it avoids IK ambiguity. For reference, this maps to TCP pose
    # ≈ (475.79 mm, -1.14 mm, 244.72 mm, 179.13°, -0.01°, 0.78°) on the lab xArm 7.
    home_joint_angles_deg: Tuple[float, float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 70.0, 0.0, 70.0, 0.0,
    )
    home_speed: float = 50.0  # deg/s for the homing motion

    # Cameras
    # Two paths are supported:
    #   - RealSense (default): goes through pyrealsense2. agent/wrist IDs may be either
    #     a 12-digit serial number ('327122079374') OR a small int (0, 1) interpreted
    #     as an index into the enumerated device list. If unset, the first two D-series
    #     devices found are used (agent=0, wrist=1).
    #   - OpenCV (--no-realsense): cv2.VideoCapture with USB indices like 0, 2 — useful
    #     when only USB webcams are attached.
    agent_cam_id: int | str | None = "327122079374"
    wrist_cam_id: int | str | None = "332322072612"
    use_realsense: bool = True
    # RealSense capture parameters — must match the collection-time settings, otherwise
    # the deployment-time pixel distribution differs from training and the policy
    # silently collapses to a noop attractor. Defaults below mirror the vla-finetune
    # collect/inference pipeline (1280x720 @ 30 fps, manual exposure=120, gain=0,
    # white_balance=5900K, all auto disabled).
    rs_width: int = 1280
    rs_height: int = 720
    rs_fps: int = 30
    rs_exposure: int = 120
    rs_gain: int = 0
    rs_white_balance: int = 5900
    image_size: int = 224

    # Task
    prompt: str = "pick up the red block"
    max_steps: int = 200
    replan_steps: int = 5  # consume this many actions from each chunk before re-planning

    # Safety
    action_scale: float = 1.0
    max_action_norm: float = 0.05  # 5 cm Cartesian per step (rad+m mixed but xArm tolerances reasonable)
    auto_start: bool = False
    dry_run: bool = False

    # Tactile safety (REQUIRED — script hard-fails if these aren't streaming).
    # Defaults mirror tactile-data-collection/tactile_config.py: two A31301 boards
    # on /dev/ttyACM{0,1}, sum_abs_z metric in delta-from-idle mode with a
    # threshold of 1500 (idle-noise band ~50 counts; firm grip ~1500-2000).
    left_port: str = "/dev/ttyACM0"
    right_port: str = "/dev/ttyACM1"
    tactile_baud: int = 115200
    safety_metric: str = "sum_abs_z"
    safety_threshold: float = 1500.0
    stale_after_sec: float = 0.2
    baseline_duration_sec: float = 1.5    # how long to sample at-rest before the rollout
    tactile_init_timeout_sec: float = 5.0  # how long to wait for both boards to publish

    # Video
    save_video: bool = True
    video_path: Path | None = None

    # Live operator visualization window.
    # If True, pops a cv2 window each control tick showing agent | wrist with the
    # tactile force arrows drawn (mirrors what the collection-time --viz window shows).
    # Model input path is unchanged by this flag alone — this is operator debug only.
    show_windows: bool = False
    # Path to the phone_data_collection repo (provides environment.tactile_overlay.SensorOverlay).
    # Override if the repo lives elsewhere on the deployment host. Shared by both
    # the operator-only LiveViz path AND the policy-input InferenceOverlay path.
    viz_overlay_repo: str = "/home/u-ril/edward/phone_data_collection"
    viz_mode_key: str = "points1_arrow"  # closest live equivalent to the legacy 'arrow' overlay

    # Policy-input tactile overlay (MUST match the training-time renderer).
    # If `overlay_stats_path` is set, every camera frame fed to the policy gets
    # the overlay drawn on it using stats extracted from the training overlay
    # zarrs by examples/xarm/inference/extract_overlay_norm.py. The pipeline
    # matches phone_data_collection/scripts/render_overlays.py exactly:
    # Hampel raw clip -> subtract per-rollout baseline -> normalize with
    # cross-task pooled scales -> adaptive deadband -> SensorOverlay.draw.
    # If unset, no overlay is applied — only correct when the checkpoint was
    # trained on raw frames (no overlay variant). Default unset to preserve
    # legacy behavior; flip via --overlay-stats <path>.
    overlay_stats_path: str | None = None
    # Optional override of the mode_key stored inside the stats npz. Use this
    # if you want to render a different variant than the one extract_overlay_norm.py
    # baked in (rare; normally leave it None to match training).
    overlay_mode_key: str | None = None
    # Manifeel mode: agent + wrist camera frames are NOT modified; a synthetic
    # tactile-only image (sensordrawing third_image) is rendered per-tick and
    # passed as observation/tactile_image. Mutually exclusive with overlay_*
    # paths. For pi05_xarm_<task>_manifeel_baseline_lora checkpoints.
    manifeel: bool = False
    # Path to the manifeel overlay-norm npz (used for scale_xy/scale_z/deadband
    # only; the actual rendering mode is hardcoded to sensordrawing "third_image"
    # in ManifeelRenderer).
    manifeel_stats_path: str | None = None

    # Flow-matching sampler seed. None -> derive from wall clock + os.urandom so
    # each fresh launch gets a different noise sequence (and therefore a
    # different action chunk for the same observation). Pass an int to
    # reproduce a specific rollout. Logged on policy load so you can replay.
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.save_video and self.video_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.video_path = Path(f"rollouts/xarm_inference_{stamp}.mp4")


# ───────── tactile overlay hook ─────────
#
# The actual overlay rendering at inference time happens in
# inference_overlay.InferenceOverlay (built in run() after the per-rollout
# tactile baseline is captured). It mirrors the training-time renderer
# (phone_data_collection/scripts/render_overlays.py) exactly. We don't ship
# an identity-stub anymore — opt in by passing --overlay-stats <path>.


# ───────── live operator visualization ─────────


class LiveViz:
    """Operator-only side-by-side cv2 window with tactile arrows drawn.

    Uses environment.tactile_overlay.SensorOverlay from the phone_data_collection
    repo (lazy import — only when --show-windows is set). The model never sees
    these annotated frames; only the on-screen window is augmented.

    Arrows are drawn at native 640x480 (where the sensordrawing kinematics
    and per-finger calibration were computed) on a copy of each camera frame.
    """

    WINDOW_NAME = "tactile-openpi rollout (agent | wrist)"

    def __init__(self, cfg: "XArmInferenceConfig", baseline: np.ndarray) -> None:
        self.cfg = cfg
        if cfg.viz_overlay_repo and cfg.viz_overlay_repo not in sys.path:
            sys.path.insert(0, cfg.viz_overlay_repo)
        try:
            from environment.tactile_overlay import SensorOverlay  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"--show-windows requires environment.tactile_overlay from "
                f"phone_data_collection (looked at {cfg.viz_overlay_repo!r}). "
                f"Set --viz-overlay-repo to the right path or drop --show-windows. "
                f"Underlying error: {e}"
            ) from e
        self.overlay = SensorOverlay(baseline=baseline)
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 480)
        log.info("Live viz window opened (mode=%s).", cfg.viz_mode_key)

    def update(
        self,
        agent_bgr: np.ndarray,
        wrist_bgr: np.ndarray,
        joint_angles_deg,
        grip_pos_raw: float,
        raw_L: np.ndarray | None,
        raw_R: np.ndarray | None,
    ) -> None:
        # SensorOverlay + cv2.imshow both expect BGR — inputs are already BGR
        # (the model-input channel order matches training; see CameraManager).
        agent_640 = cv2.resize(agent_bgr, (640, 480))
        wrist_640 = cv2.resize(wrist_bgr, (640, 480))
        nL, nR = self.overlay.normalize(raw_L, raw_R)
        try:
            agent_drawn = self.overlay.draw(
                "side", agent_640, joint_angles_deg, float(grip_pos_raw),
                nL, nR, mode_key=self.cfg.viz_mode_key,
            )
            wrist_drawn = self.overlay.draw(
                "wrist", wrist_640, joint_angles_deg, float(grip_pos_raw),
                nL, nR, mode_key=self.cfg.viz_mode_key,
            )
        except Exception as e:
            log.warning("[viz] overlay draw failed: %s — showing plain frames", e)
            agent_drawn, wrist_drawn = agent_640, wrist_640
        composite = np.concatenate([agent_drawn, wrist_drawn], axis=1)
        cv2.imshow(self.WINDOW_NAME, composite)
        cv2.waitKey(1)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except cv2.error:
            pass


# ───────── cameras ─────────


class ThreadedCamera:
    """Mirrors collect_xarm_demos.py's ThreadedCamera. cv2.VideoCapture in a background
    thread that always exposes the latest frame as BGR uint8, already resized to
    (image_size, image_size). Fails fast if the device isn't producing frames.

    Color order is BGR (the native order cv2.VideoCapture returns + the order
    phone_data_collection records and stores). Do NOT convert to RGB here —
    see inference_overlay.InferenceOverlay class docstring for why."""

    def __init__(self, source: int | str, image_size: int = 224) -> None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source!r}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize buffer for freshest frame
        self.image_size = image_size
        self._latest: np.ndarray | None = None
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        # Warm-up — ~1 s. Raise if the camera never produces frames.
        for _ in range(20):
            if self._latest is not None:
                break
            time.sleep(0.05)
        if self._latest is None:
            raise RuntimeError(f"Camera {source!r} did not produce frames after warm-up")

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame_bgr = self.cap.read()
            if not ok or frame_bgr is None:
                time.sleep(0.005)
                continue
            frame_bgr = cv2.resize(
                frame_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
            self._latest = frame_bgr

    def read(self) -> np.ndarray:
        assert self._latest is not None
        return self._latest.copy()

    def close(self) -> None:
        self._stop.set()
        self._th.join(timeout=1.0)
        self.cap.release()


class CameraManager:
    """Returns (agent, wrist) BGR uint8 frames at `image_size`×`image_size`.

    Defaults to RealSense (pyrealsense2). Set `use_realsense=False` to fall back
    to plain OpenCV USB capture (cv2.VideoCapture with integer USB indices),
    which is what collect_xarm_demos.py uses.

    Color order is BGR — matches phone_data_collection's capture path
    (rs.format.bgr8) and what gets stored in the training-time zarrs. See
    inference_overlay.InferenceOverlay class docstring for the full chain.
    """

    def __init__(self, cfg: XArmInferenceConfig) -> None:
        self.cfg = cfg
        if cfg.use_realsense:
            self._init_realsense()
        else:
            self._init_threaded(cfg)

    def _init_threaded(self, cfg: XArmInferenceConfig) -> None:
        def _maybe_int(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return v  # leave as path / string source

        agent_src = _maybe_int(cfg.agent_cam_id if cfg.agent_cam_id is not None else 0)
        wrist_src = _maybe_int(cfg.wrist_cam_id if cfg.wrist_cam_id is not None else 2)
        log.info("Opening cameras via cv2.VideoCapture: agent=%r, wrist=%r (matching collect_xarm_demos.py)",
                 agent_src, wrist_src)
        self.agent_cam = ThreadedCamera(agent_src, cfg.image_size)
        self.wrist_cam = ThreadedCamera(wrist_src, cfg.image_size)

    @staticmethod
    def _resolve_realsense_serial(rs_module, raw, available: list[str], default_idx: int) -> str:
        """Map a CLI cam-id (None, an int-like 0/1, or a 12-digit serial) to a serial.

        Pure logic, no side-effects — easy to test if anyone wants to.
        """
        if raw is None:
            if default_idx >= len(available):
                raise RuntimeError(
                    f"Need {default_idx + 1} RealSense device(s) but only {len(available)} found: {available}"
                )
            return available[default_idx]
        raw_str = str(raw)
        # 12-digit serial? Use it directly (validate it's actually attached).
        if raw_str.isdigit() and len(raw_str) >= 6:
            if raw_str not in available:
                raise RuntimeError(
                    f"RealSense serial {raw_str!r} not found among attached devices {available}"
                )
            return raw_str
        # Small int? Treat as index into the enumerated device list.
        try:
            idx = int(raw_str)
        except ValueError as e:
            raise RuntimeError(f"Bad --*-cam-id value: {raw!r}") from e
        if idx < 0 or idx >= len(available):
            raise RuntimeError(
                f"RealSense index {idx} out of range; only {len(available)} device(s): {available}"
            )
        return available[idx]

    def _init_realsense(self) -> None:
        """Open both RealSense color streams.

        Mirrors the vla-finetune CameraManager (experiments/robot/xarm/run_xarm_inference.py):
        1280x720 BGR8 @ 30 fps, manual exposure=120, gain=0, white_balance=5900K, with
        all auto- modes disabled. These settings have to match the data collection
        pipeline otherwise the model sees an out-of-distribution color cast / exposure
        and collapses to predicting noop actions.
        """
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise SystemExit(
                "pyrealsense2 not found. Install with: "
                "uv pip install pyrealsense2  (or .venv/bin/pip install pyrealsense2)"
            ) from e

        ctx = rs.context()
        attached = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
        if not attached:
            raise RuntimeError("No RealSense devices found. Plug in the cameras or use --no-realsense.")
        log.info("Found %d RealSense device(s): %s", len(attached), attached)

        agent_serial = self._resolve_realsense_serial(rs, self.cfg.agent_cam_id, attached, default_idx=0)
        wrist_serial = self._resolve_realsense_serial(rs, self.cfg.wrist_cam_id, attached, default_idx=1)
        if agent_serial == wrist_serial:
            raise RuntimeError(
                f"agent and wrist resolved to the same camera ({agent_serial}); "
                f"available serials: {attached}"
            )

        self.agent_pipe = rs.pipeline()
        self.wrist_pipe = rs.pipeline()
        profiles = {}
        for pipe, serial, label in ((self.agent_pipe, agent_serial, "agent"),
                                    (self.wrist_pipe, wrist_serial, "wrist")):
            cfg_rs = rs.config()
            cfg_rs.enable_device(serial)
            cfg_rs.enable_stream(
                rs.stream.color, self.cfg.rs_width, self.cfg.rs_height, rs.format.bgr8, self.cfg.rs_fps,
            )
            try:
                profiles[label] = pipe.start(cfg_rs)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to start RealSense pipeline for {label}={serial}: {e}. "
                    f"Is another process holding the camera? Try `pkill -f realsense` or replug."
                ) from e

        # Apply manual exposure / white-balance to match collection. Color sensor is
        # index 1 on the D435 (index 0 is the stereo/IR module).
        for label, profile in profiles.items():
            color_sensor = None
            for s in profile.get_device().query_sensors():
                if s.supports(rs.option.enable_auto_exposure) and s.supports(rs.option.white_balance):
                    color_sensor = s
                    break
            if color_sensor is None:
                log.warning("Could not locate color sensor for %s; skipping manual exposure/WB.", label)
                continue
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, self.cfg.rs_exposure)
            color_sensor.set_option(rs.option.gain, self.cfg.rs_gain)
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            color_sensor.set_option(rs.option.white_balance, self.cfg.rs_white_balance)

        # Warm-up: AE/AWB take a few frames to converge even when disabled.
        for _ in range(30):
            self.agent_pipe.wait_for_frames()
            self.wrist_pipe.wait_for_frames()

        self._agent_serial = agent_serial
        self._wrist_serial = wrist_serial
        log.info(
            "RealSense ready: agent=%s, wrist=%s  (%dx%d @ %d fps, exposure=%d, wb=%dK)",
            agent_serial, wrist_serial,
            self.cfg.rs_width, self.cfg.rs_height, self.cfg.rs_fps,
            self.cfg.rs_exposure, self.cfg.rs_white_balance,
        )

    def get_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.use_realsense:
            agent_frames = self.agent_pipe.wait_for_frames()
            wrist_frames = self.wrist_pipe.wait_for_frames()
            agent_color = agent_frames.get_color_frame()
            wrist_color = wrist_frames.get_color_frame()
            if not agent_color or not wrist_color:
                raise RuntimeError("RealSense returned an empty color frame")
            agent_bgr = np.asanyarray(agent_color.get_data())
            wrist_bgr = np.asanyarray(wrist_color.get_data())
            # Keep BGR — matches phone_data_collection/environment/cameras.py
            # (rs.format.bgr8) and the training-time zarr byte order. Do NOT
            # convert to RGB; see InferenceOverlay class docstring.
            agent = cv2.resize(agent_bgr,
                               (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
            wrist = cv2.resize(wrist_bgr,
                               (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
        else:
            agent = self.agent_cam.read()  # already BGR uint8 at image_size
            wrist = self.wrist_cam.read()
        # Overlay is NOT applied here — it needs joint_angles, grip_pos, and
        # tactile readings, which CameraManager doesn't have. The main loop
        # (run()) applies the overlay via InferenceOverlay after fetching
        # those inputs, before packing the obs dict.
        return agent, wrist

    def close(self) -> None:
        if self.cfg.use_realsense:
            for pipe in (getattr(self, "agent_pipe", None), getattr(self, "wrist_pipe", None)):
                if pipe is not None:
                    try:
                        pipe.stop()
                    except Exception as e:
                        log.warning("RealSense pipe.stop() failed: %s", e)
        else:
            if hasattr(self, "agent_cam"):
                self.agent_cam.close()
            if hasattr(self, "wrist_cam"):
                self.wrist_cam.close()


# ───────── xArm controller ─────────


class XArmController:
    # Gripper bounds in xArm SDK units. grasp=0.0 -> open_pos, grasp=1.0 -> close_pos.
    # Matches XArmConfig defaults in tactile-data-collection/environment/xarm_controller.py
    # so the safety threshold (calibrated against that gripper speed) holds.
    GRIPPER_OPEN_POS = 850
    GRIPPER_CLOSE_POS = 0
    GRIPPER_SPEED = 1000           # set once at init; same speed used during teleop collection
    GRIPPER_EPS = 0.01             # min |Δgrasp| in [0,1] before re-issuing a gripper command

    def __init__(self, cfg: XArmInferenceConfig, tactile: TactileSensors) -> None:
        from xarm.wrapper import XArmAPI  # local import; only needed when not in dry_run

        self.cfg = cfg
        self.tactile = tactile           # required, never None on the real-robot path
        self.previous_grasp = 0.0        # last grasp we commanded, in [0,1]; 0 = open
        self._safety_active = False     # one-shot warning latch so the log isn't spammed

        self.arm = XArmAPI(cfg.xarm_ip)
        self.arm.connect()
        self.arm.clean_error()
        self.arm.clean_warn()
        code = self.arm.motion_enable(enable=True)
        if code != 0:
            raise RuntimeError(f"motion_enable failed: code {code}")
        # Servo mode for cartesian streaming
        self.arm.set_mode(1)
        self.arm.set_state(0)
        self.arm.set_collision_sensitivity(3)
        # Gripper init — matches tactile-data-collection so continuous position
        # commands (set_gripper_position with target in [0, 850]) actually track.
        # Without set_gripper_enable(True) some firmware versions silently ignore
        # set_gripper_position. Speed is set once and persists across calls.
        code = self.arm.set_gripper_mode(0)
        if code != 0:
            raise RuntimeError(f"set_gripper_mode failed: code {code}")
        code = self.arm.set_gripper_enable(True)
        if code != 0:
            raise RuntimeError(f"set_gripper_enable failed: code {code}")
        code = self.arm.set_gripper_speed(self.GRIPPER_SPEED)
        if code != 0:
            raise RuntimeError(f"set_gripper_speed failed: code {code}")
        log.info("xArm connected at %s (tactile safety wired)", cfg.xarm_ip)

    def home(self) -> None:
        """Joint-space homing — same routine as ril_env.xarm_controller.XArm.home():
        position-mode → open gripper → set_servo_angle(home_joint_angles) → servo-mode.

        Also resets previous_grasp = 0.0 and clears the safety latch, mirroring
        XArm.home() in tactile-data-collection — without this, the gripper-eps
        gate in _apply_grasp would silently swallow the first command after a
        home since previous_grasp would still hold its last pre-home value.
        """
        log.info("Homing via set_servo_angle(angle=%s, speed=%g)",
                 list(self.cfg.home_joint_angles_deg), self.cfg.home_speed)
        # Position mode for the discrete homing motion.
        self.arm.set_mode(0)
        self.arm.set_state(0)
        # Open the gripper first so the first observed grasp state is OPEN (matches training start).
        self.arm.set_gripper_position(self.GRIPPER_OPEN_POS, wait=True)
        code = self.arm.set_servo_angle(
            angle=list(self.cfg.home_joint_angles_deg),
            speed=self.cfg.home_speed,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(f"set_servo_angle(home) failed: code {code}")
        # Back to streaming servo mode for the rollout.
        self.arm.set_mode(1)
        self.arm.set_state(0)
        # Gripper is now open after homing; sync our cached state.
        self.previous_grasp = 0.0
        self._safety_active = False
        # Log the resulting TCP pose so the user can sanity-check against the
        # reference (~ 475.79, -1.14, 244.72, 179.13, -0.01, 0.78).
        code, pose = self.arm.get_position()
        if code == 0:
            log.info("Reached home. TCP pose = (%.2f mm, %.2f mm, %.2f mm,  %.2f°, %.2f°, %.2f°)", *pose[:6])
        else:
            log.warning("Homed but get_position returned code %d", code)

    def get_state_8dim(self) -> np.ndarray:
        """Return (8,) state in the model's input convention:
            [ee_x_m, ee_y_m, ee_z_m, ax_rad, ay_rad, az_rad, grasp, grasp]

        Matches what `examples/xarm/convert_zarr_to_lerobot.py` wrote into the dataset.
        """
        code, pose = self.arm.get_position()
        if code != 0:
            raise RuntimeError(f"get_position failed: code {code}")
        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = pose[:6]
        xyz_m = np.array([x_mm, y_mm, z_mm], dtype=np.float32) / 1000.0
        axis_angle = Rot.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_rotvec().astype(np.float32)
        code, grip_pos = self.arm.get_gripper_position()
        if code != 0:
            log.warning("get_gripper_position failed: code %d; assuming open.", code)
            grip_pos = 850
        # phone convention: 0 = open, 1 = closed.  xArm: 850 = open, 0 = closed.
        grasp = 1.0 - float(grip_pos) / 850.0
        grasp = float(np.clip(grasp, 0.0, 1.0))
        return np.concatenate([xyz_m, axis_angle, [grasp, grasp]]).astype(np.float32)

    def execute_delta(self, action: np.ndarray) -> None:
        """Apply one 7-dim delta action: [dxyz_m, daxis_angle_rad, grasp_pm1].

        We compose the delta onto the *current measured* pose (not the previous target),
        which matches what the model was trained to predict (`tcp[t+1] - tcp[t]` in
        measured-EE-pose space; see `examples/xarm/convert_zarr_to_lerobot.py`).

        Gripper command is routed through the tactile safety wrapper: if the metric
        is over threshold (or readings are stale) and the model is asking to close
        further than `previous_grasp`, the grasp is clamped to `previous_grasp`.
        Opening is always allowed.
        """
        cur_state = self.get_state_8dim()
        cur_xyz_m = cur_state[:3]
        cur_aa = cur_state[3:6]

        # Cartesian delta — apply scale + per-step magnitude clip on the 6D pose component only.
        delta = action[:6].astype(np.float64) * self.cfg.action_scale
        norm = np.linalg.norm(delta)
        if norm > self.cfg.max_action_norm:
            delta *= self.cfg.max_action_norm / norm
            log.warning("Action norm %.3f exceeded max %.3f; clipped.", norm, self.cfg.max_action_norm)

        target_xyz_m = cur_xyz_m + delta[:3]
        # Compose rotations in axis-angle: target_R = exp(delta_aa) * exp(cur_aa).
        target_rot = Rot.from_rotvec(delta[3:6]) * Rot.from_rotvec(cur_aa)
        # xArm SDK takes Euler degrees in xyz convention.
        target_euler_deg = target_rot.as_euler("xyz", degrees=True)
        target_pose_mm_deg = [
            float(target_xyz_m[0] * 1000.0),
            float(target_xyz_m[1] * 1000.0),
            float(target_xyz_m[2] * 1000.0),
            float(target_euler_deg[0]),
            float(target_euler_deg[1]),
            float(target_euler_deg[2]),
        ]
        code = self.arm.set_servo_cartesian(target_pose_mm_deg, speed=100, mvacc=2000)
        if code != 0:
            log.warning("set_servo_cartesian returned code %d (target=%s)", code, target_pose_mm_deg)

        # Gripper. The model emits action[6] in {-1=open, +1=closed} (training convention from
        # `examples/xarm/convert_zarr_to_lerobot.py`: grasp_pm1 = 2*phone_raw - 1 with
        # phone_raw ∈ {0=open, 1=closed}). Invert that back to the [0,1] grasp axis used
        # by tactile-data-collection's XArm.step_abs, then route through:
        #   - _apply_tactile_safety:  clamps to previous_grasp when closing into contact
        #   - _apply_grasp_continuous: eps-gated set_gripper_position with continuous target
        grasp = float(np.clip((action[6] + 1.0) * 0.5, 0.0, 1.0))
        grasp = self._apply_tactile_safety(grasp)
        self.previous_grasp = self._apply_grasp_continuous(grasp)

    def _apply_tactile_safety(self, grasp: float) -> float:
        """Clamp a CLOSING command to previous_grasp when tactile says unsafe.

        Mirrors XArm._apply_tactile_safety in tactile-data-collection. Stale or
        zero-connected-taxel readings count as unsafe (fail-safe). Opening
        (grasp <= previous_grasp) is always allowed regardless of contact.
        """
        try:
            metric_val, is_safe = self.tactile.safety()
        except Exception as e:
            log.error("[tactile] safety read failed: %s — treating as unsafe", e)
            metric_val, is_safe = float("nan"), False

        closing = grasp > self.previous_grasp
        if not is_safe and closing:
            if not self._safety_active:
                log.warning(
                    "[tactile] safety engaged (metric=%.2f, threshold=%.2f); "
                    "holding grasp at %.3f", metric_val,
                    self.tactile.config.safety_threshold, self.previous_grasp,
                )
            self._safety_active = True
            return self.previous_grasp
        if self._safety_active and is_safe:
            log.info("[tactile] safety released (metric=%.2f)", metric_val)
        self._safety_active = False
        return grasp

    def _apply_grasp_continuous(self, grasp: float) -> float:
        """Map grasp in [0,1] to an xArm gripper position and command it.

        Eps-gated so tiny numerical drift doesn't spam the gripper SDK every tick.
        Returns the new previous_grasp value (unchanged if below epsilon).
        """
        grasp = float(np.clip(grasp, 0.0, 1.0))
        if abs(grasp - self.previous_grasp) < self.GRIPPER_EPS:
            return self.previous_grasp
        target = int(round(
            self.GRIPPER_OPEN_POS + grasp * (self.GRIPPER_CLOSE_POS - self.GRIPPER_OPEN_POS)
        ))
        code = self.arm.set_gripper_position(target, wait=False)
        if code != 0:
            log.warning("set_gripper_position(%d) returned code %d", target, code)
            return self.previous_grasp
        return grasp

    def emergency_stop(self) -> None:
        log.warning("EMERGENCY STOP")
        try:
            self.arm.set_state(4)
            time.sleep(0.1)
            self.arm.set_state(0)
        except Exception as e:
            log.error("emergency_stop secondary failure: %s", e)

    def close(self) -> None:
        try:
            self.home()
        finally:
            self.arm.disconnect()
            log.info("xArm disconnected.")


# ───────── video recorder ─────────


class VideoRecorder:
    def __init__(self, path: Path, fps: int, frame_size: Tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        self.path = path
        log.info("Recording rollout to %s", path)

    def write(self, *images_bgr: np.ndarray) -> None:
        # Inputs are already BGR (matches training-time channel order);
        # cv2.VideoWriter also expects BGR, so no conversion needed.
        # Pass 2 images for the overlay path (agent | wrist) or 3 for the
        # manifeel path (agent | wrist | manifeel third_image).
        frame = np.concatenate(images_bgr, axis=1)
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
        log.info("Saved %s", self.path)


# ───────── tactile init + baseline ─────────


def _build_tactile(cfg: XArmInferenceConfig) -> TactileSensors:
    """Open both A31301 boards and wait until they're streaming live data.

    Hard-fails on any of: port-open failure, no frames within
    tactile_init_timeout_sec, or zero connected taxels on either board. The
    intent is to never start the rollout with a dead safety wrapper — matches
    the user spec that tactile is mandatory.
    """
    tcfg = TactileConfig(
        ports=[cfg.left_port, cfg.right_port],
        baud=cfg.tactile_baud,
        safety_metric=cfg.safety_metric,
        safety_threshold=cfg.safety_threshold,
        stale_after_sec=cfg.stale_after_sec,
    )
    log.info("Opening tactile sensors: L=%s, R=%s @ %d baud (metric=%s, thresh=%.1f, delta-from-idle)",
             cfg.left_port, cfg.right_port, cfg.tactile_baud,
             cfg.safety_metric, cfg.safety_threshold)
    tactile = TactileSensors(tcfg, names=["L", "R"])
    tactile.__enter__()
    if not tactile.all_open:
        tactile.__exit__(None, None, None)
        raise SystemExit(
            f"[tactile] required ports failed to open: {tactile.failed_ports}. "
            f"Check `ls /dev/ttyACM*`, board power, and udev permissions "
            f"(dialout group). Run with the boards connected — this script "
            f"has no --no-tactile escape hatch by design."
        )
    if not tactile.wait_until_fresh(timeout_sec=cfg.tactile_init_timeout_sec):
        tactile.__exit__(None, None, None)
        raise SystemExit(
            f"[tactile] ports opened but neither board published a fresh frame "
            f"within {cfg.tactile_init_timeout_sec:.1f}s. Likely the ESP32 isn't "
            f"streaming — try power-cycling the boards or check the firmware."
        )
    log.info("Tactile sensors streaming (both boards fresh).")
    return tactile


def _capture_tactile_baseline(tactile: TactileSensors, duration_sec: float) -> np.ndarray:
    """Average per-cell xyz over `duration_sec` while the gripper is open + no contact.

    Caller must guarantee the open + no-contact condition (i.e. call this
    immediately after homing). Returns (n_sensors, n_taxels, 3) float32.
    Raises if no usable samples arrive — better to fail loudly than to run
    with a zero baseline (which would put the safety threshold into raw-value
    units, where 1500 trips constantly).
    """
    log.info("Sampling tactile baseline (%.1f s, keep fingers untouched)...", duration_sec)
    samples = []
    t_end = time.time() + duration_sec
    while time.time() < t_end:
        states = tactile.get_latest()
        if all(s.get("host_timestamp", 0.0) > 0 for s in states):
            xyz = np.stack(
                [np.asarray(s["xyz"], dtype=np.float32) for s in states], axis=0
            )
            samples.append(xyz)
        time.sleep(0.05)
    if not samples:
        raise SystemExit(
            "[tactile] no usable baseline samples — boards opened but stopped "
            "streaming during baseline capture."
        )
    baseline = np.mean(np.stack(samples), axis=0).astype(np.float32)
    # Report sum_abs_z at idle so the user can sanity-check it's well under threshold.
    sum_abs_z = float(np.sum(np.abs(baseline[..., 2])))
    log.info("Baseline captured from %d frames. Idle sum|Bz| (raw) = %.1f; "
             "delta-from-idle will be measured against threshold %.1f.",
             len(samples), sum_abs_z, tactile.config.safety_threshold)
    return baseline


# ───────── main loop ─────────


def run(cfg: XArmInferenceConfig) -> None:
    log.info("=" * 70)
    log.info("checkpoint: %s", cfg.checkpoint)
    log.info("prompt:     %s", cfg.prompt)
    log.info("xarm:       %s   control %d Hz   replan every %d steps",
             cfg.xarm_ip if not cfg.dry_run else "(DRY RUN — robot not connected)",
             cfg.control_hz, cfg.replan_steps)
    log.info("=" * 70)

    # Tactile is mandatory — open and validate BEFORE anything else so that a
    # missing sensor fails fast (before we've loaded the model or moved the arm).
    tactile = _build_tactile(cfg)
    # Track resources allocated *after* tactile so the outer finally can release
    # them in the right order (cams + robot first so safety reads stay live while
    # the robot is being homed during close, then tactile last).
    cams: CameraManager | None = None
    robot: XArmController | None = None
    video: VideoRecorder | None = None
    viz: LiveViz | None = None
    try:
        log.info("Loading policy via openpi…")
        train_cfg = _config.get_config(cfg.train_config_name)
        policy = policy_config.create_trained_policy(
            train_cfg, str(cfg.checkpoint), default_prompt=cfg.prompt
        )
        # Reseed the flow-matching sampler. Without this, Policy.__init__
        # defaults to jax.random.key(0), so every fresh launch reuses the
        # same noise sequence — two runs against identical observations
        # would produce identical action chunks. Derive a 32-bit seed from
        # the wall clock unless the user pinned --seed for reproducibility.
        if cfg.seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
            log.info("Sampler seed (random): %d  (pin via --seed to reproduce)", seed)
        else:
            seed = int(cfg.seed) & 0xFFFFFFFF
            log.info("Sampler seed (user-pinned): %d", seed)
        import jax  # local import — openpi already pulled it in via create_trained_policy
        policy._rng = jax.random.key(seed)
        log.info("Policy loaded.")

        cams = CameraManager(cfg)
        robot = None if cfg.dry_run else XArmController(cfg, tactile=tactile)
        # 3-up video for manifeel mode (agent | wrist | manifeel third_image),
        # otherwise the standard 2-up (agent | wrist).
        n_video_panels = 3 if cfg.manifeel else 2
        video = (
            VideoRecorder(cfg.video_path, cfg.control_hz, (cfg.image_size * n_video_panels, cfg.image_size))
            if cfg.save_video else None
        )

        if robot is not None:
            robot.home()
        # Capture baseline AFTER homing: at this point the gripper is open and the
        # fingers should be untouched, so the per-cell xyz is the static field.
        # Install on tactile.config so the safety wrapper measures delta-from-idle
        # against cfg.safety_threshold (which is in delta units; see tactile_safety.py).
        baseline = _capture_tactile_baseline(tactile, cfg.baseline_duration_sec)
        tactile.config.baseline = baseline

        # Policy-input overlay. Built only if --overlay-stats was passed; lives
        # in inference_overlay.InferenceOverlay. Mirrors the training-time
        # render_overlays.py pipeline (Hampel clip + baseline subtract +
        # normalize + deadband + draw). The same `baseline` we just captured
        # serves as the per-rollout offset (= per-episode offset at training).
        inference_overlay = None
        manifeel_renderer = None
        if cfg.manifeel:
            # Manifeel inference path: do NOT modify camera frames. Render a
            # separate tactile-only image (sensordrawing "third_image") per
            # tick and pack it as observation/tactile_image so the model's
            # third image slot is fed with the same content it saw at
            # training time (right_wrist_0_rgb / image_mask=True).
            from inference_overlay import ManifeelRenderer  # local sibling file
            manifeel_renderer = ManifeelRenderer(
                stats_path=cfg.manifeel_stats_path,
                phone_data_collection_repo=cfg.viz_overlay_repo,
                baseline=baseline,
                out_size=cfg.image_size,
            )
            log.info(
                "ManifeelRenderer enabled: scale_xy=[%.1f, %.1f], "
                "scale_z=[%.1f, %.1f], deadband=%.4f, out_size=%d. "
                "Camera frames will be sent RAW; tactile flows in as the "
                "third image slot (observation/tactile_image).",
                float(manifeel_renderer.scale_xy[0]), float(manifeel_renderer.scale_xy[1]),
                float(manifeel_renderer.scale_z[0]), float(manifeel_renderer.scale_z[1]),
                manifeel_renderer.deadband, cfg.image_size,
            )
        elif cfg.overlay_stats_path:
            from inference_overlay import InferenceOverlay  # local sibling file
            inference_overlay = InferenceOverlay(
                stats_path=cfg.overlay_stats_path,
                phone_data_collection_repo=cfg.viz_overlay_repo,
                baseline=baseline,
                mode_key_override=cfg.overlay_mode_key,
            )
            log.info(
                "InferenceOverlay enabled: mode=%s, arrow_length_scale=%.4f, "
                "scale_xy=[%.1f, %.1f], scale_z=[%.1f, %.1f], deadband=%.4f",
                inference_overlay.mode_key, inference_overlay.arrow_length_scale,
                float(inference_overlay.scale_xy[0]), float(inference_overlay.scale_xy[1]),
                float(inference_overlay.scale_z[0]), float(inference_overlay.scale_z[1]),
                inference_overlay.deadband,
            )
        else:
            log.warning(
                "No --overlay-stats / --overlay / --manifeel provided. Camera "
                "frames sent to the policy will be RAW. Only correct if the "
                "checkpoint was trained on raw (no-overlay) frames; otherwise "
                "expect a distribution shift."
            )

        if cfg.show_windows:
            viz = LiveViz(cfg, baseline)
        if robot is not None and not cfg.auto_start:
            input("Press ENTER to begin rollout (Ctrl+C to abort)…")

        # Install signal handler so Ctrl+C triggers a clean estop.
        def _sigint(*_):
            log.warning("SIGINT received")
            if robot is not None:
                robot.emergency_stop()
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, _sigint)

        action_plan: collections.deque = collections.deque()
        dt = 1.0 / cfg.control_hz

        try:
            for step in range(cfg.max_steps):
                t0 = time.time()
                # BGR uint8 throughout — see CameraManager + InferenceOverlay docstrings.
                agent_bgr, wrist_bgr = cams.get_observation()

                # If overlay or viz needs arm state + tactile, fetch them
                # ONCE per tick. (Overlay needs them to draw; viz needs them
                # for its own draw.)
                need_arm_tactile = (inference_overlay is not None) or (viz is not None)
                angles_deg = None
                grip_raw = None
                raw_L = raw_R = None
                if need_arm_tactile:
                    if robot is not None:
                        code, angles_deg = robot.arm.get_servo_angle()
                        if code != 0:
                            angles_deg = list(cfg.home_joint_angles_deg)
                        code2, grip_raw = robot.arm.get_gripper_position()
                        if code2 != 0:
                            grip_raw = 850
                    else:
                        angles_deg = list(cfg.home_joint_angles_deg)
                        grip_raw = 850
                    tac_states = tactile.get_latest()
                    if len(tac_states) >= 1 and "xyz" in tac_states[0]:
                        raw_L = np.asarray(tac_states[0]["xyz"], dtype=np.float32)
                    if len(tac_states) >= 2 and "xyz" in tac_states[1]:
                        raw_R = np.asarray(tac_states[1]["xyz"], dtype=np.float32)

                # Draw the SAME overlay the training data was rendered with
                # onto the policy-input frames. This is what eliminates the
                # raw-vs-overlay distribution shift between training and
                # inference. Fails open (returns input unchanged) if tactile
                # data isn't available — never crashes the rollout over it.
                if inference_overlay is not None:
                    agent_bgr = inference_overlay.apply(
                        agent_bgr, "side", angles_deg, float(grip_raw),
                        raw_L, raw_R,
                    )
                    wrist_bgr = inference_overlay.apply(
                        wrist_bgr, "wrist", angles_deg, float(grip_raw),
                        raw_L, raw_R,
                    )

                # Manifeel: agent/wrist stay raw; render the synthetic
                # tactile-only image to feed as the third image slot.
                manifeel_bgr = None
                if manifeel_renderer is not None:
                    manifeel_bgr = manifeel_renderer.render(raw_L, raw_R)

                if not action_plan:
                    if robot is not None:
                        state = robot.get_state_8dim()
                    else:
                        # Dry-run: synthesize a plausible state at the documented home TCP pose
                        # (the values observed after homing on the lab xArm 7 — see
                        # `home_joint_angles_deg` docstring). Gripper assumed open.
                        home_tcp_mm_deg = (475.79, -1.14, 244.72, 179.13, -0.01, 0.78)
                        home_xyz = np.array(home_tcp_mm_deg[:3], dtype=np.float32) / 1000.0
                        home_aa = Rot.from_euler("xyz", home_tcp_mm_deg[3:], degrees=True).as_rotvec().astype(np.float32)
                        state = np.concatenate([home_xyz, home_aa, [0.0, 0.0]]).astype(np.float32)
                    # The keys say "image" / "wrist_image" but the bytes are
                    # BGR-ordered — that exactly matches what the training
                    # pipeline stored (BGR captured + saved by PIL as
                    # RGB-labeled PNGs, then loaded back as RGB tensors with
                    # the same bytes). See InferenceOverlay class docstring.
                    obs = {
                        "observation/image": agent_bgr,
                        "observation/wrist_image": wrist_bgr,
                        "observation/state": state,
                        "prompt": cfg.prompt,
                    }
                    if manifeel_bgr is not None:
                        # Third image slot. LiberoManifeelInputs reads this
                        # and packs it into right_wrist_0_rgb with mask=True.
                        obs["observation/tactile_image"] = manifeel_bgr
                    chunk = np.asarray(policy.infer(obs)["actions"])  # (10, 7)
                    if chunk.shape[0] < cfg.replan_steps:
                        raise RuntimeError(f"Action chunk too short: {chunk.shape}")
                    action_plan.extend(chunk[: cfg.replan_steps])

                action = action_plan.popleft()
                log.info("step=%3d  dxyz=[%+0.4f %+0.4f %+0.4f] m  daa=[%+0.4f %+0.4f %+0.4f] rad  grasp=%+0.2f",
                         step, action[0], action[1], action[2], action[3], action[4], action[5], action[6])

                if robot is not None:
                    robot.execute_delta(action)
                if video is not None:
                    if manifeel_bgr is not None:
                        # 3-up: agent | wrist | manifeel third_image —
                        # exactly the three images the model sees.
                        video.write(agent_bgr, wrist_bgr, manifeel_bgr)
                    else:
                        video.write(agent_bgr, wrist_bgr)
                if viz is not None:
                    # Arm state + tactile already fetched at the top of this
                    # tick (need_arm_tactile is True whenever viz is on). Note:
                    # if InferenceOverlay is also on, agent_bgr/wrist_bgr here
                    # already have the policy-input overlay drawn — viz will
                    # show the same image the policy is seeing. That's the
                    # intended behavior (visualizes the actual policy input).
                    viz.update(agent_bgr, wrist_bgr, angles_deg, float(grip_raw), raw_L, raw_R)

                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            log.info("Reached max_steps=%d", cfg.max_steps)

        except KeyboardInterrupt:
            log.warning("Aborted by user.")
        except Exception as e:
            log.error("Rollout failed: %s", e, exc_info=True)
            if robot is not None:
                robot.emergency_stop()
    finally:
        if viz is not None:
            viz.close()
        if video is not None:
            video.close()
        if cams is not None:
            cams.close()
        if robot is not None:
            robot.close()
        # Stop the tactile reader threads + release serial ports LAST so we still
        # have safety reads available while the robot is being homed in robot.close().
        tactile.__exit__(None, None, None)


# ───────── CLI ─────────


def parse_args() -> XArmInferenceConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to the trained step-N checkpoint dir, e.g. "
                        ".../pi05_xarm_finetune_lora/droid_init_20k/19999")
    p.add_argument("--train-config-name", type=str, default="pi05_xarm_finetune_lora")
    p.add_argument("--xarm-ip", type=str, default="192.168.1.223")
    p.add_argument("--control-hz", type=int, default=10)
    p.add_argument("--agent-cam-id", type=str, default="327122079374",
                   help="RealSense: 12-digit serial OR small index (0/1) into the enumerated "
                        "device list. Default is the lab rig's agent (third-person) D-series "
                        "serial. OpenCV (--no-realsense): USB index.")
    p.add_argument("--wrist-cam-id", type=str, default="332322072612",
                   help="RealSense: serial or index. Default is the lab rig's wrist-mounted "
                        "D-series serial. OpenCV (--no-realsense): USB index.")
    rs_group = p.add_mutually_exclusive_group()
    rs_group.add_argument("--use-realsense", dest="use_realsense", action="store_true", default=True,
                          help="Use pyrealsense2 (default). Cameras auto-discovered if cam-ids unset.")
    rs_group.add_argument("--no-realsense", dest="use_realsense", action="store_false",
                          help="Use plain cv2.VideoCapture instead of pyrealsense2.")
    p.add_argument("--prompt", type=str, default="pick up the red block")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--max-action-norm", type=float, default=0.05)
    p.add_argument("--auto-start", action="store_true",
                   help="Skip the manual ENTER prompt before the first action.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run the full inference loop but don't connect to / move the robot. "
                        "Tactile sensors are STILL required (script hard-fails without them).")
    p.add_argument("--no-video", dest="save_video", action="store_false", default=True)
    p.add_argument("--video-path", type=Path, default=None)

    # Tactile safety — see XArmInferenceConfig for the rationale on these defaults.
    p.add_argument("--left-port", type=str, default="/dev/ttyACM0",
                   help="Serial port for the LEFT-finger ESP32 (default /dev/ttyACM0).")
    p.add_argument("--right-port", type=str, default="/dev/ttyACM1",
                   help="Serial port for the RIGHT-finger ESP32 (default /dev/ttyACM1).")
    p.add_argument("--tactile-baud", type=int, default=115200)
    p.add_argument("--safety-metric", type=str, default="sum_abs_z",
                   choices=["sum_abs_z", "max_abs_z", "max_norm"],
                   help="Reduction over per-taxel xyz used to detect contact. "
                        "sum_abs_z is what tactile-data-collection settled on; "
                        "max_abs_z does NOT work for this sensor mounting.")
    p.add_argument("--safety-threshold", type=float, default=1500.0,
                   help="Delta-from-idle metric above this -> hold grasp (no further closing). "
                        "Default 1500 matches tactile_config.SAFETY_THRESHOLD; raise to allow "
                        "harder grip before clamping.")
    p.add_argument("--stale-after-sec", type=float, default=0.2,
                   help="Tactile readings older than this count as unsafe (fail-safe).")
    p.add_argument("--baseline-duration-sec", type=float, default=1.5,
                   help="How long to average per-cell xyz at rest after homing.")
    p.add_argument("--tactile-init-timeout-sec", type=float, default=5.0,
                   help="Hard-fail if both boards aren't streaming a fresh frame "
                        "within this window after open.")

    # Live operator viz (optional).
    p.add_argument("--show-windows", action="store_true",
                   help="Pop a cv2 window showing agent | wrist with tactile arrows "
                        "drawn each tick. Operator debug only; model input is unchanged.")
    p.add_argument("--viz-overlay-repo", type=str,
                   default="/home/u-ril/edward/phone_data_collection",
                   help="Path to phone_data_collection (provides environment.tactile_overlay).")
    p.add_argument("--viz-mode-key", type=str, default="points1_arrow",
                   help="Which renderer variant to draw (closest live equivalent to the "
                        "legacy 'arrow' overlay is 'points1_arrow').")

    # Policy-input tactile overlay — eliminates the raw-vs-overlay distribution
    # shift between training and inference.
    p.add_argument("--overlay", type=str, default=None, metavar="MODE",
                   help="Shortcut: use the bundled pooled overlay npz for "
                        "this mode (e.g. 'points9_arrow' or 'points1_arrow'). "
                        "Resolves to examples/xarm/inference/overlay_norm_<MODE>.npz "
                        "and uses MODE as the renderer mode_key. The bundled "
                        "files pool stats across all 4 task zarrs and work "
                        "for any task — no per-task file needed. "
                        "Mutually exclusive with --overlay-stats.")
    p.add_argument("--overlay-stats", type=str, default=None,
                   help="Power-user override: explicit path to any "
                        "overlay_norm.npz (from extract_overlay_norm.py). "
                        "Use this if you want a per-task variant instead of "
                        "the bundled pooled file. Prefer --overlay for the "
                        "common case. When unset (and --overlay also unset), "
                        "raw frames are sent (only correct for "
                        "*_baseline_lora checkpoints).")
    p.add_argument("--overlay-mode-key", type=str, default=None,
                   help="Override the mode_key stored inside the --overlay-stats "
                        "npz. Ignored when --overlay is used (the mode is "
                        "taken from the --overlay value). Leave unset to use "
                        "the value baked into the npz (matches training).")
    p.add_argument("--manifeel", action="store_true",
                   help="Enable manifeel inference path for pi05_xarm_<task>_"
                        "manifeel_baseline_lora checkpoints. Agent + wrist "
                        "camera frames are left RAW (no overlay drawn on them); "
                        "instead, a synthetic tactile-only image is rendered "
                        "from live tactile readings (sensordrawing 'third_image' "
                        "mode) and packed as observation/tactile_image for "
                        "the third image slot (right_wrist_0_rgb with mask=True). "
                        "Video recording switches to 3-up (agent | wrist | "
                        "manifeel). Mutually exclusive with --overlay / "
                        "--overlay-stats.")
    p.add_argument("--manifeel-stats", type=str,
                   default="examples/xarm/inference/overlay_norm_manifeel.npz",
                   help="Path to the manifeel overlay-norm npz (used only for "
                        "scale_xy/scale_z/deadband; the rendering mode is "
                        "hardcoded to sensordrawing's 'third_image'). Default "
                        "is the bundled pooled npz.")

    p.add_argument("--seed", type=int, default=None,
                   help="Flow-matching sampler seed. Default (unset) draws a "
                        "fresh seed from os.urandom on every launch, so each "
                        "rollout sees a different noise sequence. Pin an int "
                        "to reproduce a specific rollout (logged at startup).")

    args = p.parse_args()

    # Resolve --overlay shortcut into (overlay_stats_path, overlay_mode_key).
    overlay_stats_path = args.overlay_stats
    overlay_mode_key = args.overlay_mode_key
    if args.overlay is not None:
        if args.overlay_stats is not None:
            raise SystemExit(
                "Pass either --overlay <MODE> or --overlay-stats <PATH>, "
                "not both. --overlay is the easy path (bundled pooled "
                "npz); --overlay-stats is the explicit-path override."
            )
        script_dir = Path(__file__).resolve().parent
        inferred_path = script_dir / f"overlay_norm_{args.overlay}.npz"
        if not inferred_path.is_file():
            raise SystemExit(
                f"--overlay {args.overlay!r} expected the bundled npz at "
                f"{inferred_path}, but it does not exist. Either run "
                f"extract_overlay_norm.py to produce it, or pass an "
                f"explicit --overlay-stats <path>."
            )
        overlay_stats_path = str(inferred_path)
        # Use the requested mode as the renderer mode_key (overrides
        # whatever is baked into the npz — usually identical anyway since
        # the pooled file's mode_key matches the --overlay value).
        overlay_mode_key = args.overlay

    # Manifeel mode: disables the overlay-on-camera path. The two flags are
    # mutually exclusive — manifeel sends a SEPARATE tactile image as obs
    # rather than overlaying onto the existing camera frames.
    if args.manifeel:
        if args.overlay is not None or args.overlay_stats is not None:
            raise SystemExit(
                "--manifeel is mutually exclusive with --overlay / "
                "--overlay-stats. The manifeel path keeps camera frames RAW "
                "and adds a separate tactile image to the obs dict."
            )
        if not Path(args.manifeel_stats).is_file():
            raise SystemExit(
                f"--manifeel-stats {args.manifeel_stats!r} not found. The "
                f"default bundled path is examples/xarm/inference/"
                f"overlay_norm_manifeel.npz."
            )

    return XArmInferenceConfig(
        checkpoint=args.checkpoint,
        train_config_name=args.train_config_name,
        xarm_ip=args.xarm_ip,
        control_hz=args.control_hz,
        agent_cam_id=args.agent_cam_id,
        wrist_cam_id=args.wrist_cam_id,
        use_realsense=args.use_realsense,
        prompt=args.prompt,
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        action_scale=args.action_scale,
        max_action_norm=args.max_action_norm,
        auto_start=args.auto_start,
        dry_run=args.dry_run,
        save_video=args.save_video,
        video_path=args.video_path,
        left_port=args.left_port,
        right_port=args.right_port,
        tactile_baud=args.tactile_baud,
        safety_metric=args.safety_metric,
        safety_threshold=args.safety_threshold,
        stale_after_sec=args.stale_after_sec,
        baseline_duration_sec=args.baseline_duration_sec,
        tactile_init_timeout_sec=args.tactile_init_timeout_sec,
        show_windows=args.show_windows,
        viz_overlay_repo=args.viz_overlay_repo,
        viz_mode_key=args.viz_mode_key,
        overlay_stats_path=overlay_stats_path,
        overlay_mode_key=overlay_mode_key,
        manifeel=args.manifeel,
        manifeel_stats_path=args.manifeel_stats if args.manifeel else None,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(run(parse_args()) or 0)
