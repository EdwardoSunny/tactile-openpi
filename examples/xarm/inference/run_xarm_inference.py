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

The tactile arrow overlay (drawn on training-time `img_0`/`img_1`) is left as a stub
hook in `apply_tactile_overlay`. The teleop_data.zarr we trained on had `n_contacts`
all-zero, so the trained model never saw arrows — running with the hook as identity is
correct for THIS checkpoint. If you later retrain on data with real tactile contacts,
fill in the hook with the same renderer the collection pipeline used.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# openpi imports (must be importable; the .venv created by `uv sync` has them).
from openpi.policies import policy_config
from openpi.training import config as _config


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
    agent_cam_id: int | str | None = None
    wrist_cam_id: int | str | None = None
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

    # Video
    save_video: bool = True
    video_path: Path | None = None

    def __post_init__(self) -> None:
        if self.save_video and self.video_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.video_path = Path(f"rollouts/xarm_inference_{stamp}.mp4")


# ───────── tactile overlay hook ─────────


def apply_tactile_overlay(img: np.ndarray, tactile_reading: np.ndarray | None) -> np.ndarray:
    """Stub for re-rendering the tactile arrow overlay onto a live frame.

    For the current checkpoint (trained on teleop_data.zarr where n_contacts was all-zero),
    no arrows were drawn at training time, so this can be identity. If you retrain with
    real contact data, fill this in with the same renderer used at collection — otherwise
    the deployment-time pixel distribution differs from training and the policy will fall
    out of distribution.
    """
    return img


# ───────── cameras ─────────


class ThreadedCamera:
    """Mirrors collect_xarm_demos.py's ThreadedCamera. cv2.VideoCapture in a background
    thread that always exposes the latest frame as RGB uint8, already resized to
    (image_size, image_size). Fails fast if the device isn't producing frames."""

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
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(
                frame_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
            self._latest = frame_rgb

    def read(self) -> np.ndarray:
        assert self._latest is not None
        return self._latest.copy()

    def close(self) -> None:
        self._stop.set()
        self._th.join(timeout=1.0)
        self.cap.release()


class CameraManager:
    """Returns (agent, wrist) RGB uint8 frames at `image_size`×`image_size`.

    Defaults to RealSense (pyrealsense2). Set `use_realsense=False` to fall back
    to plain OpenCV USB capture (cv2.VideoCapture with integer USB indices),
    which is what collect_xarm_demos.py uses.
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
            agent = cv2.resize(cv2.cvtColor(agent_bgr, cv2.COLOR_BGR2RGB),
                               (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
            wrist = cv2.resize(cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB),
                               (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
        else:
            agent = self.agent_cam.read()  # already RGB uint8 at image_size
            wrist = self.wrist_cam.read()
        agent = apply_tactile_overlay(agent, None)
        wrist = apply_tactile_overlay(wrist, None)
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
    def __init__(self, cfg: XArmInferenceConfig) -> None:
        from xarm.wrapper import XArmAPI  # local import; only needed when not in dry_run

        self.cfg = cfg
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
        log.info("xArm connected at %s", cfg.xarm_ip)

    def home(self) -> None:
        """Joint-space homing — same routine as ril_env.xarm_controller.XArm.home():
        position-mode → open gripper → set_servo_angle(home_joint_angles) → servo-mode.
        """
        log.info("Homing via set_servo_angle(angle=%s, speed=%g)",
                 list(self.cfg.home_joint_angles_deg), self.cfg.home_speed)
        # Position mode for the discrete homing motion.
        self.arm.set_mode(0)
        self.arm.set_state(0)
        # Open the gripper first so the first observed grasp state is OPEN (matches training start).
        self.arm.set_gripper_position(850, wait=True, speed=5000)
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

        # Gripper: model emits action[6] in {-1=open, +1=closed} (training convention from
        # converter: grasp_pm1 = 2*phone_raw - 1, phone_raw ∈ {0=open, 1=closed}).
        # Threshold to a discrete xArm position to avoid chatter.
        grasp = float(action[6])
        target_grip = 0 if grasp > 0.0 else 850   # closed if model says +, open if -
        self.arm.set_gripper_position(target_grip, wait=False, speed=5000)

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

    def write(self, agent_rgb: np.ndarray, wrist_rgb: np.ndarray) -> None:
        frame = np.concatenate([agent_rgb, wrist_rgb], axis=1)
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()
        log.info("Saved %s", self.path)


# ───────── main loop ─────────


def run(cfg: XArmInferenceConfig) -> None:
    log.info("=" * 70)
    log.info("checkpoint: %s", cfg.checkpoint)
    log.info("prompt:     %s", cfg.prompt)
    log.info("xarm:       %s   control %d Hz   replan every %d steps",
             cfg.xarm_ip if not cfg.dry_run else "(DRY RUN — robot not connected)",
             cfg.control_hz, cfg.replan_steps)
    log.info("=" * 70)

    log.info("Loading policy via openpi…")
    train_cfg = _config.get_config(cfg.train_config_name)
    policy = policy_config.create_trained_policy(train_cfg, str(cfg.checkpoint), default_prompt=cfg.prompt)
    log.info("Policy loaded.")

    cams = CameraManager(cfg)
    robot = None if cfg.dry_run else XArmController(cfg)
    video = VideoRecorder(cfg.video_path, cfg.control_hz, (cfg.image_size * 2, cfg.image_size)) if cfg.save_video else None

    if robot is not None:
        robot.home()
        if not cfg.auto_start:
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
            agent_rgb, wrist_rgb = cams.get_observation()

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
                obs = {
                    "observation/image": agent_rgb,
                    "observation/wrist_image": wrist_rgb,
                    "observation/state": state,
                    "prompt": cfg.prompt,
                }
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
                video.write(agent_rgb, wrist_rgb)

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
        if video is not None:
            video.close()
        cams.close()
        if robot is not None:
            robot.close()


# ───────── CLI ─────────


def parse_args() -> XArmInferenceConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to the trained step-N checkpoint dir, e.g. "
                        ".../pi05_xarm_finetune_lora/droid_init_20k/19999")
    p.add_argument("--train-config-name", type=str, default="pi05_xarm_finetune_lora")
    p.add_argument("--xarm-ip", type=str, default="192.168.1.223")
    p.add_argument("--control-hz", type=int, default=10)
    p.add_argument("--agent-cam-id", type=str, default=None,
                   help="RealSense: 12-digit serial OR small index (0/1) into the enumerated "
                        "device list (default 0 = first device). OpenCV (--no-realsense): "
                        "USB index, default 0.")
    p.add_argument("--wrist-cam-id", type=str, default=None,
                   help="RealSense: serial or index (default 1). OpenCV: USB index (default 2).")
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
                   help="Run the full inference loop but don't connect to / move the robot.")
    p.add_argument("--no-video", dest="save_video", action="store_false", default=True)
    p.add_argument("--video-path", type=Path, default=None)
    args = p.parse_args()

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
    )


if __name__ == "__main__":
    sys.exit(run(parse_args()) or 0)
