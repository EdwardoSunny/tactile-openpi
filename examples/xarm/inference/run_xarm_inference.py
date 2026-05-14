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
import time
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# openpi imports (must be importable; the .venv created by `uv sync` has them).
from openpi.policies import policy_config
from openpi.training import config as _config


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("xarm-infer")


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
    agent_cam_id: int | str = 0
    wrist_cam_id: int | str = 2
    use_realsense: bool = True
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


class CameraManager:
    """Returns (agent, wrist) RGB uint8 frames at `image_size`×`image_size`."""

    def __init__(self, cfg: XArmInferenceConfig) -> None:
        self.cfg = cfg
        if cfg.use_realsense:
            self._init_realsense()
        else:
            self._init_usb()

    def _init_realsense(self) -> None:
        import pyrealsense2 as rs  # local import so the script can be used with USB cams without the dep

        self._rs = rs
        self.agent_pipe = rs.pipeline()
        self.wrist_pipe = rs.pipeline()
        # Match phone_data_collection's RealSense config — 1280×720 @ 30 fps, manual exposure/WB
        # so the live pixel statistics match training. If your collection used different
        # settings, edit here to match.
        for pipe, dev in ((self.agent_pipe, self.cfg.agent_cam_id), (self.wrist_pipe, self.cfg.wrist_cam_id)):
            cfg_rs = rs.config()
            cfg_rs.enable_device(str(dev))
            cfg_rs.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            profile = pipe.start(cfg_rs)
            sensor = profile.get_device().query_sensors()[1]
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, 120)
            sensor.set_option(rs.option.gain, 0)
            sensor.set_option(rs.option.enable_auto_white_balance, 0)
            sensor.set_option(rs.option.white_balance, 5900)
        # Warm-up — first few frames after option change are stale.
        for _ in range(30):
            self.agent_pipe.wait_for_frames()
            self.wrist_pipe.wait_for_frames()
        log.info("RealSense initialized (agent=%s, wrist=%s, 1280x720@30, exp=120, wb=5900K)",
                 self.cfg.agent_cam_id, self.cfg.wrist_cam_id)

    def _init_usb(self) -> None:
        self.agent_cap = cv2.VideoCapture(int(self.cfg.agent_cam_id))
        self.wrist_cap = cv2.VideoCapture(int(self.cfg.wrist_cam_id))
        if not (self.agent_cap.isOpened() and self.wrist_cap.isOpened()):
            raise RuntimeError("Failed to open one or both USB cameras")
        for cap in (self.agent_cap, self.wrist_cap):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(10):
            self.agent_cap.read()
            self.wrist_cap.read()
        log.info("USB cameras initialized (agent=%s, wrist=%s)", self.cfg.agent_cam_id, self.cfg.wrist_cam_id)

    def _read_one(self, which: str) -> np.ndarray:
        if self.cfg.use_realsense:
            pipe = self.agent_pipe if which == "agent" else self.wrist_pipe
            frames = pipe.wait_for_frames()
            return np.asanyarray(frames.get_color_frame().get_data())  # BGR
        cap = self.agent_cap if which == "agent" else self.wrist_cap
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"USB cam read failed ({which})")
        return img  # BGR

    def get_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        agent_bgr = self._read_one("agent")
        wrist_bgr = self._read_one("wrist")
        # Training pipeline: BGR->RGB then resize to (image_size, image_size). The
        # phone collection wrote frames already at 224×224 float32 [0,1]; we feed the
        # policy uint8 (H,W,3) in RGB. LiberoInputs._parse_image handles both, but
        # being explicit avoids any dtype surprises.
        agent = cv2.resize(cv2.cvtColor(agent_bgr, cv2.COLOR_BGR2RGB),
                           (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
        wrist = cv2.resize(cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB),
                           (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
        # Hook for tactile arrow overlay (no-op for the current checkpoint).
        agent = apply_tactile_overlay(agent, None)
        wrist = apply_tactile_overlay(wrist, None)
        return agent, wrist

    def close(self) -> None:
        if self.cfg.use_realsense:
            self.agent_pipe.stop()
            self.wrist_pipe.stop()
        else:
            self.agent_cap.release()
            self.wrist_cap.release()


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
    p.add_argument("--agent-cam-id", type=str, default="0")
    p.add_argument("--wrist-cam-id", type=str, default="2")
    p.add_argument("--use-usb-cameras", action="store_true",
                   help="Use OpenCV USB capture instead of RealSense.")
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
        use_realsense=not args.use_usb_cameras,
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
