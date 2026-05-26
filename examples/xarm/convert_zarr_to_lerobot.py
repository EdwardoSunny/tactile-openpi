"""Convert phone-teleop xArm zarr dataset to a LeRobot dataset for openpi fine-tuning.

Input zarr schema — the OVERLAY-augmented zarr produced by
``scripts/render_overlays.py`` in the tactile-data-collection repo:

    /data
        state                (N, 7)  float32  [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, grasp{0,1}]
        img_0                (N, 224, 224, 3) float32 [0,1]  RAW (no overlay) agentview
        img_0_arrow / _grid / _point / _bar                  same agentview with the named overlay drawn
        img_1, img_1_*       same for the wrist-mounted camera
        n_contacts, tactile, tactile_connected, tactile_ts_ms, tactile_lag_ms
                                                             raw force readings; not consumed by training
    /meta
        episode_ends         (E,) int64        cumulative episode boundary indices
        tactile_baseline     (2, 9, 3) float32  one-time idle field (informational)
        camera_serials, camera_intrinsics_native, camera_native_size,
        agent_camera_serial, wrist_camera_serial, trc_agent, recorded_img_size
                                                 camera metadata (informational)

Output LeRobot dataset (matches openpi's LIBERO schema, so LeRobotLiberoDataConfig works as-is):
    image          uint8 (224, 224, 3)  <- img_0 in the chosen overlay mode (see --mode)
    wrist_image    uint8 (224, 224, 3)  <- img_1 in the chosen overlay mode
    state          float32 (8,)         <- [ee_pos_m(3), ee_ori_axis_angle_rad(3), grasp, grasp]
    actions        float32 (7,)         <- [dxyz_m(3), daxis_angle_rad(3), grasp{-1=open,+1=close}]

Action = tcp[t+1] - tcp[t] (delta), matching pi0.5/LIBERO convention. The last frame of each
episode is dropped because there is no t+1. Leading "paused" frames are also trimmed so the
policy doesn't learn 'this scene -> noop' attractors.

The tactile overlay (if any) is *part of the training signal*. At deployment, the same overlay
must be re-rendered onto the live camera feed before the model sees it. Pass --mode raw to
fine-tune on un-overlaid frames; pass --mode arrow/grid/point/bar to fine-tune on a specific
overlay rendering. The five modes are ablation-comparable because the underlying recording is
identical — only the burned-in overlay differs.

Usage:
    uv run examples/xarm/convert_zarr_to_lerobot.py \\
        --zarr /data/edward/teleop_data_overlay.zarr \\
        --mode arrow \\
        --repo-id local/xarm_teleop_arrow \\
        --language "pick up the red block"
"""

import dataclasses
import logging
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import tyro
import zarr

logger = logging.getLogger(__name__)

# Match the reference converter (phone_data_bridge/zarr_to_libero_hdf5.py).
PAUSE_XYZ_MM_THRESHOLD = 0.5
PAUSE_RPY_DEG_THRESHOLD = 0.5
CONTROL_HZ = 10


# Overlay modes available in teleop_data_overlay.zarr (output of
# scripts/render_overlays.py). "raw" reads the bare img_{i} arrays
# (un-overlaid frames); any other value reads img_{i}_<mode>.
# The newer overlay renderer (post-2026-05-19) uses points{N}_arrow style
# names where N is the number of arrows per finger; we accept those too so
# this converter handles both schemas without manual fix-up.
_OVERLAY_MODES = (
    "raw",
    "arrow", "grid", "point", "bar",                 # legacy renderer
    "points1_arrow", "points9_arrow",                # new renderer (preferred)
    "points1_contact_flat", "points1_contact_spatial",
    "points9_color_flat", "points9_color_spatial",
)


@dataclasses.dataclass
class Args:
    zarr: Path = Path("/data/edward/teleop_data_overlay.zarr")
    repo_id: str = "local/xarm_teleop"
    language: str = "pick up the red block"
    # Which overlay variant to use as the training image. "raw" = no overlay;
    # arrow/grid/point/bar = the corresponding rendering from render_overlays.py.
    # The repo_id is NOT auto-suffixed with the mode — pick a distinct repo_id
    # per ablation (e.g. local/xarm_teleop_arrow) so they don't overwrite.
    mode: str = "arrow"
    # If True, skip episodes with no grasp open->close transition (operator forgot to grasp).
    require_grasp_transition: bool = True


def _pick_image_key(g: zarr.Group, cam_idx: int, mode: str) -> str:
    """Resolve the zarr key for camera ``cam_idx`` in the requested overlay mode.

    Schema (post-2026-05-15 refactor of the tactile-data-collection repo):
      - mode == "raw"  -> data/img_{cam_idx}            (un-overlaid)
      - mode == "arrow"/"grid"/"point"/"bar"
                       -> data/img_{cam_idx}_<mode>     (overlay burned in)
    """
    if mode not in _OVERLAY_MODES:
        raise ValueError(f"mode must be one of {_OVERLAY_MODES}, got {mode!r}")
    key = f"data/img_{cam_idx}" if mode == "raw" else f"data/img_{cam_idx}_{mode}"
    if key not in g:
        available = sorted(k for k in g["data"].keys() if k.startswith("img_"))
        raise KeyError(
            f"missing zarr key {key!r}; available image keys: {available}. "
            f"Did you run scripts/render_overlays.py on the raw zarr first?"
        )
    return key


def _tcp_to_delta_actions(tcp_pose_mm_deg: np.ndarray) -> np.ndarray:
    """(T, 6) absolute TCP [mm + deg] -> (T-1, 6) per-step delta [m + axis-angle rad]."""
    xyz_m = tcp_pose_mm_deg[:, :3] / 1000.0
    rot = Rot.from_euler("xyz", tcp_pose_mm_deg[:, 3:6], degrees=True)
    delta_xyz = (xyz_m[1:] - xyz_m[:-1]).astype(np.float32)
    delta_rot = (rot[1:] * rot[:-1].inv()).as_rotvec().astype(np.float32)
    return np.concatenate([delta_xyz, delta_rot], axis=1)


def _euler_deg_to_axis_angle(rpy_deg: np.ndarray) -> np.ndarray:
    return Rot.from_euler("xyz", rpy_deg, degrees=True).as_rotvec().astype(np.float32)


def _count_leading_paused(tcp: np.ndarray) -> int:
    if len(tcp) < 2:
        return 0
    dxyz = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1)
    drpy = np.linalg.norm(np.diff(tcp[:, 3:6], axis=0), axis=1)
    is_paused = np.concatenate([[True], (dxyz < PAUSE_XYZ_MM_THRESHOLD) & (drpy < PAUSE_RPY_DEG_THRESHOLD)])
    return int(np.argmax(~is_paused)) if not is_paused.all() else len(is_paused)


def _float01_to_uint8(img_f: np.ndarray) -> np.ndarray:
    # Newer renderer writes uint8 images directly; legacy renderer wrote float32 in [0, 1].
    if img_f.dtype == np.uint8:
        return img_f
    return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    g = zarr.open(str(args.zarr), mode="r")
    ep_ends = np.asarray(g["meta/episode_ends"][:])
    img0_key = _pick_image_key(g, 0, args.mode)
    img1_key = _pick_image_key(g, 1, args.mode)
    logger.info("zarr=%s, mode=%s, episodes=%d, image keys: %s, %s",
                args.zarr, args.mode, len(ep_ends), img0_key, img1_key)

    out_path = HF_LEROBOT_HOME / args.repo_id
    if out_path.exists():
        logger.info("Removing existing LeRobot dataset at %s", out_path)
        shutil.rmtree(out_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="xarm",
        fps=CONTROL_HZ,
        features={
            "image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    ep_start = 0
    n_written = n_skipped = n_frames = 0
    for ep_idx, ep_end in enumerate(ep_ends.tolist()):
        ep_end = int(ep_end)
        state = np.asarray(g["data/state"][ep_start:ep_end], dtype=np.float32)
        img_a = np.asarray(g[img0_key][ep_start:ep_end])
        img_w = np.asarray(g[img1_key][ep_start:ep_end])
        ep_start = ep_end

        tcp = state[:, :6]
        grasp = state[:, 6]

        n_lead = _count_leading_paused(tcp)
        if n_lead > 0:
            tcp, grasp, img_a, img_w = tcp[n_lead:], grasp[n_lead:], img_a[n_lead:], img_w[n_lead:]

        if len(tcp) < 5:
            logger.warning("episode %d: only %d steps after trim; skipping", ep_idx, len(tcp))
            n_skipped += 1
            continue

        # Action is delta to next state, so we drop the last frame.
        delta = _tcp_to_delta_actions(tcp)
        grasp_pm1 = (2.0 * grasp[:-1] - 1.0).astype(np.float32).reshape(-1, 1)
        actions = np.concatenate([delta, grasp_pm1], axis=1)

        if args.require_grasp_transition and int((np.diff(grasp) != 0).sum()) == 0:
            logger.warning("episode %d: no grasp transition; skipping", ep_idx)
            n_skipped += 1
            continue

        ee_pos_m = (tcp[:-1, :3] / 1000.0).astype(np.float32)
        ee_axis_angle = np.stack([_euler_deg_to_axis_angle(rpy) for rpy in tcp[:-1, 3:6]], axis=0)
        # LIBERO uses 2-finger gripper_qpos; we duplicate the single binary grasp so the
        # 8-dim state slot lines up with what pi05_libero was trained on.
        grasp_dup = np.stack([grasp[:-1], grasp[:-1]], axis=1).astype(np.float32)
        obs_state = np.concatenate([ee_pos_m, ee_axis_angle, grasp_dup], axis=1)

        agentview = _float01_to_uint8(img_a[:-1])
        wrist = _float01_to_uint8(img_w[:-1])

        for t in range(actions.shape[0]):
            dataset.add_frame(
                {
                    "image": agentview[t],
                    "wrist_image": wrist[t],
                    "state": obs_state[t],
                    "actions": actions[t],
                    "task": args.language,
                }
            )
        dataset.save_episode()
        n_written += 1
        n_frames += int(actions.shape[0])
        logger.info("  episode %d -> %d frames (cumulative: %d eps, %d frames)", ep_idx, actions.shape[0], n_written, n_frames)

    logger.info("Done. Wrote %d/%d episodes (%d skipped), %d total frames to %s",
                n_written, len(ep_ends), n_skipped, n_frames, out_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
