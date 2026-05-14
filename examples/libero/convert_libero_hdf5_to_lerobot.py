"""Convert raw LIBERO HDF5 demos into a LeRobot dataset for openpi fine-tuning.

The raw LIBERO release ships per-task HDF5 files under
``<LIBERO_root>/libero/datasets/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/*.hdf5``.
Each file contains ``data/demo_<i>/{actions, obs/{agentview_rgb, eye_in_hand_rgb, joint_states,
gripper_states, ee_pos, ee_ori, ...}}`` for ~50 episodes of one task.

We convert this to a LeRobot dataset with keys matching the openpi LIBERO pipeline
(see ``LeRobotLiberoDataConfig`` in ``src/openpi/training/config.py``)::

    image          uint8  (H, W, 3)   <- agentview_rgb (rotated 180 to match eval)
    wrist_image    uint8  (H, W, 3)   <- eye_in_hand_rgb (rotated 180 to match eval)
    state          float32 (8,)       <- [ee_pos (3), ee_ori axis-angle (3), gripper_qpos (2)]
    actions        float32 (7,)       <- delta xyz + delta axis-angle + gripper

Usage:
    uv run examples/libero/convert_libero_hdf5_to_lerobot.py \
        --raw-dir /data/edward/openpi/LIBERO/libero/datasets \
        --suite libero_spatial \
        --repo-id local/libero_spatial
"""

import dataclasses
import logging
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

logger = logging.getLogger(__name__)

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


@dataclasses.dataclass
class Args:
    raw_dir: Path = Path("/data/edward/LIBERO/libero/datasets")
    suite: str = "libero_spatial"
    repo_id: str | None = None
    image_size: int = 128
    skip_noops: bool = True
    noop_threshold: float = 1e-3


def _task_from_filename(p: Path) -> str:
    stem = p.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    for scene_prefix in ("KITCHEN_SCENE", "LIVING_ROOM_SCENE", "STUDY_SCENE", "BATHROOM_SCENE"):
        for i in range(20):
            tag = f"{scene_prefix}{i}_"
            if stem.startswith(tag):
                stem = stem[len(tag) :]
                break
    return stem.replace("_", " ")


def _build_state(demo: h5py.Group) -> np.ndarray:
    ee_pos = np.asarray(demo["obs/ee_pos"], dtype=np.float32)
    ee_ori = np.asarray(demo["obs/ee_ori"], dtype=np.float32)
    gripper = np.asarray(demo["obs/gripper_states"], dtype=np.float32)
    return np.concatenate([ee_pos, ee_ori, gripper], axis=-1)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.suite not in SUITES:
        raise ValueError(f"--suite must be one of {SUITES}, got {args.suite}")

    suite_dir = args.raw_dir / args.suite
    files = sorted(suite_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files under {suite_dir}")

    repo_id = args.repo_id or f"local/{args.suite}"
    out_path = HF_LEROBOT_HOME / repo_id
    if out_path.exists():
        logger.info("Removing existing LeRobot dataset at %s", out_path)
        shutil.rmtree(out_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=20,
        features={
            "image": {"dtype": "image", "shape": (args.image_size, args.image_size, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (args.image_size, args.image_size, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_episodes_total = 0
    n_frames_total = 0
    for hdf5_path in files:
        task_name = _task_from_filename(hdf5_path)
        logger.info("=== %s :: %s", hdf5_path.name, task_name)
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
            for dk in demo_keys:
                demo = f["data"][dk]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                state = _build_state(demo)
                # LIBERO renders are flipped vertically and horizontally vs. the eval
                # frame fed to the policy (see examples/libero/main.py: img[::-1, ::-1]).
                agentview = np.asarray(demo["obs/agentview_rgb"])[:, ::-1, ::-1, :]
                wrist = np.asarray(demo["obs/eye_in_hand_rgb"])[:, ::-1, ::-1, :]
                if args.image_size != agentview.shape[1]:
                    raise ValueError(
                        f"image_size={args.image_size} but raw images are {agentview.shape[1]}; "
                        "set --image-size to match the source"
                    )

                if args.skip_noops:
                    mag = np.linalg.norm(actions[:, :6], axis=-1)
                    keep = mag > args.noop_threshold
                    if not keep.any():
                        logger.warning("  %s: all frames are no-ops, skipping", dk)
                        continue
                    actions = actions[keep]
                    state = state[keep]
                    agentview = agentview[keep]
                    wrist = wrist[keep]

                for t in range(actions.shape[0]):
                    dataset.add_frame(
                        {
                            "image": agentview[t],
                            "wrist_image": wrist[t],
                            "state": state[t],
                            "actions": actions[t],
                            "task": task_name,
                        }
                    )
                dataset.save_episode()
                n_episodes_total += 1
                n_frames_total += int(actions.shape[0])
        logger.info("  cumulative: %d episodes, %d frames", n_episodes_total, n_frames_total)

    logger.info("Done. Wrote %d episodes / %d frames to %s", n_episodes_total, n_frames_total, out_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
