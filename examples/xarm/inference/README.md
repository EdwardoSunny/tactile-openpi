# pi05 xArm fine-tune — testing & deployment

End-to-end guide for taking a trained `pi05_xarm_finetune_lora` checkpoint, putting it on a fresh inference machine, verifying it produces sensible outputs, and running it on the real xArm.

**Run the stages in order.** Each one rules out a class of failure before you risk the next.

| Stage | What it tests | Hardware needed | Risk |
|---|---|---|---|
| [Setup](#setup) — clone, venv, deps | code is ready to run | none | none |
| [1. Download](#1-download-the-checkpoint) the checkpoint | HF / scp / rsync working | none | none |
| [2. Offline sanity](#2-offline-sanity-check) check | model outputs sensible values | none | none |
| [3. Robot](#3-robot-reachability) reachability | xArm IP responds | xArm powered, on network | none |
| [4. Cameras](#4-camera-check) | cameras enumerable + framing right | cameras | none |
| [5. Dry run](#5-dry-run-cameras--inference-no-arm-motion) | full inference loop, no servo commands | cameras (+ optional arm) | none |
| [6. Slow live](#6-slow-live-rollout-first-robot-test) rollout | model actually drives the arm | arm + cameras | low |
| [7. Full-speed](#7-full-speed-rollout) rollout | production speed run | arm + cameras | medium |

If any stage fails, stop and debug before moving on. The first three are cheap to repeat.

---

## Setup

On the **inference machine** (the host wired to the xArm and cameras):

```bash
# 1) clone the openpi fork that has the pi05_xarm_finetune_lora config + inference scripts
git clone <your-fork-url> openpi
cd openpi

# 2) standard openpi install (Python 3.11)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 3) inference-only extras (xArm SDK + RealSense + huggingface_hub)
./.venv/bin/pip install -r examples/xarm/inference/requirements.txt
./.venv/bin/pip install huggingface_hub      # only if downloading from HF
```

The openpi `uv sync` installs JAX, scipy, opencv, zarr, etc. The line above only adds `xArm-Python-SDK`, `pyrealsense2`, and optionally `huggingface_hub`.

Verify the config + checkpoint loader can be imported:

```bash
./.venv/bin/python -c "
from openpi.training import config as _config
cfg = _config.get_config('pi05_xarm_finetune_lora')
print('config:', cfg.name)
print('repo:  ', cfg.data.repo_id)
"
# expected:
# config: pi05_xarm_finetune_lora
# repo:   local/xarm_teleop
```

If this fails, the openpi clone is missing the config edits.

---

## 1. Download the checkpoint

You need ~7 GB of weights + norm stats. Three ways to get them, in decreasing order of recommendedness:

### Option A — Hugging Face Hub (recommended)

```bash
# Log in if the repo is still private. (Public repos: skip login.)
./.venv/bin/huggingface-cli login

# Pull. Defaults to fetching only the inference essentials (params + assets + metadata);
# pass --include-train-state if you intend to resume training too.
./.venv/bin/python examples/xarm/inference/pull_from_hub.py \
    --repo-id EdwardoSunny/pi05-xarm-finetune-lora-droid-init-20k \
    --local-dir ./checkpoints/pi05_xarm_finetune_lora_pulled
```

Public URL: <https://huggingface.co/EdwardoSunny/pi05-xarm-finetune-lora-droid-init-20k>

### Option B — rsync from the training host

If both machines can reach each other over SSH:

```bash
rsync -avP --exclude='train_state' \
    edward@<training-host>:/data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999/ \
    ./checkpoints/pi05_xarm_finetune_lora_pulled/
```

The trailing slashes matter — they copy *the contents* of `19999/` into the destination.

### Option C — tarball (air-gapped / USB / S3)

```bash
# on the training host
cd /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k
tar --exclude='train_state' -czvf pi05_xarm_finetune_lora_19999.tar.gz 19999/

# transfer the .tar.gz however you like, then on the inference machine:
mkdir -p ./checkpoints/pi05_xarm_finetune_lora_pulled
tar -xzvf pi05_xarm_finetune_lora_19999.tar.gz \
    -C ./checkpoints/pi05_xarm_finetune_lora_pulled --strip-components=1
```

### Confirm the layout

After any of the three options:

```bash
ls ./checkpoints/pi05_xarm_finetune_lora_pulled
# expected:
#   _CHECKPOINT_METADATA  params/  assets/    (and train_state/ if you included it)

ls ./checkpoints/pi05_xarm_finetune_lora_pulled/assets/local/xarm_teleop/
# expected:
#   norm_stats.json
```

If either listing differs, the download was incomplete — re-run the pull.

---

## 2. Offline sanity check

**No robot, no cameras needed.** Pulls training-distribution frames from the source zarr, runs them through the policy, and compares each predicted action chunk to the ground-truth `tcp[t+1] - tcp[t]` chunk built from the same demo.

For this you also need the zarr the model was trained on. If you have it locally, point at it:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  ./.venv/bin/python examples/xarm/inference/sanity_check.py \
    --checkpoint ./checkpoints/pi05_xarm_finetune_lora_pulled \
    --zarr-path /path/to/your/teleop_data.zarr
```

(The default `--zarr-path /data/edward/teleop_data.zarr` only works on the training host.)

For each of 3 episodes, you'll see predictions for 3 phases. Expected patterns:

| Phase | What the predicted chunk should show |
|---|---|
| **approach** (state z ≈ 0.2 m) | `pred z-trend: DOWN`, all 10 Δz values negative, grasp held ≈ −1 (open) |
| **pre-grasp** (state z ≈ 0.03 m) | Δz decelerating to ≈ 0, **grasp flips −1 → +1 mid-chunk** |
| **lift** (state z ≈ 0.01 m, grasp = 1) | grasp held ≈ +1 (closed), Δz turns positive at step ~8 |

Per-step abs errors should be **sub-mm in xyz** (≤ 1.5 mm), **sub-mrad in axis-angle** (≤ 0.005 rad), and **near-zero in grasp** (≤ 0.04). If errors are 10× bigger or trends are wrong, the checkpoint download is corrupt or the model is bad — **stop here** and don't move on to robot tests.

If you don't have the training zarr on the inference machine, skip this step and rely on the dry-run (stage 5) to verify the inference path.

---

## 3. Robot reachability

Connect to the arm, read its TCP pose, disconnect. Confirms IP + power + network:

```bash
./.venv/bin/python -c "
from xarm.wrapper import XArmAPI
arm = XArmAPI('192.168.1.223')
arm.connect()
code, pose = arm.get_position()
print('connected:', arm.connected, 'code:', code, 'pose:', pose)
arm.disconnect()
"
```

If the arm happens to be at the home joint config `[0, 0, 0, 70, 0, 70, 0]`, the printed TCP pose should be ≈ `(475.79, -1.14, 244.72, 179.13, -0.01, 0.78)` (mm, deg). It's fine if the arm is somewhere else — the live script will home it.

**Common failures**: timeout → wrong IP or arm not powered. Connect=True but pose all zeros → arm needs `motion_enable(True)`.

---

## 4. Camera check

The inference script defaults to plain OpenCV `cv2.VideoCapture(int)` — same as production data collection (`collect_xarm_demos.py`). RealSense cameras also enumerate as UVC devices through this path, so you don't need pyrealsense2 unless you specifically want the SDK pipeline.

### Default — OpenCV (recommended)

```bash
./.venv/bin/python -c "
import cv2
for i in range(8):
    c = cv2.VideoCapture(i); ok, _ = c.read()
    print(f'cam {i}:', 'OK' if ok else 'no signal'); c.release()
"
```

Pick the indices that responded `OK`. Default mapping is `--agent-cam-id 0 --wrist-cam-id 2`, matching collection-time defaults. If your USB-bus enumeration differs, override on the CLI.

### Opt-in — RealSense SDK

If you specifically need the RealSense pipeline (different exposure / WB controls, depth streams, etc.), pass `--use-realsense`. In that mode `--agent-cam-id` and `--wrist-cam-id` must be **real D400 serial numbers**, not USB indices. Discover them with:

```bash
./.venv/bin/python -c "
import pyrealsense2 as rs
for d in rs.context().query_devices():
    print(d.get_info(rs.camera_info.name), '->', d.get_info(rs.camera_info.serial_number))
"
```

Note: the model in `EdwardoSunny/pi05-xarm-finetune-lora-droid-init-20k` was trained on data captured via the OpenCV path. Using RealSense at inference is fine optically, but won't match training-time pixel statistics any closer than OpenCV does.

### Visual check (optional but recommended)

Save one frame from each camera and open it — confirm the agent view shows the table from roughly the angle the training demos used:

```bash
./.venv/bin/python -c "
import cv2
for src, name in [(0, 'agent'), (2, 'wrist')]:
    c = cv2.VideoCapture(src); ok, frame = c.read(); c.release()
    if ok: cv2.imwrite(f'/tmp/{name}.png', frame); print(f'{name} saved')
    else:   print(f'{name} FAILED')
"
```

---

## 5. Dry run (cameras + inference, no arm motion)

Full inference loop with cameras and video recording, **but no servo commands**. This catches camera framing, image preprocessing, and inference issues without any robot risk:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  ./.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
    --checkpoint ./checkpoints/pi05_xarm_finetune_lora_pulled \
    --dry-run --max-steps 30
```

Expected log:

```
… Policy loaded.
… RealSense initialized (agent=0, wrist=2, 1280x720@30, exp=120, wb=5900K)
… Recording rollout to rollouts/xarm_inference_<timestamp>.mp4
… step=  0  dxyz=[+0.0050 +0.0010 -0.0070] m  daa=[+0.0010 +0.0020 +0.0040] rad  grasp=-0.99
…
```

After it finishes, open the saved `.mp4` (side-by-side agent + wrist) and confirm the camera views look like what the policy trained on. If the views look very different (different table, different angle, different lighting), expect the live rollout to behave poorly — fix the camera setup, not the model.

---

## 6. Slow live rollout (first robot test)

The first time you run on the real arm, use conservative safety flags:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  ./.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
    --checkpoint ./checkpoints/pi05_xarm_finetune_lora_pulled \
    --xarm-ip 192.168.1.223 \
    --action-scale 0.5 \
    --max-action-norm 0.03 \
    --max-steps 50
```

Flag meanings:
- `--action-scale 0.5` — half-magnitude every commanded delta
- `--max-action-norm 0.03` — clip any single-step 6-D motion above 3 cm
- `--max-steps 50` — cap rollout at 5 seconds (50 × 0.1 s at 10 Hz)

What happens:
1. Arm switches to position mode, opens the gripper, homes to joint angles `[0, 0, 0, 70, 0, 70, 0]`.
2. Script logs the reached TCP pose — should be ≈ `(475.79, -1.14, 244.72, 179.13, -0.01, 0.78)` ± a few mm. If it's wildly off, the arm's URDF / calibration differs from the lab rig; review before pressing ENTER.
3. Script prints `Press ENTER to begin rollout…` — **keep one hand on Ctrl+C, then press ENTER**.
4. Robot executes 50 commanded deltas at 10 Hz, then returns to home and disconnects.

Watch for: smooth descent toward the block, gripper closes when the EE is over the object, lift after close. Sub-mm tracking errors are normal; centimeter-scale deviation from the demo trajectory is fine on a slow run.

If anything looks unsafe, Ctrl+C. The script catches SIGINT, triggers `arm.set_state(4)` (emergency stop), homes, and exits.

---

## 7. Full-speed rollout

Once the slow run looks safe, drop the safety flags back to defaults:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  ./.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
    --checkpoint ./checkpoints/pi05_xarm_finetune_lora_pulled \
    --xarm-ip 192.168.1.223 \
    --max-steps 200
```

Defaults: `--action-scale 1.0`, `--max-action-norm 0.05`, 200 steps (~20 s of execution at 10 Hz). Re-records the rollout video and homes the arm at the end.

---

# Reference

## CLI reference (`run_xarm_inference.py`)

```
--checkpoint PATH              # required, path to step-N checkpoint dir
--train-config-name NAME       # default: pi05_xarm_finetune_lora
--xarm-ip IP                   # default: 192.168.1.223
--control-hz N                 # default: 10  (matches training fps)
--agent-cam-id ID              # default: "0"  (RealSense serial or USB index)
--wrist-cam-id ID              # default: "2"
--use-usb-cameras              # OpenCV USB capture instead of RealSense
--prompt TEXT                  # default: "pick up the red block"
--max-steps N                  # default: 200
--replan-steps N               # default: 5  (consume N of 10-step chunk, then re-infer)
--action-scale F               # default: 1.0  (scale on the 6-D pose delta)
--max-action-norm F            # default: 0.05  (clip per-step 6-D delta norm, meters)
--auto-start                   # skip the manual ENTER prompt
--dry-run                      # full loop, but don't send servo commands
--no-video                     # skip video recording
--video-path PATH              # default: rollouts/xarm_inference_<timestamp>.mp4
```

## CLI reference (`sanity_check.py`)

```
--checkpoint PATH              # required, path to step-N checkpoint dir
--train-config-name NAME       # default: pi05_xarm_finetune_lora
--zarr-path PATH               # default: /data/edward/teleop_data.zarr
--prompt TEXT                  # default: "pick up the red block"
--n-episodes N                 # default: 3
--horizon N                    # default: 10
```

## CLI reference (`pull_from_hub.py` / `push_to_hub.py`)

```
# pull
--repo-id REPO_ID              # required, e.g. EdwardoSunny/pi05-xarm-finetune-lora-droid-init-20k
--local-dir PATH               # required, where to download into
--include-train-state          # also fetch train_state/ (~2 GB), only for resuming training

# push  (run on the training host)
--checkpoint PATH              # required, local checkpoint directory
--repo-id REPO_ID              # required
--private                      # make repo private (default: public)
--exclude-train-state          # skip train_state/ (saves ~2 GB)
--commit-message TEXT          # default: "upload xarm fine-tune checkpoint"
```

## Conventions reference

These are baked into the converter, the training data, and this script. **Don't change one without changing all three.**

| Quantity | Format | Source-of-truth |
|---|---|---|
| state input | `[ee_pos_m(3), ee_axis_angle_rad(3), grasp, grasp]` (8-dim) | `convert_zarr_to_lerobot.py:159-164` |
| action output | `[Δxyz_m, Δaxis_angle_rad, grasp_pm1]` (7-dim) | `convert_zarr_to_lerobot.py:150-152` |
| Δ-rotation composition | `target_R = exp(Δaa) · exp(cur_aa)` (left-multiply) | converter line 80; inference line 315 |
| Euler convention | scipy `"xyz"` intrinsic, degrees in/out | converter line 78; inference line 285 |
| Grasp sign | `+1` ↔ closed (xArm `set_gripper_position(0)`); `−1` ↔ open (`850`) | converter line 151; inference line 334 |
| Image format | 224×224 RGB uint8 | converter line 100; `LiberoInputs._parse_image` |
| Control rate | 10 Hz | converter `CONTROL_HZ = 10` |
| Chunk length | 10, consume 5, replan | inference line 102 |

## Homing

Joint-space, via `arm.set_servo_angle([0, 0, 0, 70, 0, 70, 0], speed=50, wait=True)`, matching `ril_env.xarm_controller.XArmConfig.home_pos`. Maps to TCP ≈ `(475.79 mm, -1.14 mm, 244.72 mm, 179.13°, -0.01°, 0.78°)` on the lab xArm 7.

Joint-space homing is more robust than Cartesian (`set_position`) because there's no IK ambiguity — same joint angles always produce the same arm configuration.

## Why we don't go through `ril_env.XArm.step()`

`ril_env.XArm.step(dpos, drot, grasp)` applies `position_gain = orientation_gain = 2.0` to the delta — that's a teleop ergonomic choice (operator gets 2× the motion they push). Our training labels are *measured* TCP deltas (`tcp[t+1] − tcp[t]`), so the model output is already the desired physical motion per step. Going through `step()` would double it. This script bypasses the gain by calling `set_servo_cartesian(cur + model_delta)` directly.

If you want to route through the ril_env wrapper anyway, pre-divide the 6-D delta by `position_gain` and `orientation_gain` (= 2.0 each) before passing it to `step()`.

## Tactile arrow overlay

Checkpoints trained on overlaid frames (`pi05_xarm_<task>_points{1,9}_arrow_lora`) need the **same** overlay rendered onto live camera frames at inference. Without it, the deployment pixel distribution diverges from training and the policy goes out of distribution.

### Easy path — `--overlay <mode>` (recommended)

```
--overlay points9_arrow      # for a points9 checkpoint
--overlay points1_arrow      # for a points1 checkpoint
```

That's it. The flag resolves to the bundled `examples/xarm/inference/overlay_norm_<mode>.npz` and uses `<mode>` as the renderer's mode_key. **One npz works for all 4 tasks** — `scale_xy` / `scale_z` / `deadband` are identical across tasks (computed jointly across the 4 task zarrs by `phone_data_collection/scripts/compute_overlay_normalization.py`); only `raw_clip_low/high` differ, and the bundled file uses the elementwise union so it's the most permissive correct bound.

For `*_baseline_lora` checkpoints (trained on raw frames, no overlay), **omit** the flag.

### Power-user path — `--overlay-stats <path>` (+ optional `--overlay-mode-key`)

If you have a per-task `overlay_norm.npz` you produced yourself with `extract_overlay_norm.py` (one zarr in, tighter clip bounds), pass it explicitly. Mutually exclusive with `--overlay`.

### Re-generating the bundled npz

If the data collection re-runs, refresh the bundled files:

```bash
./.venv/bin/python examples/xarm/inference/extract_overlay_norm.py \
    /path/to/teleop_data_cube_overlay.zarr \
    /path/to/teleop_data_tube_overlay.zarr \
    /path/to/teleop_data_charger_overlay.zarr \
    /path/to/teleop_data_dishwasher_overlay.zarr \
    --mode points9_arrow \
    --out examples/xarm/inference/overlay_norm_points9.npz
# repeat with --mode points1_arrow --out overlay_norm_points1.npz
```

## What gets transferred (and what doesn't)

| Piece | Where it lives | Needed at inference? |
|---|---|---|
| `params/` (orbax shards, ~7 GB) | inside the checkpoint dir | **yes** — the model weights |
| `assets/local/xarm_teleop/norm_stats.json` | inside the checkpoint dir | **yes** — input/output normalization |
| `_CHECKPOINT_METADATA` | inside the checkpoint dir | yes |
| `train_state/` (~2 GB) | inside the checkpoint dir | **no** — only for resuming training |
| The xArm zarr | `/data/edward/teleop_data.zarr` (training host) | only for `sanity_check.py`, not for live rollout |
| The LeRobot dataset | `~/.cache/huggingface/lerobot/local/xarm_teleop` | no — irrelevant at inference |
| Base `pi05_droid` checkpoint | `gs://openpi-assets/checkpoints/pi05_droid/` | **no** — fully baked into our LoRA weights |

So the minimal package to ship is `params/` + `assets/` + `_CHECKPOINT_METADATA` (≈ 7 GB) plus the openpi repo with our config + inference scripts. Everything else stays on the training host.

## Troubleshooting

**`from xarm.wrapper import XArmAPI` fails**
→ `pip install xArm-Python-SDK` into the openpi venv. The PyPI package name has a hyphen and capital A; the import module is lowercase `xarm`.

**`policy.infer` returns NaN actions**
→ Norm stats are stale or wrong. Confirm the checkpoint has `assets/local/xarm_teleop/norm_stats.json` and that it matches what came out of training. If you don't have access to the original norm stats, recompute them with `compute_norm_stats.py --config-name pi05_xarm_finetune_lora` *but only against the same training zarr* — they must come from the data the model was trained on.

**Robot moves the wrong direction in one axis**
→ Almost certainly an Euler-convention mismatch. The converter and inference script both use scipy `from_euler("xyz", …, degrees=True)`. If your xArm firmware reports `zyx` or similar, *both* would have to be flipped together. Test on the bench with very small motions before doing a real task.

**Servo command returns non-zero code mid-rollout**
→ Usually means the target exceeded a joint limit or collision threshold. Lower `--action-scale` or `--max-action-norm`. If it happens at step 0, the model's first prediction is OOD — check the home pose matches your training-time start poses.

**Gripper opens when it should close (or vice versa)**
→ The conversion is `+1 = closed → set_gripper_position(0)`. If your gripper is wired inverted (some custom grippers), flip the threshold in `XArmController.execute_delta`.

**Policy outputs look reasonable in `sanity_check.py` but the arm sits still on the live rollout**
→ Known "noop attractor" mode — the model predicts near-zero deltas when the start scene looks too quiescent. Mitigations: re-collect data with leading paused frames trimmed harder (already enabled in the converter via `_count_leading_paused`), or temporarily inject a small fixed descent for the first few steps. The vla-finetune repo's `--warmup_steps` flag is the precedent.

**Out of GPU memory during policy load**
→ The pi0.5 LoRA model needs ~12 GB. If GPU 0 is occupied, switch with `CUDA_VISIBLE_DEVICES=N`. The model also respects `XLA_PYTHON_CLIENT_MEM_FRACTION` (drop from 0.9 to 0.7 if a workspace conflict appears).

**Pull from HF asks for credentials**
→ The repo is still private. Either run `huggingface-cli login`, or flip the repo to public in the HF web UI (`Settings → Change visibility`).
