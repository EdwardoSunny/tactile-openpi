# Testing & running the xArm fine-tune

End-to-end guide to verify a `pi05_xarm_finetune_lora` checkpoint and put it on the robot. Run the stages **in order** — each one rules out a class of failure before you risk the next.

| Stage | What it tests | Hardware needed | Risk |
|---|---|---|---|
| 0. Install extras | venv has xArm SDK + RealSense | none | none |
| 1. Smoke imports | openpi + config + ckpt readable | none | none |
| 2. Offline sanity check | model outputs sensible values | none | none |
| 3. Robot reachability | xArm IP responds | xArm powered, on network | none |
| 4. Camera check | cameras enumerable + framing right | cameras | none |
| 5. Dry run | full inference loop, no servo commands | cameras (+ optional arm) | none |
| 6. Slow live rollout | model actually drives the arm | arm + cameras | low |
| 7. Full-speed rollout | production speed run | arm + cameras | medium |

If any stage fails, stop and debug *before* moving to the next.

---

## 0. One-time install

The openpi venv at `/data/edward/openpi/.venv` already has JAX, openpi, scipy, opencv, zarr. Add the inference-only deps:

```bash
/data/edward/openpi/.venv/bin/pip install -r examples/xarm/inference/requirements.txt
```

Installs `xArm-Python-SDK`, `pyrealsense2`, `opencv-python` (last one is no-op if already present). Skip `pyrealsense2` if you only use USB cameras.

---

## 1. Smoke imports

Verifies the venv can load the config and locate the checkpoint:

```bash
/data/edward/openpi/.venv/bin/python -c "
from openpi.training import config as _config
cfg = _config.get_config('pi05_xarm_finetune_lora')
print('config:', cfg.name)
print('loader:', cfg.weight_loader)
print('repo:  ', cfg.data.repo_id)
"
```

Expected output:
```
config: pi05_xarm_finetune_lora
loader: CheckpointWeightLoader(params_path='gs://openpi-assets/checkpoints/pi05_droid/params')
repo:   local/xarm_teleop
```

Also check the checkpoint dir exists and has both `params/` and `assets/`:

```bash
ls /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999/
# expected: _CHECKPOINT_METADATA  assets  params  train_state
```

---

## 2. Offline sanity check (most important test)

Pulls training-distribution frames from the zarr, runs them through the policy, and checks each predicted action chunk against ground truth. **No robot, no cameras needed.**

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  /data/edward/openpi/.venv/bin/python examples/xarm/inference/sanity_check.py \
  --checkpoint /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999
```

For each of 3 episodes, prints predictions for 3 phases (`approach`, `pre-grasp`, `lift`). Look for these patterns:

| Phase | What the predicted chunk should show |
|---|---|
| **approach** (state z ≈ 0.2 m) | `pred z-trend: DOWN`, all 10 Δz values negative, grasp held ≈ −1 (open) |
| **pre-grasp** (state z ≈ 0.03 m) | Δz decelerating to ≈ 0, **grasp flips −1 → +1 mid-chunk** |
| **lift** (state z ≈ 0.01 m, grasp = 1) | grasp held ≈ +1 (closed), Δz turns positive at step ~8 |

Per-step abs errors should be **sub-mm in xyz** (≤ 1.5 mm), **sub-mrad in axis-angle** (≤ 0.005 rad), and **near-zero in grasp** (≤ 0.04). If errors are 10× bigger or trends are wrong, the checkpoint is bad — don't proceed.

---

## 3. xArm reachability

Connect to the arm, read its TCP pose, disconnect. Confirms IP + power + network:

```bash
/data/edward/openpi/.venv/bin/python -c "
from xarm.wrapper import XArmAPI
arm = XArmAPI('192.168.1.223')
arm.connect()
code, pose = arm.get_position()
print('connected:', arm.connected, 'code:', code, 'pose:', pose)
arm.disconnect()
"
```

If the arm is at the known home joint config `[0, 0, 0, 70, 0, 70, 0]`, the printed TCP pose should be ≈ `(475.79, -1.14, 244.72, 179.13, -0.01, 0.78)` (mm, deg). If the arm is somewhere else, that's fine — the live script's homing step will move it to that reference pose.

**Common failures**: timeout → wrong IP or arm not powered. Connect=True but pose all zeros → arm needs `motion_enable(True)`.

---

## 4. Camera check

### RealSense

```bash
/data/edward/openpi/.venv/bin/python -c "
import pyrealsense2 as rs
for d in rs.context().query_devices():
    print(d.get_info(rs.camera_info.name), '->', d.get_info(rs.camera_info.serial_number))
"
```

Note the two serial numbers — those are the values for `--agent-cam-id` and `--wrist-cam-id`. The defaults `0` and `2` correspond to the first and third connected devices (1280×720 @ 30 fps, exp=120, wb=5900K — match the collection-time settings).

### USB

```bash
/data/edward/openpi/.venv/bin/python -c "
import cv2
for i in range(4):
    c = cv2.VideoCapture(i)
    ok, _ = c.read()
    print(f'cam {i}:', 'OK' if ok else 'no signal')
    c.release()
"
```

Pick the indices that responded `OK`, in the order that matches agent-view + wrist-view of your rig.

### Visual check (optional)

Capture and save one frame from each:

```bash
/data/edward/openpi/.venv/bin/python -c "
import cv2, pyrealsense2 as rs
for serial, name in (('AGENT_SERIAL', 'agent'), ('WRIST_SERIAL', 'wrist')):
    p = rs.pipeline()
    c = rs.config(); c.enable_device(serial); c.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    p.start(c)
    for _ in range(30): p.wait_for_frames()
    img = np.asanyarray(p.wait_for_frames().get_color_frame().get_data())
    cv2.imwrite(f'/tmp/{name}.png', img); p.stop()
"
```

Open `/tmp/agent.png` and `/tmp/wrist.png` and confirm the agent view shows the table from where the policy expects to see it.

---

## 5. Dry run

Full inference loop with cameras and video recording, **but no servo commands to the arm**. This catches camera framing, image-preprocessing, and inference issues without any robot risk:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  /data/edward/openpi/.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
  --checkpoint /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999 \
  --dry-run --max-steps 30
```

Expected log lines:
```
… Policy loaded.
… RealSense initialized (agent=0, wrist=2, 1280x720@30, exp=120, wb=5900K)
… Recording rollout to rollouts/xarm_inference_<timestamp>.mp4
… step=  0  dxyz=[+0.0050 +0.0010 -0.0070] m  daa=[+0.0010 +0.0020 +0.0040] rad  grasp=-0.99
…
```

Then watch the saved `.mp4` to confirm the camera views look like what the policy trained on.

---

## 6. Slow live rollout (first robot test)

`--action-scale 0.5` halves the magnitude of every commanded delta; `--max-action-norm 0.03` clips any single-step Cartesian motion over 3 cm. `--max-steps 50` caps the rollout at ≈ 5 seconds:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  /data/edward/openpi/.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
  --checkpoint /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999 \
  --xarm-ip 192.168.1.223 \
  --action-scale 0.5 \
  --max-action-norm 0.03 \
  --max-steps 50
```

What happens:
1. Arm switches to position mode, opens the gripper, moves to `[0, 0, 0, 70, 0, 70, 0]` (joint angles).
2. Script logs the reached TCP pose — should be `(475.79, -1.14, 244.72, 179.13, -0.01, 0.78)` ± a few mm.
3. Script prints `Press ENTER to begin rollout…` — **keep one hand on Ctrl+C, then press ENTER**.
4. Robot executes 50 commanded deltas at 10 Hz, then returns to home and disconnects.

Watch for: smooth motion toward the block, gripper closes when the EE is over the object, lift after close. Sub-mm tracking errors are normal; centimeter-scale deviation from the demo trajectory is fine on a slow run.

If anything looks unsafe — Ctrl+C immediately. The script catches SIGINT, triggers `arm.set_state(4)` (emergency stop), homes, and exits cleanly.

---

## 7. Full-speed rollout

Once the slow run looks safe, drop back to defaults:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  /data/edward/openpi/.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
  --checkpoint /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999 \
  --xarm-ip 192.168.1.223 \
  --max-steps 200
```

Defaults: `--action-scale 1.0`, `--max-action-norm 0.05`, 200 steps (≈ 20 s of execution at 10 Hz). Re-records the rollout video and homes the arm at the end.

---

## CLI reference

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
--dry-run                      # full loop, but don't send servo commands to the arm
--no-video                     # skip video recording
--video-path PATH              # default: rollouts/xarm_inference_<timestamp>.mp4
```

---

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

Why joint-space, not Cartesian: same joint angles always produce the same arm configuration — no IK ambiguity, no risk of hitting a singularity on the way home.

## Why we don't go through `ril_env.XArm.step()`

`ril_env.XArm.step(dpos, drot, grasp)` applies `position_gain = orientation_gain = 2.0` to the delta — that's a teleop ergonomic choice (operator gets 2× the motion they push). Our training labels are *measured* TCP deltas (`tcp[t+1] − tcp[t]`), so the model output is already the desired physical motion per step. Going through `step()` would double it. This script bypasses the gain by calling `set_servo_cartesian(cur + model_delta)` directly.

If you want to route through the ril_env wrapper anyway, pre-divide the 6-D delta by `position_gain` and `orientation_gain` (= 2.0 each) before passing it to `step()`.

## Tactile arrow overlay

The `apply_tactile_overlay(img, tactile_reading)` hook in the inference script currently returns the image unchanged. The dataset we trained on had `n_contacts` all-zero, so no arrows were drawn at training time — the policy effectively learned on plain camera frames. Identity at inference is correct for **this** checkpoint.

For future retraining on data with real tactile contacts, fill this hook in with the same renderer used at collection. Otherwise the deployment pixel distribution diverges from training and the policy will fall out of distribution.

## Troubleshooting

**`from xarm.wrapper import XArmAPI` fails**
→ `pip install xArm-Python-SDK` into the openpi venv. The PyPI package name has a hyphen and capital A; the import module is lowercase `xarm`.

**`policy.infer` returns NaN actions**
→ Norm stats are stale or wrong. Re-run `compute_norm_stats.py --config-name pi05_xarm_finetune_lora` and confirm a fresh `norm_stats.json` appears under `assets/pi05_xarm_finetune_lora/local/xarm_teleop/`.

**Robot moves the wrong direction in one axis**
→ Almost certainly an Euler-convention mismatch. The converter and inference script both use scipy `from_euler("xyz", …, degrees=True)` — if your xArm firmware reports `zyx` or similar, *both* would have to be flipped together. Test on the bench with very small motions before doing a real task.

**Servo command returns non-zero code mid-rollout**
→ Usually means the target exceeded a joint limit or collision threshold. Lower `--action-scale` or `--max-action-norm`. If it happens at step 0, the model's first prediction is OOD — check the home pose matches your training-time start poses.

**Gripper opens when it should close (or vice versa)**
→ The conversion is `+1 = closed → set_gripper_position(0)`. If your gripper is wired inverted (some custom grippers), flip the threshold in `XArmController.execute_delta`.

**Policy outputs look reasonable in sanity_check.py but the arm sits still on the live rollout**
→ Known "noop attractor" mode where the model predicts near-zero deltas when the start scene looks too quiescent. Mitigations: re-collect data with leading paused frames trimmed harder (already enabled in the converter via `_count_leading_paused`), or temporarily inject a small fixed descent for the first few steps. The vla-finetune repo's `--warmup_steps` flag is the precedent.

**Out of GPU memory during policy load**
→ The pi0.5 LoRA model needs ~12 GB. If GPU 0 is occupied, switch with `CUDA_VISIBLE_DEVICES=N`. The model also respects `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (drops to 0.7 if a workspace conflict appears).
