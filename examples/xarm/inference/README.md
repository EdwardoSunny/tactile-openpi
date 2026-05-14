# xArm closed-loop inference for pi0.5 LoRA fine-tune

Single-process script that loads a checkpoint trained with `pi05_xarm_finetune_lora`,
captures images + EE pose from the real xArm, queries the policy, and executes the
returned action chunks at 10 Hz with re-planning.

## One-time setup

```bash
# Extra deps not part of the openpi default install:
/data/edward/openpi/.venv/bin/pip install -r examples/xarm/inference/requirements.txt
```

## Dry run (no robot needed)

Useful for verifying the inference path + camera capture without touching the arm:

```bash
/data/edward/openpi/.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
    --checkpoint /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999 \
    --dry-run --max-steps 30
```

Logs the predicted action per step; writes a side-by-side rollout video to `rollouts/`.

## Real rollout

```bash
/data/edward/openpi/.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
    --checkpoint /data/edward/openpi/checkpoints/pi05_xarm_finetune_lora/droid_init_20k/19999 \
    --xarm-ip 192.168.1.223 \
    --max-steps 200
```

The script homes the arm and **pauses for ENTER** before sending the first servo
command. Pass `--auto-start` to skip that prompt.

USB cameras instead of RealSense:

```bash
... --use-usb-cameras --agent-cam-id 0 --wrist-cam-id 2
```

## Important conventions

These must match what the converter (`examples/xarm/convert_zarr_to_lerobot.py`)
wrote into the LeRobot dataset — otherwise the model gets out-of-distribution
inputs and outputs and the robot will move in ways that don't match the demos.

| Quantity | At inference (this script) | Notes |
|---|---|---|
| state input | `[ee_pos_m (3), ee_axis_angle_rad (3), grasp, grasp]` | Built from `arm.get_position()` + Euler→axis-angle + `(1 − gripper_pos/850)` duplicated |
| action output | `[dxyz_m, daxis_angle_rad, grasp_pm1]` | Composed onto current measured pose; gripper thresholded |
| grasp sign | `+1` → CLOSED (`set_gripper_position(0)`), `-1` → OPEN (`set_gripper_position(850)`) | phone teleop raw `{0=open, 1=closed}` → `2x−1` in dataset |
| images | 224×224 RGB uint8 | LiberoInputs handles the float→uint8 case too |
| control rate | 10 Hz (`--control-hz`) | Matches phone teleop fps |
| chunk length | 10 actions, consume 5, re-plan | Same pattern as `examples/libero/main.py` |

## Tactile arrow overlay

`apply_tactile_overlay(img, tactile_reading)` is a stub that currently returns the
image unchanged. The dataset we trained on had `n_contacts` all-zero, so no arrows
were drawn at training time — the model effectively learned on plain camera frames,
and identity at inference is correct for **this** checkpoint.

For future retraining with real contact data, fill that hook in with whatever
renderer the collection pipeline uses to overlay arrows on `img_0`/`img_1`. If the
deployment pixel distribution diverges from the training one, the policy will fall
out of distribution.

## Homing

Homing is done in **joint space** using `arm.set_servo_angle(...)` with the
known-good rest joint configuration from `ril_env.xarm_controller.XArmConfig`:

```
home_joint_angles_deg = (0, 0, 0, 70, 0, 70, 0)   # 7 joint angles in degrees
home_speed            = 50.0                       # deg/s
```

On the lab xArm 7 this maps to TCP pose ≈ `(475.79 mm, -1.14 mm, 244.72 mm,
179.13°, -0.01°, 0.78°)`. The script logs the actual TCP pose after homing so
you can verify against this reference. If your rig differs, edit
`XArmInferenceConfig.home_joint_angles_deg`.

Joint-space homing is more robust than Cartesian (`set_position`) because there's
no IK ambiguity — same joint angles always produce the same arm configuration.

## Why we don't go through `ril_env.XArm.step()`

`ril_env.XArm.step(dpos, drot, grasp)` applies **`position_gain = orientation_gain
= 2.0`** before commanding the arm — that was a teleop ergonomic choice (operator
gets 2× the motion they push). Our training labels are *measured* TCP deltas
(`tcp[t+1] - tcp[t]` from the demo buffer), so the model's output is already the
desired physical motion per step. Going through `XArm.step()` would double it.
This script calls `set_servo_cartesian(cur + model_delta)` directly, bypassing
the gain.

If you ever want the ril_env wrapper for other reasons, pre-divide the model's
6-D delta by `position_gain` / `orientation_gain` (= 2.0 each) before calling
`step()`.

## Safety

- `Ctrl+C` triggers `emergency_stop()` then homes the arm before disconnect.
- `--max-action-norm 0.05` clips any 6-D delta whose norm exceeds 5 cm. Tune lower
  if you want a slower first rollout.
- `--action-scale 0.5` halves the magnitude of every commanded delta. Useful when
  you don't trust the policy yet.
- `--dry-run` runs the full inference + video loop with **zero** servo commands.

## Troubleshooting

- **"xarm not found"**: `pip install xArm-Python-SDK` into the openpi venv (see
  `requirements.txt`). The package import name is `xarm`.
- **Servo errors with non-zero code**: usually means the robot needs `clean_error()`
  or is in collision-detected state. The script clears errors on connect, but if the
  arm hits a hard limit mid-rollout it'll log the code and continue — stop and home
  manually.
- **Policy outputs look like noise**: confirm the `train_config_name` matches the
  one used at training and that `<checkpoint>/assets/local/xarm_teleop/norm_stats.json`
  is the one written during training (not stale from a different run).
- **Robot drifts away from object**: the policy was trained on demos starting from a
  specific home pose. If the rollout starts far from that pose, the first observation
  is OOD. The `home_pose_mm_deg` default is set near the mean training state — adjust
  if your teleop home was different.
