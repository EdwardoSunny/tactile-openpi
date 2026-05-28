# xArm tactile-overlay fine-tuning — methods

End-to-end recipe for fine-tuning `pi0.5` on xArm phone-teleop data with four overlay representations of the tactile signal, to test whether and how the tactile readout helps a VLA do contact-rich manipulation.

> **Paper-writing checklist** — what this doc covers
>
> ```
> % TODO (paper methods section):
> %  [x] model size               (§ Model + training config)
> %  [x] LoRA / full fine-tuning  (§ Model + training config, Why LoRA subsection)
> %  [x] learning rate            (§ Model + training config)
> %  [x] batch size               (§ Model + training config)
> %  [x] epochs                   (§ Model + training config — "Epochs equivalent" row)
> %  [x] image resolution         (§ Model + training config)
> %  [x] data augmentation        (§ Data augmentation)
> %
> %  [x] model variant            (§ Model + training config — "Base model")
> %  [x] fine-tuning recipe       (§ Model + training config, § Reproducing one variant)
> %  [x] dataset                  (§ Tasks, § Data pipeline)
> %  [x] training steps           (§ Model + training config — "Max steps" + "Early-stop")
> %  [x] evaluation protocol      (§ Evaluation protocol)
> ```

## Headline question

Does visually rendering the gripper's tactile readings onto the camera frame at training time (and matching that exact rendering at deployment) make a pi0.5 finetune better at contact-rich xArm tasks? If yes, which rendering format helps most?

We trained 16 LoRA finetunes — 4 tasks × 4 overlay variants — under identical hyperparameters, on a single shared base checkpoint, with the same data, prompts, and stopping criterion. The only thing that varies is what's drawn on each camera frame.

## Tasks

Four phone-teleop xArm tasks, 100 episodes each, recorded at 10 Hz with two RealSense cameras (agentview + wrist) plus per-finger tactile (3×3 grid of 3-axis force sensors per finger).

| Task | Prompt (baked into `meta/tasks.jsonl`) | Episodes | Frames |
|---|---|---|---|
| **cube** | `pick up the cube` | 100 | 13,308 |
| **tube** | `pick up the tube and put it in the nearest slot` | 100 | 18,608 |
| **charger** | `pick up the charger and plug it into the nearest plug` | 100 | 24,009 |
| **dishwasher** | `pull the basket outside of the dishwasher and pick up the mug and put it into the basket` | 100 | 28,298 |

Source data lives at `~edwardosunny/data/teleop_data_<task>.zarr` (raw, no overlay) and `~edwardosunny/data/teleop_data_<task>_overlay.zarr` (post-rendered).

## The four overlay variants

All four use the same camera frames + same tactile readings + same prompts + same model and hyperparameters. Only the burned-in overlay differs:

| Variant | What's drawn on each 224×224 frame |
|---|---|
| **baseline** (`_baseline_lora`) | Nothing — raw camera image. The control. |
| **points1_arrow** (`_points1_arrow_lora`) | One arrow per finger at the gripper-tip position; arrow direction = aggregate force direction, length = aggregate magnitude. Spatially-grounded but coarse (9 cells collapsed → 1 vector per finger). |
| **points9_arrow** (`_points9_arrow_lora`) | Nine arrows per finger, one per tactile pad, each at its real 3D pad position projected via camera intrinsics. Full spatial-force field. |
| **bin_bar** (`_bin_bar_lora`) | A thin alpha-blended horizontal bar at the bottom edge of each camera image; width = ‖mean force vector‖ × 300 px, capped at half-image. Left finger = blue bar growing right from left edge; right finger = red bar growing left from right edge. Zero spatial information, just magnitude. |

The first three live in the `tactile-data-collection` renderer (`MODES` list in `environment/tactile_overlay.py`). The fourth (`bin_bar`) was ported in from `/home/edwardosunny/sensordrawing` and is now the canonical mode in the renderer (the others have been dropped from `MODES` since this fork).

## Data pipeline

```
phone-teleop session
        │
        │  data collection (tactile-data-collection)
        ▼
teleop_data_<task>.zarr        ← raw: camera + tactile + state + actions
        │
        │  compute_overlay_normalization.py
        │  (writes /meta/normalization/ — per-finger scale + clip bounds)
        ▼
teleop_data_<task>.zarr        ← raw zarr now has normalization stats
        │
        │  scripts/render_overlays.py  (tactile-data-collection)
        │  (reads /meta/normalization, draws overlay onto each frame,
        │   writes data/img_{0,1}_<mode_key> arrays)
        ▼
teleop_data_<task>_overlay.zarr ← raw frames + overlay frames per mode
        │
        │  convert_zarr_to_lerobot.py --mode <mode> --repo-id local/xarm_<task>_<mode>
        ▼
LeRobot dataset                 ← pre-rendered uint8 frames + 8-D state + 7-D action
        │
        │  compute_norm_stats.py --config-name <cfg>
        ▼
assets/<cfg>/local/<repo_id>/norm_stats.json  ← state+action normalization
        │
        │  scripts/train.py <cfg>      (LoRA on pi05_droid)
        ▼
checkpoints/<cfg>/auto_<timestamp>_<jobid>/<step>/
        │ ├── params/                 ← merged base + LoRA weights
        │ ├── assets/local/<repo_id>/ ← norm stats copied in
        │ └── train_state/            ← optimizer state (inference doesn't need)
        ▼
trained policy
```

## State and action representation

The converter (`convert_zarr_to_lerobot.py`) lifts the raw 7-D xArm state into an 8-D LIBERO-compatible state vector and turns absolute TCP poses into per-step deltas, so the model's input/output match the existing `LeRobotLiberoDataConfig` pipeline:

- **State (8 dims)** — `[ee_pos_m(3), ee_axis_angle_rad(3), grasp, grasp]`. The duplicated grasp matches LIBERO's 2-finger qpos convention.
- **Action (7 dims)** — `[Δxyz_m(3), Δaxis_angle_rad(3), grasp ∈ {-1=open, +1=close}]`. Both translation and rotation deltas are in the **world frame**: `new_xyz = xyz + Δxyz`, `new_R = ΔR @ R` (left-multiply).
- `action_horizon = 10` → each `policy.infer()` call returns a `(10, 7)` chunk = 1.0 s of future actions at 10 Hz.

## Model + training config

All 16 configs are identical except for the data repo_id (and therefore the loaded LeRobot dataset). See `src/openpi/training/config.py` — search for `pi05_xarm_<task>_<variant>_lora`.

| Setting | Value |
|---|---|
| Base model | `pi05` — flow-matching VLA (πᵢ-team architecture). Vision encoder: SigLIP-So400m. Language + manipulation backbone: Gemma-2B (PaliGemma init). Action expert: Gemma-300M. |
| Model size (total) | **~2.3 B params** (≈2 B PaliGemma + 311 M action expert) |
| Init | `gs://openpi-assets/checkpoints/pi05_droid/params` (pi0.5 pre-trained on the full DROID corpus with knowledge insulation) |
| Fine-tune type | **LoRA on both backbone and action expert** (`paligemma_variant="gemma_2b_lora"`, `action_expert_variant="gemma_300m_lora"`). LoRA rank 16 (α=16) on the 2B backbone, rank 32 (α=32) on the 300M action expert — applied to both `attn` and `ffn` projections. |
| Frozen | All base weights. Only the LoRA adapters + LoRA-attached scale/bias updates train. |
| Image resolution | **224 × 224 × 3** per camera, 3 cameras per observation (`base_0_rgb` = agent view; `left_wrist_0_rgb` = wrist view; `right_wrist_0_rgb` is zero-padded with `image_mask=False` since xArm has one wrist cam). |
| Action horizon | 10 (each `policy.infer()` returns a `(10, 7)` chunk = 1.0 s @ 10 Hz) |
| `discrete_state_input` | False |
| Batch size | **8** |
| Optimizer | AdamW, `clip_gradient_norm=1.0` |
| LR schedule | Cosine with warmup; `warmup_steps=500`, `peak_lr=1e-4`, `decay_steps=20_000`, `decay_lr=1e-5` |
| EMA | Off (LoRA) |
| Max steps | 20,000 (most runs early-stop well before this) |
| Epochs equivalent | ~13,000–28,000 frames per task at batch=8 → one epoch ≈ 1,600–3,500 steps. The 5,100–7,900-step early-stops are ≈ 2–4 epochs; the full 20k cap is ≈ 6–12 epochs depending on task. |
| `save_interval` | 2,000 steps |
| Early-stop | rolling-1000-step-window mean improves <0.5% relative for 2 consecutive checks past step 3,000 |
| `wandb_enabled` | False (no `WANDB_API_KEY` on the cluster) |

### Data augmentation

Applied per-frame inside the model's `preprocess_observation` (`src/openpi/models/model.py`) only when `train=True`. Each augmentation is sampled fresh per frame per epoch from a JAX PRNG. Different policy for "agent" vs "wrist" cameras so geometric augmentations don't break wrist-camera calibration:

| Camera | Augmentations (in order) |
|---|---|
| **agent view** (`base_0_rgb`) | RandomCrop to 95% × 95% → Resize back to 224 × 224 → Rotate ±5° → ColorJitter (brightness=0.3, contrast=0.4, saturation=0.5) |
| **wrist view** (`left_wrist_0_rgb`) | ColorJitter only (brightness=0.3, contrast=0.4, saturation=0.5) — no crop/rotate so the wrist camera's tight geometric relationship to the gripper is preserved |

All augmentations operate on `[0, 1]`-normalized float; the result is mapped back to `[-1, 1]` before going through SigLIP. At inference, no augmentation is applied — the deployment image goes straight through `resize → normalize → SigLIP`.

### Why LoRA (not full finetune)

- Each task has only 100 episodes (~13–28k frames). Full finetune would overfit before the model adapts; LoRA on a strong pi05_droid prior keeps the language + visual backbones intact and only adapts the manipulation policy.
- Single-GPU LoRA fits easily on one H200; full pi05 finetune wants ≥4 GPUs with FSDP.
- Faster: LoRA hits ~3.0 it/s on H200 at batch=8 (~333 ms/step), vs the much slower full backprop.

## Compute setup (cluster / SLURM)

Trained on a SLURM cluster where compute nodes have per-node-local `/home` and the only NFS-shared directory is `/workspace-vast/`. **Anything an sbatch job needs must be on `/workspace-vast/`** — putting the venv or dataset on `~` makes jobs exit with `command not found` because the compute node sees an empty `/home/edwardosunny/`.

Layout that works:
```
/workspace-vast/edwardosunny/
├── tactile-openpi/            ← repo + .venv (recreated here via uv sync)
│   ├── .venv/                 ← all dependencies, transformers patch applied
│   ├── assets/<cfg>/...       ← norm stats (one per config)
│   ├── checkpoints/<cfg>/...  ← training outputs
│   ├── logs/                  ← slurm-%x-%j.out / .err
│   └── scripts/slurm_train.sbatch
├── openpi-cache/              ← OPENPI_DATA_HOME (paligemma_tokenizer.model + base ckpt)
└── lerobot-cache/local/       ← HF_LEROBOT_HOME (the 16 LeRobot datasets)
```

`scripts/slurm_train.sbatch` is the single submission template:

- `--qos=high`, `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=96G`, `--time=24:00:00`
- `--chdir=/workspace-vast/edwardosunny/tactile-openpi` (absolute)
- Sets `OPENPI_DATA_HOME` and `HF_LEROBOT_HOME` so the tokenizer and dataset resolve without network
- Sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` + `PYTHONUNBUFFERED=1`
- Runs `./.venv/bin/python -u scripts/train.py <config_name> --exp-name=auto_<timestamp>_<jobid> --overwrite`

To submit all variants of one task in parallel:
```bash
cd /workspace-vast/edwardosunny/tactile-openpi
for v in baseline points1_arrow points9_arrow bin_bar; do
  sbatch --job-name="cube_$v" scripts/slurm_train.sbatch "pi05_xarm_cube_${v}_lora"
done
```

## Reproducing one variant end-to-end

```bash
# 1. Render the overlay zarr (writes img_{0,1}_<mode_key> arrays).
cd /home/edwardosunny/tactile-data-collection
PYTHONPATH=$PWD .venv/bin/python scripts/render_overlays.py \
  /home/edwardosunny/data/teleop_data_cube.zarr \
  /home/edwardosunny/data/teleop_data_cube_overlay.zarr

# 2. Convert to LeRobot (one dataset per variant; --mode=raw for the baseline).
cd /workspace-vast/edwardosunny/tactile-openpi
HF_LEROBOT_HOME=/workspace-vast/edwardosunny/lerobot-cache \
  ./.venv/bin/python examples/xarm/convert_zarr_to_lerobot.py \
    --zarr /home/edwardosunny/data/teleop_data_cube_overlay.zarr \
    --mode bin_bar \
    --repo-id local/xarm_cube_bin_bar \
    --language "pick up the cube"

# 3. Compute the state+action norm stats (one per config; CPU is fine).
HF_LEROBOT_HOME=/workspace-vast/edwardosunny/lerobot-cache \
OPENPI_DATA_HOME=/workspace-vast/edwardosunny/openpi-cache \
JAX_PLATFORMS=cpu \
  ./.venv/bin/python scripts/compute_norm_stats.py \
    --config-name pi05_xarm_cube_bin_bar_lora

# 4. Submit a SLURM training job.
sbatch --job-name=cube_bin_bar scripts/slurm_train.sbatch pi05_xarm_cube_bin_bar_lora

# 5. Wait. Each run is 35–50 min wall on one H200; early-stops between step 5,100 and 7,900
#    (or runs to 20,000 if the overlay+prompt combination keeps unlocking gradient).
```

To redo all 16: loop steps 2–4 over the 16 `(task, variant)` pairs. Convert + norm-stats can run 4-way (or 12-way) in parallel.

## Inference deployment

The trained checkpoint is self-contained for inference. Each `<step>/` directory contains:
- `params/` — full merged model weights (LoRA fused into base)
- `assets/local/<repo_id>/norm_stats.json` — state+action normalization
- `_CHECKPOINT_METADATA`

The only thing **not** inside the checkpoint is the PaligemmaTokenizer (`gs://big_vision/paligemma_tokenizer.model`, ~4 MB) — it's auto-downloaded on first use into `$OPENPI_DATA_HOME` (or `~/.cache/openpi/`).

For overlay variants, inference must render the **same** overlay onto the live camera frame before sending it to the policy. Otherwise the deployment-time pixel distribution diverges from training and the policy goes out-of-distribution. The runtime renderer is `examples/xarm/inference/inference_overlay.py` which delegates to the same `SensorOverlay` class used at training, so the rendering is byte-for-byte identical (as long as the source-of-truth `tactile-data-collection` renderer is the same on the inference host).

A pooled overlay-norm npz ships in `examples/xarm/inference/overlay_norm_<mode>.npz` (3 KB each). Pooled across all four task overlay zarrs because `scale_xy`, `scale_z`, and `deadband` are identical across tasks (compute_overlay_normalization was run jointly), and `raw_clip_low/high` use the elementwise min/max — so one file works for all four tasks per mode.

Run inference:
```bash
./.venv/bin/python examples/xarm/inference/run_xarm_inference.py \
  --checkpoint /workspace-vast/.../pi05_xarm_charger_bin_bar_lora/auto_*/5100 \
  --train-config-name pi05_xarm_charger_bin_bar_lora \
  --overlay bin_bar \
  --prompt "pick up the charger and plug it into the nearest plug" \
  --xarm-ip 192.168.1.223 \
  --max-steps 400 --replan-steps 10 --action-scale 0.4 \
  --servo-speed 60 --servo-mvacc 400 --safety-threshold 200
```

`--overlay bin_bar` resolves to `examples/xarm/inference/overlay_norm_bin_bar.npz` and uses `bin_bar` as the renderer mode_key. For `_baseline_lora` checkpoints (raw frames), **omit** the `--overlay` flag entirely.

## Results so far

All 16 finetunes use the same hyperparameters and the same early-stop criterion, so the training-loss numbers below are directly comparable. Lower = better.

| Task | Baseline | points1_arrow | points9_arrow | bin_bar |
|---|---|---|---|---|
| **cube** | 7900 / 0.0437 | 7900 / 0.0427 | 7900 / 0.0425 | 7900 / 0.0438 |
| **tube** | 5800 / 0.0411 | 5800 / 0.0411 | **18600 / 0.0275** | 5800 / 0.0414 |
| **charger** | 5700 / 0.0413 | 5100 / 0.0402 | 5300 / 0.0400 | 5100 / 0.0399 |
| **dishwasher** | 7800 / 0.0290 | 7800 / 0.0287 | **19900 / 0.0240** | 7800 / 0.0291 |

(format: `early_stop_step / final_loss`)

**The clean result**: `points9_arrow` is the only variant where the loss curve *kept descending past the plateau where everything else early-stops*, and only for the two tasks with long multi-step prompts (`tube`, `dishwasher`). Both ran the full 20k-step budget and converged to 17–29% lower loss than the baseline / points1 / bin_bar runs. The other two tasks (cube, charger) are within ±1% across all four variants — the overlay does not measurably help when the prompt is short or the task is simple.

Interpretation: spatial-force information (9 arrows at 9 pad positions) carries genuine signal that a richer prompt can lean on. Collapsing the 9-pad force field into either one arrow (`points1_arrow`) or one scalar bar (`bin_bar`) throws that signal away. Training-loss-only — real-robot evaluation pending.

## Evaluation protocol

Three layers, currently used in this order:

1. **Training-loss tracking** — every config logs scalar mean flow-matching loss + grad-norm + param-norm every 100 steps. The matrix in the previous section is the head-to-head comparison. Direct comparability comes from identical hyperparameters, identical optimizer, identical early-stop rule across all 16 runs — the only thing varied is the overlay pixels.

2. **Offline policy probe** (`examples/xarm/inference/probe_offline.py`) — for every trained checkpoint we load the policy, hand it the documented xArm home-pose observation (TCP at `(475.79 mm, -1.14 mm, 244.72 mm, 179.13°, -0.01°, 0.78°)`) with synthetic gray + noise images, and run one or more forward passes. Validates the checkpoint loads cleanly, action shapes are correct `(10, 7)`, magnitudes are physically plausible (typically per-step |Δxyz| ≤ 1.25 cm, grasp in `[-1, +1]`), and no NaN. All 16 trained checkpoints passed this in their respective phases — see the memory notes for details (e.g. `charger_bin_bar` learned a t=6 grasp-release transition matching the "plug it in and let go" motion). This is a **shape + sanity** check, not a behavior validation.

3. **Real-robot rollout** (`examples/xarm/inference/run_xarm_inference.py`) — closed-loop deployment on the live xArm at 10 Hz with re-planning every 10 steps. Renders the overlay onto live RealSense frames using the same `SensorOverlay` code as training (no distribution shift). Includes a per-tick tactile safety clamp on gripper-close commands (`--safety-threshold` on the delta-from-idle `sum|Bz|` metric). **This is the metric that actually counts.** As of the latest snapshot, real-robot evaluation is in progress and not yet reported here — the training-loss matrix above is suggestive but not conclusive of behavioral improvement.

Things explicitly NOT in the protocol yet:
- Held-out validation split on the existing 100 episodes (a small one is straightforward to add).
- Quantitative success-rate scoring over fixed rollout count.
- Cross-checkpoint A/B on the same scene reset.

## Reading more

- Per-phase result notes (paths, losses, observations) — see the `MEMORY.md` index under `~/.claude/projects/-home-edwardosunny-tactile-openpi/memory/`. Phase 1 is the original points1_arrow run, Phase 2 added points9 + baseline, Phase 3 corrected the prompts, Phase 4 added bin_bar.
- Inference-side details — `examples/xarm/inference/README.md`.
- Renderer internals — `tactile-data-collection/environment/tactile_overlay.py` and the vendored `sensordrawing/` package.
