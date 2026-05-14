# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

openpi is Physical Intelligence's open-source robotics repo. It ships three vision-language-action (VLA) model families with both JAX (flax/nnx) and PyTorch implementations:

- **π₀** — flow-matching VLA
- **π₀-FAST** — autoregressive VLA using the FAST action tokenizer (JAX only)
- **π₀.₅** — upgraded π₀ with knowledge insulation (flow-matching head only in this repo)

Base checkpoints live in `gs://openpi-assets/checkpoints/...` and are auto-downloaded to `~/.cache/openpi` (override with `OPENPI_DATA_HOME`).

## Environment setup

Python **3.11** is required. Use `uv` (not pip) and initialize submodules:

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` is required because LeRobot is pulled as a git dependency. `pyproject.toml` overrides `ml-dtypes` and `tensorstore` via `[tool.uv]` — do not change these without checking JAX/orbax compatibility.

For RLDS / DROID workflows, sync the optional group: `uv sync --group rlds` (TensorFlow-cpu 2.15 only ships cp311 wheels).

### PyTorch-specific setup

PyTorch support requires patching the installed `transformers` library (AdaRMS, precision control, KV-cache-without-update):

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

With uv's default hardlink mode this **mutates the uv cache** — undo with `uv cache clean transformers`. PyTorch path does **not** support: π₀-FAST, mixed-precision, FSDP, LoRA, EMA.

## Common commands

```bash
# Compute norm stats (required before training)
uv run scripts/compute_norm_stats.py --config-name <config>

# JAX training (set MEM_FRACTION to avoid OOM)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<name> [--overwrite|--resume]

# PyTorch training — single GPU
uv run scripts/train_pytorch.py <config> --exp_name <name>

# PyTorch training — multi-GPU (single node)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<N> scripts/train_pytorch.py <config> --exp_name <name>

# Policy server (works with JAX or PyTorch checkpoints; same flags)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<ckpt_dir>

# Convert JAX → PyTorch checkpoint
uv run examples/convert_jax_model_to_pytorch.py --config_name <config> --checkpoint_dir <jax> --output_path <pt>
```

Use `--fsdp-devices N` (JAX) to shard across GPUs when OOM. Multi-node training is not supported by `scripts/train.py`; the PyTorch script supports it via `torchrun` env vars.

## Tests, lint, pre-commit

CI runs `uv run pytest --strict-markers -m "not manual"`. `testpaths` in `pyproject.toml` are `src`, `scripts`, `packages`. The `manual` marker is for tests that must be run by hand — don't unmark them. Run a single test with `uv run pytest path/to/test_file.py::TestName -k name`.

Lint/format: `ruff check .` and `ruff format .` (line length 120, target `py311`, single-line imports, force-sort within sections). `third_party/` and `src/openpi/models_pytorch/transformers_replace/` are excluded from ruff. Install hooks with `pre-commit install`; they run `uv-lock` + `ruff --fix` + `ruff-format`.

## Architecture

### Where named training configs live

All training entrypoints (`train.py`, `train_pytorch.py`, `compute_norm_stats.py`, `serve_policy.py`) accept a **config name** like `pi05_libero`. The full list is the `_CONFIGS` array at the bottom of `src/openpi/training/config.py`; lookup goes through `get_config(name)`. To add a robot/task, append a new `TrainConfig` there — don't invent a new lookup path.

`TrainConfig` composes:
- `model` — `BaseModelConfig` subclass (`Pi0Config`, `Pi0FASTConfig`)
- `data` — a `DataConfigFactory` (`LeRobotAlohaDataConfig`, `LeRobotLiberoDataConfig`, `RLDSDroidDataConfig`, `SimpleDataConfig`, `FakeDataConfig`)
- `weight_loader` (JAX) or `pytorch_weight_path` (PyTorch)
- optimizer / lr schedule / EMA / FSDP / freeze filter

### Data pipeline

Per-step transformation order is fixed and important to understand when adding a robot:

1. Raw batch from LeRobot/RLDS dataset
2. `repack_transforms` — rename keys to common schema
3. `data_transforms` — robot-specific `*Inputs`/`*Outputs` in `src/openpi/policies/{aloha,droid,libero}_policy.py`; turns robot-native fields into the model's `Observation` dict
4. `Normalize` (or quantile-normalize) using stats from `compute_norm_stats.py`
5. `model_transforms` — usually `TokenizePrompt` / `TokenizeFASTInputs` / `ResizeImages`
6. Model

Inference reverses 4→3 via `Unnormalize` and the robot's `*Outputs` transform. The canonical model-input shape is documented at the top of `src/openpi/models/model.py` (`IMAGE_KEYS`, `IMAGE_RESOLUTION`, the `Observation`/`Actions` dataclasses).

### Modules

- `src/openpi/models/` — JAX/flax-nnx model code. `pi0.py`, `pi0_fast.py`, Gemma backbone (`gemma.py`, `gemma_fast.py`), SigLIP / ViT vision, tokenizer, LoRA.
- `src/openpi/models_pytorch/` — PyTorch ports. Note `transformers_replace/` is the patch directory that must be copied into the installed `transformers` package.
- `src/openpi/policies/` — robot-specific transforms + `Policy` runtime (`policy.py`) and `create_trained_policy(...)` factory (`policy_config.py`). Auto-detects JAX vs PyTorch checkpoint format.
- `src/openpi/training/` — `config.py` (configs registry), `data_loader.py`, `droid_rlds_dataset.py`, `checkpoints.py`, `optimizer.py`, `sharding.py` (FSDP mesh), `weight_loaders.py`.
- `src/openpi/serving/websocket_policy_server.py` — server side of the remote-inference protocol.
- `src/openpi/shared/` — `download.py` (GCS asset fetcher), `normalize.py`, `image_tools.py`, `nnx_utils.py`, `array_typing.py`.
- `packages/openpi-client/` — lightweight uv workspace member (Python ≥3.7, no JAX/torch deps) used by robot runtimes to talk to the policy server over websocket+msgpack. Keep it dependency-light when editing.
- `scripts/train.py` (JAX, flax-nnx, optax, FSDP via `sharding.py`) and `scripts/train_pytorch.py` (DDP via torchrun) are independent code paths; changes to training behaviour usually need to land in both.

### Remote inference

The policy server pattern decouples robot env (often Python 3.10+ROS+old numpy) from the model env. Client uses `openpi_client.websocket_client_policy`. The server is `scripts/serve_policy.py`. See `docs/remote_inference.md`.

## Conventions

- Configs are frozen `@dataclasses.dataclass(frozen=True)` — never mutate; create a new one with `dataclasses.replace`.
- New robot? Add `*Inputs`/`*Outputs` in `src/openpi/policies/`, a `DataConfigFactory` and a `TrainConfig` entry — not a new top-level module.
- Print statements are allowed (`T201` ignored); `F722` is ignored because of array-typing annotations.
- JAX precision: bfloat16 weights/compute, float32 for select stable ops; toggle full-fp32 training with `dtype` in model config. PyTorch: full bf16 (default) or fp32 via `pytorch_training_precision` — no mixed precision.
