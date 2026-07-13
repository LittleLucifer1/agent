# DistillWheel

DistillWheel is a small, framework-neutral training orchestrator. Recipes and
JSONL samples are validated in the main process; ms-swift and VERL training run
in isolated subprocess environments.

## Install

```bash
python -m pip install -e ".[dev,verl]"
```

The `verl` extra installs `pyarrow` in the orchestrator environment so it can
write prompt parquet files. Backend training dependencies are installed into
separate Python 3.10 environments:

```bash
distillwheel-setup-backends swift
distillwheel-setup-backends verl
```

Each backend manifest requires Python 3.10. If it is not discoverable as
`python3.10`, pass it explicitly:

```bash
distillwheel-setup-backends swift --python /usr/bin/python3.10
```

Managed environments default to `./.venvs/<backend>`. Set
`DISTILLWHEEL_ENV_ROOT` before both setup and training to use another absolute
root. To use an existing Conda environment without modifying it, point directly
at its interpreter:

```bash
export DISTILLWHEEL_SWIFT_PYTHON=/opt/conda/envs/swift/bin/python
export DISTILLWHEEL_VERL_PYTHON=/opt/conda/envs/verl/bin/python
distillwheel-setup-backends swift   # validates only
```

On Windows, use the full `python.exe` path. Package installation into an
external interpreter requires the explicit `--install-into-external` flag.

VERL has a large, CUDA-sensitive dependency matrix. For production RL runs,
prefer the VERL project's official Docker image or reproduce its documented
CUDA/PyTorch versions instead of mixing arbitrary latest packages.

## Run

```bash
distillwheel run --recipe configs/sft.yaml --data data/sft.jsonl
distillwheel run --recipe configs/dpo.yaml --data data/dpo.jsonl
distillwheel run --recipe configs/grpo.yaml --data data/prompts.jsonl
```

Supported stages are `sft`, `dpo`, `kto`, `grpo`, `ppo`, and `rloo`. OPD is not
advertised until the Recipe schema can represent its teacher/distillation
inputs.

An output directory represents exactly one run and must be empty. Reusing it is
rejected by default. `--overwrite-output` removes only known managed artifacts,
and only when the directory contains a DistillWheel ownership marker; unrelated
files are preserved.

The default preflight is configuration-only. It verifies Recipe-to-backend
translation but does not load a model or detect runtime CUDA/OOM failures.
