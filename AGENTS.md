# AGENTS.md

## Project Overview

Agricultural drone RL environment (Gymnasium API) for training a hexacopter agent on farming tasks: crop mapping, disease spraying, irrigation, obstacle avoidance, and return-to-base. Built with PyBullet for 3D rendering.

## Key Commands

```bash
# Run demos (requires display for PyBullet GUI)
uv run environment/demo_env.py        # Full agri environment with field grid
uv run environment/minimal_fly_env.py  # Minimal fly-only env (no agri logic)

# Tests (pytest is installed but no pytest-compatible tests exist yet)
# environment/test_dynamics.py is a manual script, not a pytest test
cd environment && python test_dynamics.py

# Install dependencies
uv pip install -r requirements.txt
```

## Architecture

```
DRL2/
├── environment/               # Main package
│   ├── agri_drone_env.py      # AgriDroneEnv (gym.Env) — primary env
│   ├── minimal_fly_env.py     # MinimalFlyEnv — stripped-down fly-only env
│   ├── demo_env.py            # Demo runner for AgriDroneEnv
│   ├── obstacles.py           # ObstacleManager (spherical obstacles)
│   ├── pybullet_renderer.py   # 3D rendering (PyBullet, decoupled from physics)
│   ├── physics/
│   │   ├── drone_dynamics.py  # DroneDynamics + DroneParams + DroneState
│   │   └── wind_model.py      # WindModel (gusts, domain randomization)
│   ├── reward/
│   │   ├── reward_function.py # RewardCalculator (nav + agri rewards)
│   │   └── physics/           # ⚠️ DUPLICATE of physics/ — has DEBUG=True, do not use
│   └── utils/
│       └── normalization.py   # normalize() / denormalize() to [-1, 1]
├── agent/
│   └── model.py               # Empty — agent model not yet implemented
├── train.py                   # Empty — training pipeline not yet implemented
├── requirements.txt           # Pinned deps (gymnasium, pybullet, stable-baselines3, torch, wandb)
├── specify.md                 # Empty
└── update.md                  # Empty
```

## Critical Gotchas

1. **No package manager or build system** — no `pyproject.toml`, no `setup.py`. Imports use `sys.path.insert(0, ...)` hacks in demo/test scripts. The `environment/` directory is not an installable package.

2. **Duplicate physics code** — `environment/reward/physics/` contains copies of `drone_dynamics.py` and `wind_model.py` with `DEBUG=True` enabled. The canonical versions are in `environment/physics/`. Never import from `reward/physics/`.

3. **No pytest-compatible tests** — `environment/test_dynamics.py` is a manual script with `time.sleep()` calls. No test suite exists. Run it manually: `cd environment && python test_dynamics.py`.

4. **Empty implementation files** — `train.py` and `agent/model.py` are both empty. Training pipeline and agent model are not yet built.

5. **`rl-template==0.1.1`** is in requirements but unused — this is the intended framework for the training pipeline.

6. **README vs code mismatch** — `environment/README.md` describes 24-dim observation space and a `BaseEnv` parent class; actual code uses 17-dim obs and `gym.Env` directly. Trust the code.

7. **Documentation language** — all comments, docstrings, and README content are written in French (per project convention set in `.opencode/agents/subagent/senior.md`).

## Agent Workflow

This repo uses OpenCode multi-agent setup (`.opencode/agents/`):

- **reviewer** — plans tasks, reviews code quality, delegates to others (read-only)
- **researcher** — implements env code and training pipeline (read/write/bash)
- **senior** — writes Google-style docstrings and README in French (read/edit only)
- **test-researcher** — writes and runs pytest tests (read/write/bash)

Workflow: reviewer plans → researcher implements → test-researcher validates → senior documents.

## Testing

Tests must be written as pytest-compatible (no `time.sleep()`, no manual scripts). Run with:
```bash
cd /home/darrius/project/DRL2 && python -m pytest environment/ -v
```
Currently no pytest tests exist. New tests should cover: DroneDynamics, WindModel, ObstacleManager, RewardCalculator, AgriDroneEnv step/reset, normalization utils.

## Python Version

Python 3.12 (from `.python-version`).
