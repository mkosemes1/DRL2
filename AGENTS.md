# AGENTS.md

## Project Overview

Agricultural drone RL environment (Gymnasium API) for training a hexacopter agent on farming tasks: crop mapping, disease spraying, irrigation, obstacle avoidance, and return-to-base. Built with PyBullet for 3D rendering.

## Key Commands

```bash
# Run demos (requires display for PyBullet GUI)
uv run environment/demo_env.py        # Full agri environment with field grid
uv run environment/minimal_fly_env.py  # Minimal fly-only env (no agri logic)

# Training and evaluation
uv run python run_train.py                             # Entraînement PPO
uv run python run_train.py --episodes 50 --wandb       # Entraînement avec wandb
uv run python eval_agent.py                            # Évaluation + GIF
uv run python eval_agent.py --model saved_models/agri_drone_ppo.pt --output demo.gif

# Tests (323 pytest-compatible tests)
uv run python -m pytest environment/tests/ -v                                   # All tests
uv run python -m pytest environment/tests/test_reward_function.py -v            # Reward tests
uv run python -m pytest environment/tests/test_agri_drone_env.py -v             # Env tests
uv run python -m pytest environment/tests/test_drone_dynamics.py -v             # Dynamics tests
uv run python -m pytest environment/tests/test_integration.py -v                # Integration tests
uv run python -m pytest environment/tests/test_edge_cases.py -v                 # Edge case tests

# Legacy manual test (not pytest-compatible)
cd environment && python test_dynamics.py

# Documentation
uv run python -c "import pathlib; print(pathlib.Path('docs/DOCUMENTATION_DETAILED.md').read_text()[:200])"  # Aperçu doc détaillée
# Ou ouvrir directement : docs/DOCUMENTATION_DETAILED.md

# Install dependencies
uv pip install -r requirements.txt
```

## Architecture

```
DRL2/
├── environment/               # Main package
│   ├── agri_drone_env.py      # AgriDroneEnv (BaseEnv) — primary env with water task
│   ├── minimal_fly_env.py     # MinimalFlyEnv — stripped-down fly-only env
│   ├── demo_env.py            # Demo runner for AgriDroneEnv
│   ├── obstacles.py           # ObstacleManager (spherical obstacles)
│   ├── pybullet_renderer.py   # 3D rendering (PyBullet, decoupled from physics)
│   ├── physics/
│   │   ├── drone_dynamics.py  # DroneDynamics + DroneParams + DroneState
│   │   └── wind_model.py      # WindModel (gusts, domain randomization)
│   ├── reward/
│   │   ├── reward_function.py # RewardCalculator (nav + agri + water task rewards)
│   │   └── physics/           # ⚠️ DUPLICATE of physics/ — has DEBUG=True, do not use
│   ├── tests/                 # 323 pytest-compatible tests
│   │   ├── test_drone_dynamics.py
│   │   ├── test_wind_model.py
│   │   ├── test_obstacles.py
│   │   ├── test_normalization.py
│   │   ├── test_reward_function.py
│   │   ├── test_agri_drone_env.py
│   │   ├── test_field_cell.py
│   │   ├── test_minimal_fly_env.py
│   │   ├── test_demo_config.py
│   │   ├── test_integration.py
│   │   ├── test_edge_cases.py
│   │   ├── test_base_env_interface.py
│   │   ├── test_agent_model.py
│   │   ├── test_training_pipeline.py
│   │   └── test_trainer_integration.py  # Tests d'intégration end-to-end Trainer
│   └── utils/
│       └── normalization.py   # normalize() / denormalize() to [-1, 1]
├── agent/
│   └── model.py               # Agent PPO (BaseAgent) — acteur-critique MLP
├── train.py                   # Trainer PPO (BaseTrain) — pipeline d'entraînement avec tqdm/wandb
├── eval_agent.py             # Évaluation agent sauvegardé + génération GIF
├── run_train.py              # Script d'entraînement PPO (CLI)
├── requirements.txt           # Pinned deps (gymnasium, pybullet, stable-baselines3, torch, wandb)
├── specify.md                 # MDP specification (observation space, water task, rewards)
├── docs/
│   ├── DOCUMENTATION_DETAILED.md  # Documentation détaillée (1531 lignes) — env, reward, glossaire
│   └── FORMATION.md               # Guide de formation
├── README.md                  # Project documentation (French)
└── update.md                  # Session changelog
```

## Observation Space

The observation space has `17 + 3 + 1 + N*4` dimensions (default N=5 → 39 dims):

| Dims | Description |
|------|-------------|
| 0–16 | Drone state (position, velocity, attitude, angular rates, goal distance, heading error, battery, obstacle dist, wind) |
| 17–19 | Water basin coordinates (x, y, z) |
| 20 | Water tank level (0–100, normalized to [-1, 1]) |
| 21–20+N*4 | Plant groups matrix (x, y, z, is_watered) per group, is_watered normalized to [-1, 1] |

## Action Space

6 continuous dimensions in [-1, 1]:

| Dim | Action |
|-----|--------|
| 0 | Throttle |
| 1 | Roll |
| 2 | Pitch |
| 3 | Yaw |
| 4 | Spray (>0 = activate) |
| 5 | Irrigate (>0 = activate) |

## Water Task Mechanics

- **Watering**: When drone is within `watering_proximity` (default 2.0m) of an unwatered plant group and `action[5] > 0` and tank >= 2.0, the group is marked watered and tank decreases by `water_consumption` (default 2.0).
- **Refilling**: When drone is within `basin_refill_radius` (default 3.0m) of the water basin, tank refills to 100.0. Reward +1.0 only if tank was below 98.0.
- **Mission complete**: When all plant groups are watered, episode terminates with +100.0 bonus.

## Reward Structure (water task)

| Term | Value | Trigger |
|------|-------|---------|
| watering | +5.0 | A plant group transitions to watered |
| refill | +1.0 | Tank refilled at basin (only if tank < 98.0) |
| time_penalty | -0.02 × num_unwatered | Every step |
| distance_shaping | +0.05 | Drone moved closer to nearest unwatered group |
| mission_complete | +100.0 | All groups watered (early stop) |

## Critical Gotchas

1. **No package manager or build system** — no `pyproject.toml`, no `setup.py`. Imports use `sys.path.insert(0, ...)` hacks in demo/test scripts. The `environment/` directory is not an installable package.

2. **Duplicate physics code** — `environment/reward/physics/` contains copies of `drone_dynamics.py` and `wind_model.py` with `DEBUG=True` enabled. The canonical versions are in `environment/physics/`. Never import from `reward/physics/`.

3. **BaseEnv integration** — `AgriDroneEnv` inherits from `rl_template.env.BaseEnv` (not directly from `gym.Env`). The `step()` method accepts both numpy arrays and torch tensors (converted via `np.asarray`). The `rollout_phase()` in `BaseTrain` passes torch tensors to `step()`.

4. **Continuous action PPO** — `BaseTrain.rollout_phase()` uses `.item()` on actions which breaks for continuous (6D) actions. The custom `Trainer.rollout_phase()` in `train.py` overrides this to handle numpy arrays correctly. `PPOTrainer.update()` from rl_template also has shape mismatches with continuous actions — `Trainer.update_weights()` implements the PPO loop directly with proper log-prob summing over action dimensions.

5. **`rl-template==0.1.1`** is in requirements but unused — this is the intended framework for the training pipeline.

6. **Documentation language** — all comments, docstrings, and README content are written in French (per project convention set in `.opencode/agents/subagent/senior.md`).

7. **Use `uv run`** for all Python commands — the project uses a `.venv` with Python 3.12 managed by `uv`. Direct `python` calls may use system Python and miss dependencies.

8. **rgb_array rendering** — `AgriDroneEnv` supports `render_mode="rgb_array"` for headless frame capture via PyBullet `p.DIRECT`. The headless client is a **separate** PyBullet physics client from the GUI client. All PyBullet API calls in `_render_rgb_array` use `physicsClientId=self._client` to avoid conflicts. Frames are captured at 640×480 via `getCameraImage()`.

## Training Pipeline (Session 5)

`Trainer` in `train.py` properly extends `BaseTrain` from `rl_template`:

- **Overrides only 2 methods**:
  - `rollout_phase(state)`: Fixes `.item()` crash for 6D actions → uses `.cpu().numpy()`. Sums log-probs over action dims before `buffer.insert()` for scalar storage.
  - `update_weights(step)`: Implements PPO loop directly with log-prob summing (`new_log_probs.sum(dim=-1)` vs `old_log_probs`) to avoid shape mismatch.
- **Reuses**: `save_model()` from `BaseTrain` (no override needed).
- **Adds**: `tqdm` progress bars, `wandb` logging (optional), `_init_wandb()`, `_log_wandb()`.
- **Buffer**: Created with `action_shape=(act_dim,)` for continuous 6D actions.

Usage:
```python
from train import Trainer
from rl_template.config import PPOConfig, TrainConfig

trainer = Trainer(env=env, agent=agent, train_config=train_cfg, ppo_config=ppo_cfg)
trainer.train(verbose=True)  # tqdm + wandb
```

Key constants:
- `rollout_steps`: Number of steps per rollout (default 128)
- `batch_size`: Mini-batch size (default 64)
- `ppo_config`: lr, gamma, gae_lambda, clip_eps
- `wandb_config`: Optional dict with project, entity, name, config

## Évaluation

Le script `eval_agent.py` charge un modèle sauvegardé et génère un GIF de l'agent en action.

```bash
uv run python eval_agent.py                                    # Dernier modèle
uv run python eval_agent.py --model saved_models/agent.pt     # Modèle spécifique
uv run python eval_agent.py --episodes 3 --fps 20             # Plusieurs épisodes
```

Modes de rendu :
- `render_mode="rgb_array"` : capture headless via PyBullet DIRECT (pour GIF/video)
- `render_mode="human"` : fenêtre PyBullet GUI temps réel
- `render_mode=None` : pas de rendu (entraînement)

Le GIF est créé avec PIL (pillow) à partir des frames RGB capturées.

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
cd /home/darrius/project/DRL2 && uv run python -m pytest environment/tests/ -v
```
323 tests exist covering: DroneDynamics, WindModel, ObstacleManager, RewardCalculator (including compute_water_task), AgriDroneEnv step/reset/observation, normalization utils, FieldCell, MinimalFlyEnv, demo config, integration flows, edge cases, BaseEnv interface compliance, Agent PPO model, training pipeline (Trainer rollout/update/save/wandb), **Trainer integration (instantiation, full training loop, loss evolution, model saving, buffer/PPO, metrics, edge cases)**, and buffer/PPO integration.

## Python Version

Python 3.12 (from `.python-version`).

## Documentation

Une documentation détaillée du projet est disponible dans `docs/DOCUMENTATION_DETAILED.md` (1531 lignes). Elle couvre :

- Vue d'ensemble du projet et architecture
- Détail de l'environnement `AgriDroneEnv` (init, step, reset, obs, render)
- Physique du drone (`DroneDynamics`, `DroneParams`, `DroneState`)
- Fonction de récompense (`compute`, `compute_agri`, `compute_water_task`) avec exemples numériques
- Gestion des obstacles, normalisation, pipeline d'entraînement
- Diagrammes ASCII art pour les flux de données
- Glossaire technique (30+ termes)

Le `README.md` (669 lignes) contient également un guide exhaustif incluant les hyperparamètres PPO, la configuration de l'environnement, et un guide pratique pour les ajustements.
