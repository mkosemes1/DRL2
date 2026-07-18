# Guide de formation — Programme d'entraînement RL du drone agricole

Ce document explique en détail le programme d'entraînement par renforcement
pour le drone agricole, de l'initialisation à l'évaluation.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du système](#architecture-du-système)
3. [Installation et configuration](#installation-et-configuration)
4. [Lancement de l'entraînement](#lancement-de-lentraînement)
5. [Hyperparamètres](#hyperparamètres)
6. [Suivi de l'entraînement](#suivi-de-lentraînement)
7. [Évaluation et visualisation](#évaluation-et-visualisation)
8. [Guide des hyperparamètres](#guide-des-hyperparamètres)
9. [Résolution de problèmes](#résolution-de-problèmes)
10. [Références](#références)

---

## Vue d'ensemble

Le programme d'entraînement utilise l'algorithme **PPO (Proximal Policy Optimization)** pour entraîner un agent neuronal à accomplir des missions agricoles avec un drone hexacoptère.

### Objectifs de l'agent

L'agent doit apprendre à :
1. **Voler** de manière stable dans un espace 3D
2. **Naviguer** vers des groupes de plantes
3. **Arroser** les plantes en gérant le réservoir d'eau
4. **Se rendre** à la bassine de ravitaillement quand le réservoir est vide
5. **Accomplir** la mission (arroser tous les groupes) le plus rapidement possible

### Boucle d'entraînement

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Environnement│────▶│    Agent      │────▶│    Buffer     │
│  (PyBullet)  │◀────│  (PPO Actor)  │◀────│  (Rollout)    │
└─────────────┘     └──────────────┘     └───────────────┘
       │                    │                     │
       │                    ▼                     ▼
       │            ┌──────────────┐     ┌───────────────┐
       │            │  PPO Update  │◀────│  GAE Compute  │
       │            │  (Policy+V)  │     │  (Advantages) │
       │            └──────────────┘     └───────────────┘
       │                    │
       ▼                    ▼
┌─────────────┐     ┌──────────────┐
│  Observations│     │   Métriques  │
│  (33 dims)  │     │  (wandb/tqdm)│
└─────────────┘     └──────────────┘
```

---

## Architecture du système

### Composants principaux

| Composant | Fichier | Rôle |
|-----------|---------|------|
| Environnement | `environment/agri_drone_env.py` | Simulation du drone et de la tâche d'irrigation |
| Agent | `agent/model.py` | Réseau de neurones acteur-critique (MLP) |
| Entraîneur | `train.py` | Boucle PPO avec rollout, GAE, update |
| Buffer | `rl_template.common.Buffer` | Stockage des transitions pendant le rollout |
| PPO | `rl_template.algorithms.ppo.PPOTrainer` | Calcul du GAE et des pertes PPO |

### Architecture de l'agent

```
Input (33 dims)
    │
    ▼
Linear(33 → 64) + ReLU
    │
    ▼
Linear(64 → 64) + ReLU
    │
    ├──▶ Policy Head (Linear(64 → 6))  ──▶  Mean + LogStd  ──▶  Normal Distribution
    │                                                    ──▶  Sample Action
    │
    └──▶ Value Head (Linear(64 → 1))   ──▶  State Value V(s)
```

- **33 dimensions d'entrée** : 17 (état drone) + 3 (bassine) + 1 (réservoir) + 12 (3 groupes × 4)
- **6 dimensions de sortie** : throttle, roll, pitch, yaw, spray, irrigate
- **Distribution** : Normale paramétrique (mean appris, log_std appris)
- **Initialisation** : orthogonale (gain=√2 pour couches cachées, 0.01 pour acteur, 1.0 pour critique)

---

## Installation et configuration

### Prérequis

```bash
# Python 3.12
python --version  # Doit afficher Python 3.12.x

# Installation des dépendances
uv pip install -r requirements.txt
```

### Dépendances clés

| Package | Version | Usage |
|---------|---------|-------|
| `torch` | ≥ 2.0 | Réseaux de neurones et calcul GPU |
| `gymnasium` | ≥ 1.0 | API RL standard |
| `pybullet` | ≥ 3.2 | Simulation physique et rendu 3D |
| `rl-template` | 0.1.2 | Framework RL (BaseEnv, BaseAgent, BaseTrain) |
| `wandb` | ≥ 0.15 | Suivi des expériences (optionnel) |
| `tqdm` | ≥ 4.0 | Barres de progression |
| `pillow` | ≥ 10.0 | Création de GIFs pour l'évaluation |

---

## Lancement de l'entraînement

### Méthode 1 : Script CLI (recommandé)

```bash
# Entraînement rapide (défaut : 100 updates × 128 steps = 12,800 timesteps)
uv run python run_train.py

# Plus d'entraînement
uv run python run_train.py --episodes 200

# Avec logging wandb
uv run python run_train.py --episodes 100 --wandb --wandb-project agri-drone-rl

# Hyperparamètres personnalisés
uv run python run_train.py --lr 1e-3 --gamma 0.99 --rollout-steps 256 --batch-size 128
```

### Méthode 2 : Code Python

```python
import sys
sys.path.insert(0, "environment")
sys.path.insert(0, "agent")

from agri_drone_env import AgriDroneEnv
from model import Agent
from train import Trainer
from rl_template.config import PPOConfig, TrainConfig

# 1. Créer l'environnement
config = {
    "world": {"size_x": 60.0, "size_y": 60.0, "ground_z": 0.0, "size_z": 50.0},
    "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
    "simulation": {"dt": 0.02, "max_episode_steps": 1000},
    "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
    "water_task": {
        "basin_position": [15.0, 15.0, 0.5],
        "basin_refill_radius": 3.0,
        "water_consumption": 2.0,
        "watering_proximity": 2.0,
        "num_plant_groups": 5,
    },
}
env = AgriDroneEnv(config)

# 2. Créer l'agent
agent = Agent(
    n_state=env.observation_space.shape[0],  # 33
    n_action=env.action_space.shape[0],      # 6
)

# 3. Configurer l'entraînement
train_config = TrainConfig(
    model_name="agri_drone_ppo",
    model_saved_path="saved_models",
    timestamp=100_000,     # 100k timesteps
    batch_size=64,
    rollout_steps=128,
)

ppo_config = PPOConfig(
    lr=3e-4,              # Learning rate
    gamma=0.99,           # Discount factor
    gae_lambda=0.95,      # GAE lambda
    clip_eps=0.2,         # PPO clipping
    ent_coef=0.01,        # Entropy bonus
    value_coef=0.5,       # Value loss coefficient
)

# 4. Lancer l'entraînement
trainer = Trainer(
    env=env,
    agent=agent,
    train_config=train_config,
    ppo_config=ppo_config,
)
trainer.train(verbose=True)

env.close()
```

---

## Hyperparamètres

### Paramètres PPO

| Paramètre | Défaut | Description | Recommandation |
|-----------|--------|-------------|----------------|
| `lr` | 3e-4 | Taux d'apprentissage | 1e-4 à 3e-3 selon la taille du réseau |
| `gamma` | 0.99 | Facteur de discount | 0.99 pour tâches longues, 0.95 pour courtes |
| `gae_lambda` | 0.95 | Lambda pour le GAE | 0.95 est un bon compromis biais-variance |
| `clip_eps` | 0.2 | Plage de clipping PPO | 0.1 à 0.3 selon la stabilité |
| `ent_coef` | 0.01 | Coefficient d'entropie | Augmenter si l'agent est trop déterministe |
| `value_coef` | 0.5 | Coefficient de la loss de valeur | 0.5 est standard |

### Paramètres d'entraînement

| Paramètre | Défaut | Description | Recommandation |
|-----------|--------|-------------|----------------|
| `rollout_steps` | 128 | Steps par rollout | 64 à 2048 selon la mémoire |
| `batch_size` | 64 | Taille des mini-batches | 32 à 256 |
| `timestamp` | 1M | Timesteps totaux | 100k pour tests, 1M+ pour production |

### Calcul du nombre d'updates

```
num_update = timestamp // rollout_steps
```

Exemple : `timestamp=100_000` et `rollout_steps=128` → 781 updates.

---

## Suivi de l'entraînement

### tqdm (barre de progression)

L'entraînement affiche une barre de progression avec les métriques en temps réel :

```
Entraînement PPO:  45%|████▌     | 45/100 [00:32<00:39,  1.40s/it, loss=0.4231, π=-0.0892, V=0.0312, H=0.0018, R=45.2]
```

### wandb (logging avancé)

```bash
uv run python run_train.py --wandb --wandb-project agri-drone-rl
```

Métriques loggées :
- `train/loss` — Loss totale PPO
- `train/policy_loss` — Loss de politique (actor)
- `train/value_loss` — Loss de valeur (critic)
- `train/entropy` — Entropie de la politique (exploration)
- `train/cumulative_reward` — Récompense cumulative de l'épisode
- `train/learning_rate` — Taux d'apprentissage courant
- `train/buffer_size` — Taille du buffer

### Interprétation des métriques

| Métrique | Signification | Bon signe |
|----------|---------------|-----------|
| `loss` diminue | L'agent apprend | ✅ Tendance à la baisse |
| `policy_loss` | Perte de la politique | ✅ Stabilise ou diminue |
| `value_loss` diminue | Le critique s'améliore | ✅ Tendance à la baisse |
| `entropy` stable | L'exploration est maintenue | ✅ Pas trop bas, pas trop haut |
| `cumulative_reward` augmente | L'agent obtient plus de récompenses | ✅ Tendance à la hausse |

---

## Évaluation et visualisation

### Générer un GIF

```bash
# Évaluer le dernier modèle
uv run python eval_agent.py

# Spécifier un modèle et un output
uv run python eval_agent.py --model saved_models/agri_drone_ppo.pt --output eval.gif

# Plusieurs épisodes, plus fluide
uv run python eval_agent.py --episodes 3 --frameskip 2 --fps 20
```

### Paramètres d'évaluation

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--model` | dernier modèle | Chemin vers le fichier .pt |
| `--episodes` | 1 | Nombre d'épisodes à enregistrer |
| `--max-steps` | 500 | Nombre max de pas par épisode |
| `--frameskip` | 2 | Capturer 1 image tous les N pas |
| `--fps` | 15 | Images par seconde du GIF |
| `--output` | eval_output.gif | Nom du fichier de sortie |

### Rendu PyBullet GUI

```bash
# Pour voir l'agent en direct (nécessite un display)
cd environment && python demo_env.py
```

---

## Guide des hyperparamètres

### Scénario 1 : Entraînement rapide (tests)

```bash
uv run python run_train.py \
    --episodes 10 \
    --rollout-steps 64 \
    --batch-size 32 \
    --lr 3e-3
```

### Scénario 2 : Entraînement équilibré

```bash
uv run python run_train.py \
    --episodes 100 \
    --rollout-steps 128 \
    --batch-size 64 \
    --lr 3e-4 \
    --wandb
```

### Scénario 3 : Entraînement long (production)

```bash
uv run python run_train.py \
    --episodes 500 \
    --rollout-steps 256 \
    --batch-size 128 \
    --lr 1e-4 \
    --gamma 0.999 \
    --wandb --wandb-project agri-drone-prod
```

---

## Résolution de problèmes

### Problème : La loss ne diminue pas

**Causes possibles :**
- Learning rate trop élevé → réduire à 1e-4
- Learning rate trop bas → augmenter à 1e-3
- Batch size trop petit → augmenter à 128
- Entropie trop basse → augmenter `ent_coef` à 0.05

### Problème : L'agent ne trouve pas les plantes

**Causes possibles :**
- Nombre d'updates insuffisant → augmenter `timestamp`
- Distance trop grande → réduire `world.size_x/y`
- Reward shaping insuffisant → vérifier `distance_shaping_reward`

### Problème : L'agent arrose mais ne rempliss pas le réservoir

**Causes possibles :**
- `basin_refill_radius` trop petit → augmenter à 5.0
- Récompense de remplissage trop faible → augmenter `refill_reward`

### Problème : Erreur CUDA

**Solution :**
```bash
# Vérifier la compatibilité CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Si False, réinstaller PyTorch avec CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Références

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyBullet Documentation](https://pybullet.org/wordpress/)
- [Weights & Biases](https://wandb.ai/)
- [rl_template](https://github.com/) — Framework RL interne
