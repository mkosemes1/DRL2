# DRL2 — Environnement de Drone Agricole pour Reinforcement Learning

## Présentation

Ce dépôt implémente un **environnement d'apprentissage par renforcement (RL)** pour un drone agricole hexacoptère, construit avec l'API Gymnasium v1 et PyBullet pour le rendu 3D. L'environnement `AgriDroneEnv` est conçu pour entraîner un agent à accomplir des missions agricoles complexes :

- **Tâche d'irrigation** : arroser des groupes de plantes répartis dans la carte
- **Gestion de réservoir** : remplir le réservoir d'eau à une bassine de ravitaillement
- **Survol et cartographie** d'une parcelle
- **Détection et traitement** des plantes malades (pulvérisation)
- **Évitement d'obstacles** et gestion du vent
- **Retour automatique** à la base lorsque la batterie est faible

---

## Architecture

```
DRL2/
├── environment/                     # Package principal
│   ├── agri_drone_env.py            # AgriDroneEnv — environnement Gymnasium principal
│   ├── minimal_fly_env.py           # MinimalFlyEnv — env simplifiée (vol uniquement)
│   ├── demo_env.py                  # Script de démonstration avec rendu PyBullet
│   ├── obstacles.py                 # ObstacleManager — gestion des obstacles sphériques
│   ├── pybullet_renderer.py         # Rendu 3D PyBullet (découplé de la physique)
│   ├── physics/
│   │   ├── drone_dynamics.py        # DroneDynamics + DroneParams + DroneState
│   │   └── wind_model.py            # WindModel — modèle de vent avec rafales
│   ├── reward/
│   │   └── reward_function.py       # RewardCalculator + RewardConfig
│   └── utils/
│       └── normalization.py         # Fonctions normalize() / denormalize()
├── agent/
│   └── model.py                     # Agent PPO (BaseAgent) — réseau acteur-critique MLP
├── train.py                         # Trainer PPO (BaseTrain) — pipeline d'entraînement PPO
├── requirements.txt                 # Dépendances pinées
└── README.md                        # Ce fichier
```

---

## Espace d'observation

Le vecteur d'observation est normalisé dans `[-1, 1]` et sa dimension totale est **17 + 3 + 1 + N×4** (par défaut N=5 → 39 dimensions).

| Plage | Dims | Description |
|-------|------|-------------|
| 0–2 | 3 | Position du drone (x, y, z) |
| 3–5 | 3 | Vitesse linéaire (vx, vy, vz) |
| 6–8 | 3 | Attitude (roll, pitch, yaw) |
| 9–11 | 3 | Vitesses angulaires (roll_rate, pitch_rate, yaw_rate) |
| 12 | 1 | Distance à l'objectif |
| 13 | 1 | Erreur de cap (heading error) |
| 14 | 1 | Niveau de batterie |
| 15 | 1 | Distance à l'obstacle le plus proche |
| 16 | 1 | Intensité du vent |
| 17–19 | 3 | Coordonnées de la bassine d'eau (x, y, z) |
| 20 | 1 | Niveau du réservoir d'eau (0–100) |
| 21+ | N×4 | Matrice des groupes de plantes (x, y, z, is_watered) |

---

## Espace d'action

L'espace d'action est continu, de **6 dimensions** dans `[-1, 1]` :

| Indice | Nom | Description |
|--------|-----|-------------|
| 0 | `throttle` | Poussée totale (0–100 %) |
| 1 | `roll_cmd` | Consigne d'inclinaison en roulis |
| 2 | `pitch_cmd` | Consigne d'inclinaison en tangage |
| 3 | `yaw_cmd` | Commande de vitesse de lacet |
| 4 | `spray_on` | Pulvérisation (activée si > 0) |
| 5 | `irrigate_on` | Irrigation (activée si > 0) |

---

## Mécanique de la tâche d'irrigation (Water Task)

La tâche principale consiste à arroser N groupes de plantes répartis aléatoirement dans la carte.

### Principe

1. **Arrosage** : lorsqu'un groupe non arrosé est à moins de 2 m du drone et que le réservoir contient assez d'eau, le groupe est marqué comme arrosé et `water_consumption` unités sont déduites du réservoir.
2. **Remplissage** : lorsque le drone se trouve dans le rayon `basin_refill_radius` de la bassine, le réservoir est remis à 100.
3. **Fin de mission** : l'épisode se termine avec succès (`terminated=True`) lorsque tous les groupes sont arrosés.

### Configuration

```python
config = {
    "water_task": {
        "basin_position": [15.0, 15.0, 0.5],  # Position (x, y, z) de la bassine
        "basin_refill_radius": 3.0,            # Rayon de remplissage (m)
        "water_consumption": 2.0,              # Eau consommée par arrosage
        "num_plant_groups": 5,                 # Nombre de groupes de plantes
    }
}
```

---

## Structure de la récompense

La récompense est calculée par `RewardCalculator.compute_water_task()` et combine :

| Terme | Valeur par défaut | Description |
|-------|-------------------|-------------|
| `watering_reward` | +5.0 | Arrosage réussi d'un groupe |
| `refill_reward` | +1.0 | Remplissage du réservoir à la bassine |
| `time_penalty_per_group` | −0.02 | Pénalité par groupe non arrosé et restant |
| `distance_shaping_reward` | +0.05 | Bonus si le drone se rapproche du groupe le plus proche |
| `mission_complete_reward` | +100.0 | Bonus terminal quand tous les groupes sont arrosés |

La récompense totale est la somme de tous ces termes à chaque pas de l'environnement.

---

## Exemple de configuration complète

```python
config = {
    "world": {
        "size_x": 60.0,
        "size_y": 60.0,
        "ground_z": 0.0,
        "size_z": 50.0,
        "field_cells_x": 20,
        "field_cells_y": 20,
    },
    "drone": {
        "dry_mass": 10.0,
        "payload_mass_full": 5.0,
        "gravity": 9.81,
        "max_thrust_total": 350.0,
        "drag_coefficient": 0.08,
        "max_tilt_angle_rad": 0.5236,
        "max_angular_rate": 3.0,
        "attitude_time_constant": 0.08,
        "urdf_path": "environment/agri_hexacopter_pro.urdf",
    },
    "simulation": {"dt": 0.02, "max_episode_steps": 1000},
    "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
    "water_task": {
        "basin_position": [15.0, 15.0, 0.5],
        "basin_refill_radius": 3.0,
        "water_consumption": 2.0,
        "num_plant_groups": 5,
    },
}
```

---

## Lancement des démonstrations et tests

### Prérequis

```bash
# Installation des dépendances
pip install -r requirements.txt
```

### Démonstration avec rendu PyBullet

```bash
# Env principale avec grille de champs affichée
cd environment && python demo_env.py

# Env minimale (vol uniquement, pas de logique agricole)
cd environment && python minimal_fly_env.py
```

### Tests manuels

```bash
# Script de test de la dynamique (pas pytest — script manuel)
cd environment && python test_dynamics.py
```

### Entraînement

```bash
# Entraînement PPO (100 updates par défaut)
uv run python run_train.py

# Avec options
uv run python run_train.py --episodes 200 --lr 3e-4 --wandb
```

Voir `docs/FORMATION.md` pour le guide complet d'entraînement.

---

## Pipeline d'entraînement

Le module `train.py` fournit un pipeline d'entraînement PPO qui étend `BaseTrain` de la bibliothèque `rl_template`.

### Installation des dépendances

```bash
uv pip install -r requirements.txt
```

### Utilisation

```python
from train import Trainer
from rl_template.config import PPOConfig, TrainConfig

train_config = TrainConfig(
    model_name="agri_drone",
    model_saved_path="saved_models",
    timestamp=128,
    batch_size=64,
    rollout_steps=128,
)

ppo_config = PPOConfig(
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
)

# Optionnel : configuration wandb
wandb_config = {
    "project": "agri-drone-rl",
    "entity": None,
    "name": "ppo-training-v1",
    "config": {"num_plant_groups": 5},
}

trainer = Trainer(
    env=env,
    agent=agent,
    train_config=train_config,
    ppo_config=ppo_config,
    wandb_config=wandb_config,
)

trainer.train(verbose=True)  # tqdm + wandb
```

### Fonctionnalités

| Fonctionnalité | Description |
|---------------|-------------|
| Actions continues | Gestion correcte des actions 6D (throttle, roll, pitch, yaw, spray, irrigate) |
| tqdm | Barre de progression pendant l'entraînement |
| wandb | Logging optionnel des métriques (loss, π, V, H, R, lr) |
| Sauvegarde | Modèle sauvegardé automatiquement en fin d'entraînement |

### Métriques loggées

- `train/loss` — Loss totale PPO
- `train/policy_loss` — Loss de politique
- `train/value_loss` — Loss de valeur
- `train/entropy` — Entropie de la politique
- `train/cumulative_reward` — Récompense cumulative de l'épisode
- `train/learning_rate` — Taux d'apprentissage courant

---

## Évaluation

Le script `eval_agent.py` charge un modèle sauvegardé et génère un GIF de l'agent en action.

```bash
# Évaluer le dernier modèle
uv run python eval_agent.py

# Spécifier un modèle
uv run python eval_agent.py --model saved_models/agri_drone_ppo.pt --output demo.gif

# Plusieurs épisodes, plus fluide
uv run python eval_agent.py --episodes 3 --frameskip 2 --fps 20
```

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--model` | dernier modèle | Chemin vers le fichier .pt |
| `--episodes` | 1 | Nombre d'épisodes |
| `--max-steps` | 500 | Max pas par épisode |
| `--frameskip` | 2 | Capturer 1 image tous les N pas |
| `--fps` | 15 | Images par seconde du GIF |
| `--output` | eval_output.gif | Nom du fichier de sortie |

---

## Dépendances principales

| Package | Version | Usage |
|---------|---------|-------|
| `gymnasium` | ≥ 1.0 | API RL standard |
| `pybullet` | ≥ 3.2 | Rendu 3D et simulation physique |
| `numpy` | ≥ 2.0 | Calcul numérique |
| `stable-baselines3` | ≥ 2.0 | Algorithmes RL (PPO, SAC, etc.) |
| `torch` | ≥ 2.0 | Réseaux de neurones |
| `wandb` | ≥ 0.15 | Suivi des expériences |
| `pillow` | ≥ 10.0 | Création de GIFs pour l'évaluation |

---

## Notes techniques

- **Physique** : modèle simplifié de corps rigide avec commande en attitude (premier ordre). La poussée est projetée dans le repère monde via les angles d'Euler pour produire l'accélération horizontale.
- **Vent** : modèle de rafales aléatoires, désactivé par défaut (activable via `WindModel(enabled=True)`).
- **Obstacles** : gestionnaires d'obstacles sphériques avec détection de collision surface-à-surface.
- **Normalisation** : toutes les observations sont ramenées dans `[-1, 1]` pour stabiliser l'apprentissage.
- **Rendu** : PyBullet est **découplé** de la physique d'entraînement — le rendu n'est activé que si `render_mode="human"`.
