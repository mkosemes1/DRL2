# DRL2 — Environnement de Drone Agricole pour Reinforcement Learning

## 🎯 Présentation

**DRL2** est un environnement d'apprentissage par renforcement (RL) conçu pour entraîner un drone agricole hexacoptère à réaliser des missions autonomes de gestion de cultures. Construit sur l'API **Gymnasium v1** avec **PyBullet** pour la simulation physique et le rendu 3D, cet environnement modélise un scénario réaliste de précision agricole.

L'agent apprend à :

- **Irriguer** des groupes de plantes répartis dans une parcelle en gérant un réservoir d'eau à capacité finie
- **Remplir** son réservoir à une bassine de ravitaillement quand l'eau vient à manquer
- **Pulvériser** des zones malades en utilisant un produit phytopharmaceutique
- **Éviter** des obstacles sphériques et résister aux rafales de vent
- **Planifier** un itinéraire optimisé pour minimiser la consommation d'énergie et maximiser la couverture

Le projet cible la recherche en **planification spatiale avec gestion de ressources finies** — un problème fondamental en RL appliqué à la robotique agricole.

---

## 📁 Architecture

```
DRL2/
├── environment/                          # Package principal
│   ├── env.py                            # AgriDroneEnv (732 lignes) — env Gymnasium
│   ├── minimal_fly_env.py                # MinimalFlyEnv — env simplifiée (vol)
│   ├── demo_env.py                       # Script de démonstration PyBullet
│   ├── obstacles.py                      # ObstacleManager (60 lignes)
│   ├── pybullet_renderer.py              # Rendu 3D PyBullet (découplé physique)
│   ├── agri_hexacopter_pro.urdf          # Modèle URDF du hexacoptère
│   ├── physics/
│   │   ├── drone_dynamics.py             # DroneDynamics + Params + State (179 lignes)
│   │   └── wind_model.py                 # WindModel — vent avec rafales
│   ├── reward/
│   │   └── reward_function.py            # RewardCalculator + RewardConfig (339 lignes)
│   ├── utils/
│   │   └── normalization.py              # normalize() / denormalize() vers [-1, 1]
│   └── tests/                            # 323 tests pytest
│       ├── test_drone_dynamics.py
│       ├── test_wind_model.py
│       ├── test_obstacles.py
│       ├── test_normalization.py
│       ├── test_reward_function.py
│       ├── test_agri_drone_env.py
│       ├── test_field_cell.py
│       ├── test_minimal_fly_env.py
│       ├── test_demo_config.py
│       ├── test_integration.py
│       ├── test_edge_cases.py
│       ├── test_base_env_interface.py
│       ├── test_agent_model.py
│       ├── test_training_pipeline.py
│       └── test_trainer_integration.py
├── agent/
│   └── model.py                          # Agent PPO acteur-critique MLP (126 lignes)
├── train.py                              # Trainer PPO (96 lignes)
├── eval_agent.py                         # Évaluation agent + GIF (224 lignes)
├── requirements.txt                      # Dépendances pinées
├── specify.md                            # Spécification MDP
├── update.md                             # Journal de session
└── docs/
    └── DOCUMENTATION_DETAILED.md         # Doc détaillée (1531 lignes)
```

---

## 🌍 Environnement

L'environnement `AgriDroneEnv` hérite de `BaseEnv` (rl_template) et implémente l'interface Gymnasium standard (`reset`, `step`, `render`). La simulation physique est gérée par `DroneDynamics` (intégration Eulerienne pas-à-pas), tandis que PyBullet est utilisé uniquement pour le rendu visuel.

### Espace d'observation

Le vecteur d'observation est **totalement normalisé** dans `[-1, 1]` et possède **17 + 3 + 1 + N×4** dimensions (par défaut N=5 → **39 dimensions**).

| Dims | Nombre | Description |
|------|--------|-------------|
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
| 20 | 1 | Niveau du réservoir d'eau (0–100, normalisé) |
| 21+ | N×4 | Matrice groupes de plantes (x, y, z, is_watered) par groupe |

> **Note** : `is_watered` est un booléen (0.0 / 1.0) transformé en [-1, 1] via `2 * value - 1` lors de la construction de l'observation.

### Espace d'action

L'espace d'action est **continu**, composé de **6 dimensions** dans `[-1, 1]` :

| Indice | Nom | Description |
|--------|-----|-------------|
| 0 | `throttle` | Poussée totale (0–100 % de la poussée max) |
| 1 | `roll` | Consigne d'inclinaison en roulis |
| 2 | `pitch` | Consigne d'inclinaison en tangage |
| 3 | `yaw` | Commande de vitesse de lacet |
| 4 | `spray` | Pulvérisation (activée si > 0) |
| 5 | `irrigate` | Irrigation (activée si > 0) |

Les 4 premières dimensions (`throttle`, `roll`, `pitch`, `yaw`) sont transmises à `DroneDynamics.step()` pour la physique. Les dimensions 4 et 5 (`spray`, `irrigate`) contrôlent la logique métier de traitement et d'irrigation.

### Mécanique de la tâche d'irrigation

La tâche principale (`water task`) consiste à arroser **N groupes de plantes** aléatoirement répartis dans la parcelle tout en gérant un réservoir d'eau à capacité finie.

#### Cycle d'irrigation

1. **Arrosage** : Lorsqu'un groupe non arrosé est à moins de `watering_proximity` (défaut : 2.0 m) du drone, que l'action `irrigate > 0` et que le réservoir contient au moins `water_consumption` unités :
   - Le groupe est marqué comme arrosé (`is_watered = 1.0`)
   - Le réservoir diminue de `water_consumption` unités

2. **Remplissage** : Lorsque le drone se trouve dans le rayon `basin_refill_radius` (défaut : 3.0 m) de la bassine :
   - Le réservoir est remis à 100.0
   - Une récompense de +1.0 est accordée **uniquement si le réservoir était inférieur à 98.0** (mécanisme anti-farming)

3. **Fin de mission** : Lorsque tous les groupes de plantes sont arrosés :
   - L'épisode se termine immédiatement (`terminated = True`)
   - Un bonus de +100.0 est accordé

#### Configuration de la tâche d'eau

```python
"water_task": {
    "basin_position": [15.0, 15.0, 0.5],   # Position (x, y, z) de la bassine
    "basin_refill_radius": 3.0,             # Rayon de remplissage (m)
    "water_consumption": 2.0,               # Eau consommée par arrosage
    "watering_proximity": 2.0,              # Distance max d'arrosage (m)
    "num_plant_groups": 5,                  # Nombre de groupes de plantes
}
```

### Structure de la récompense

La récompense est calculée par `RewardCalculator.compute_water_task()` à chaque pas de simulation.

| Terme | Valeur | Description |
|-------|--------|-------------|
| `watering` | **+5.0** | Arrosage réussi d'un groupe de plantes |
| `refill` | **+1.0** | Remplissage du réservoir à la bassine (seuil : tank < 98.0) |
| `time_penalty` | **-0.02 × N** | Pénalité par groupe non arrosé (N = nbrestant) |
| `distance_shaping` | **+0.05** | Bonus si le drone se rapproche du groupe le plus proche |
| `mission_complete` | **+100.0** | Bonus terminal quand tous les groupes sont arrosés |

La récompense totale est la **somme** de tous ces termes à chaque pas. Le `distance_shaping` fournit un signal de guidance continue pour orienter l'agent vers les cibles, tandis que `watering` et `mission_complete` fournissent des signaux sparse mais importants.

---

## ⚙️ Configuration de l'environnement

L'environnement est configuré via un dictionnaire `config` passé au constructeur de `AgriDroneEnv`. Voici l'arbre complet des paramètres avec leurs valeurs par défaut.

### Paramètres du monde

```python
"world": {
    "size_x": 60.0,          # Largeur du monde (m) → bornes x: [-30, +30]
    "size_y": 60.0,          # Profondeur du monde (m) → bornes y: [-30, +30]
    "ground_z": 0.0,         # Altitude du sol (m)
    "size_z": 50.0,          # Altitude maximale (m) → bornes z: [0, 50]
    "field_cells_x": 20,     # Nombre de cellules de la grille agricole en X
    "field_cells_y": 20,     # Nombre de cellules de la grille agricole en Y
}
```

### Paramètres du drone

```python
"drone": {
    "dry_mass": 10.0,                # Masse à vide (kg)
    "payload_mass_full": 5.0,         # Masse de charge maximale (kg)
    "gravity": 9.81,                  # Accélération gravitationnelle (m/s²)
    "max_thrust_total": 350.0,        # Poussée maximale totale (N)
    "drag_coefficient": 0.08,         # Coefficient de traînée aérodynamique
    "max_tilt_angle_rad": 0.5236,     # Angle d'inclinaison max (30° en radians)
    "max_angular_rate": 3.0,          # Vitesse angulaire maximale (rad/s)
    "attitude_time_constant": 0.08,   # Constante de temps du contrôleur attitude (s)
    "urdf_path": "environment/agri_hexacopter_pro.urdf",  # Chemin vers le modèle URDF
}
```

### Paramètres de simulation

```python
"simulation": {
    "dt": 0.02,                       # Pas de temps (s) → fréquence de 50 Hz
    "max_episode_steps": 1000,        # Nombre maximal de pas par épisode
}
```

### Paramètres de normalisation

```python
"normalization": {
    "max_velocity": 50.0,             # Vitesse maximale pour la normalisation
    "max_distance": 100.0,            # Distance maximale pour la normalisation
}
```

### Configuration complète par défaut

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
    "simulation": {
        "dt": 0.02,
        "max_episode_steps": 1000,
    },
    "normalization": {
        "max_velocity": 50.0,
        "max_distance": 100.0,
    },
    "water_task": {
        "basin_position": [15.0, 15.0, 0.5],
        "basin_refill_radius": 3.0,
        "water_consumption": 2.0,
        "watering_proximity": 2.0,
        "num_plant_groups": 5,
    },
}
```

---

## 🤖 Agent PPO

L'agent est implémenté dans `agent/model.py` et hérite de `BaseAgent` (rl_template). Il s'agit d'un réseau de neurones **acteur-critique** à double tête partageant un extracteur de caractéristiques commun.

### Architecture du réseau

```
Input (obs_dim)
    │
    ▼
Linear(obs_dim → 128)  →  Tanh
    │
    ▼
Linear(128 → 64)  →  Tanh          ← Features partagées
    │
    ├──────────────────────┐
    ▼                      ▼
[Tête Acteur]         [Tête Critique]
    │                      │
Linear(64 → 64)        Linear(64 → 64)
    │                      │
  ReLU                    ReLU
    │                      │
Linear(64 → n_action)  Linear(64 → 1)
    │                      │
    ▼                      ▼
  mean                   value (V)
    │
policy_log_std (param)
    │
    ▼
  Normal(mean, exp(log_std))
```

**Résumé de l'architecture :**

- **Extracteur partagé** : 2 couches linéaires (128 → 64) avec activation Tanh
- **Tête acteur** : 2 couches linéaires (64 → 64 → n_action) avec ReLU, plus un paramètre `policy_log_std` appris
- **Tête critique** : 2 couches linéaires (64 → 64 → 1) avec ReLU

### Initialisation des poids

| Composant | Méthode | Gain | Rationale |
|-----------|---------|------|-----------|
| Couches cachées (shared) | Orthogonale | √2 | Conserve la variance du signal via ReLU |
| Tête acteur | Orthogonale | √2 | Comportement standard pour couches cachées |
| Tête critique | Orthogonale | 1.0 | Sorties de valeur centrées au début |

### Distribution de politique

La politique utilise une **distribution normale** (`torch.distributions.Normal`) paramétrée par :

- **Moyenne (μ)** : sortie du réseau acteur (forme `(batch, n_action)`)
- **Écart-type (σ)** : `exp(policy_log_std)` où `policy_log_std` est un paramètre appris (forme `(n_action,)`)

L'exponentielle garantit un écart-type toujours positif. La distribution produit des actions continues qui sont échantillonnées durant l'entraînement (stochastique) ou remplacées par la moyenne durant l'évaluation (déterministe).

---

## 🏋️ Pipeline d'entraînement

Le module `train.py` fournit un pipeline d'entraînement PPO complet qui étend `BaseTrain` de la bibliothèque `rl-template`. Il gère la collecte de données, la mise à jour des poids du réseau, la sauvegarde des modèles et le logging des métriques.

### Configuration de l'entraînement (TrainConfig)

```python
from rl_template.config import TrainConfig

train_config = TrainConfig(
    device="cpu",                    # Device PyTorch ('cpu' ou 'cuda')
    model_name="agriDrone",          # Nom du modèle (pour la sauvegarde)
    model_saved_path="./checkpoints",# Répertoire de sauvegarde des modèles
    # batch_size=64,                 # Taille des mini-batches (défaut rl_template)
    # rollout_steps=128,             # Steps par rollout (défaut rl_template)
    # num_update=100,                # Nombre d'iterations d'update (défaut rl_template)
)
```

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `device` | Device PyTorch pour l'entraînement | `"cpu"` |
| `model_name` | Nom utilisé pour nommer les fichiers sauvegardés | `"agriDrone"` |
| `model_saved_path` | Répertoire où stocker les checkpoints | `"./checkpoints"` |
| `batch_size` | Taille des mini-batches pour l'optimisation | `64` |
| `rollout_steps` | Nombre de pas collectés avant un update | `128` |
| `num_update` | Nombre total d'itérations d'entraînement | `100` |

### Configuration PPO (PPOConfig)

```python
from rl_template.config import PPOConfig

ppo_config = PPOConfig(
    lr=3e-4,           # Taux d'apprentissage
    gamma=0.99,        # Facteur de discount
    gae_lambda=0.95,   # Lambda pour GAE (Generalized Advantage Estimation)
    clip_eps=0.2,      # Epsilon pour le clipping PPO
    ent_coef=0.01,     # Coefficient de régularisation entropique
    value_coef=0.5,    # Coefficient de la loss de valeur
    # max_grad_norm=0.5,  # Clipping des gradients (défaut rl_template)
    # n_epochs=10,        # Époques par update (défaut rl_template)
    # batch_size=64,      # Mini-batch size (défaut rl_template)
)
```

### Tableau exhaustif des hyperparamètres PPO

| Paramètre | Type | Défaut | Description | Impact |
|-----------|------|--------|-------------|--------|
| `lr` | float | `3e-4` | Taux d'apprentissage de l'optimiseur Adam | Trop élevé → instabilité, trop bas → convergence lente |
| `gamma` | float | `0.99` | Facteur de discount pour les récompenses futures | Plus proche de 1.0 → l'agent valorise davantage le long terme |
| `gae_lambda` | float | `0.95` | Lambda pour le calcul du Generalized Advantage Estimation | Plus élevé → estimation de l'avantage plus lisse mais plus biaisée |
| `clip_eps` | float | `0.2` | Epsilon pour le clipping ratio PPO (π_new / π_old) | Plus petit → updates plus conservatrices, plus grand → updates plus agressives |
| `ent_coef` | float | `0.01` | Coefficient de l'entropie dans la loss totale | Plus élevé → plus d'exploration, 0.0 → exploitation pure |
| `value_coef` | float | `0.5` | Poids de la loss de valeur dans la loss totale | Plus élevé → le critique domine, plus faible → la politique domine |
| `max_grad_norm` | float | `0.5` | Seuil de clipping des gradients (norme L2) | Protège contre les explosions de gradients |
| `n_epochs` | int | `10` | Nombre de passes sur le buffer par update | Plus élevé → meilleur usage des données mais risque de sur-apprentissage |
| `batch_size` | int | `64` | Taille des mini-batches pour chaque epoch | Plus grand → gradient plus stable mais plus de mémoire |

### Guide : comment changer les hyperparamètres

#### Augmenter l'exploration

Si l'agent se concentre trop tôt sur une stratégie sous-optimale, augmentez l'entropie :

```python
ppo_config = PPOConfig(
    ent_coef=0.05,     # ↑ de 0.01 à 0.05 pour plus d'exploration
    clip_eps=0.3,      # ↑ de 0.2 à 0.3 pour des updates plus permissives
)
```

#### Accélérer la convergence

Pour des environnements plus simples ou quand l'agent a déjà une bonne base :

```python
ppo_config = PPOConfig(
    lr=1e-3,           # ↑ taux d'apprentissage
    gamma=0.95,        # ↓ discount (moins de vision long terme)
    n_epochs=15,       # ↑ plus de passes sur les données collectées
)
```

#### Stabiliser l'entraînement

En cas d'instabilité ou de divergence de la loss :

```python
ppo_config = PPOConfig(
    lr=1e-4,           # ↓ réduire le learning rate
    clip_eps=0.1,      # ↓ clipping plus strict
    max_grad_norm=0.3, # ↓ clipping des gradients plus agressif
    ent_coef=0.02,     # ↑ légère boost d'exploration
)
```

#### Entraînement long terme (recherche)

Pour des runs de recherche avec beaucoup de temps de calcul :

```python
train_config = TrainConfig(
    num_update=1000,        # ↑ 10x plus d'itérations
    rollout_steps=256,      # ↑ plus de données par rollout
    batch_size=128,         # ↑ mini-batches plus grands
)
ppo_config = PPOConfig(
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    n_epochs=10,
)
```

### Exemple complet de configuration

```python
from train import Trainer
from rl_template.config import PPOConfig, TrainConfig

# Configuration de l'entraînement
train_config = TrainConfig(
    device="cpu",
    model_name="agriDrone",
    model_saved_path="./checkpoints",
)

# Configuration PPO
ppo_config = PPOConfig(
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    value_coef=0.5,
)

# Configuration wandb (optionnel)
wandb_config = {
    "project": "agri-drone-rl",
    "entity": None,
    "name": "ppo-training-v1",
    "config": {"num_plant_groups": 5},
}

# Lancement
trainer = Trainer(
    env=env,
    agent=agent,
    train_config=train_config,
    ppo_config=ppo_config,
    wandb_config=wandb_config,
)
trainer.train(verbose=True)
```

### Fonctionnalités

| Fonctionnalité | Description |
|---------------|-------------|
| **Actions continues** | Gestion correcte des actions 6D via sommation des log-probs sur les dimensions d'action |
| **tqdm** | Barre de progression pendant l'entraînement |
| **wandb** | Logging optionnel des métriques (loss, π, V, H, R, lr) |
| **Sauvegarde automatique** | Modèle sauvegardé à chaque update dans `model_saved_path` |
| **Buffer continu** | Créé avec `action_shape=(act_dim,)` pour des actions 6D |

### Métriques loggées

Les métriques suivantes sont loggées à chaque itération via wandb :

| Métrique | Clé wandb | Description |
|----------|-----------|-------------|
| Loss totale PPO | `Loss` | Somme de policy_loss + value_loss - entropy |
| Loss de politique | `policy loss` | Pénalité PPO clipped sur la ratio π_new/π_old |
| Loss de valeur | `value loss` | MSE entre la valeur prédite et le retour calculé |
| Entropie | `entropy loss` | Entropie de la distribution de politique |
| Récompense cumulative | `reward` | Récompense totale de l'épisode courant |

### Déroulement interne

L'entrée `Trainer.train()` exécute la boucle suivante :

1. **rollout_phase(state)** : Collecte `rollout_steps` pas de données dans le buffer (states, actions, rewards, dones)
2. **update_weights(step)** : Calcule les avantages GAE, exécute `n_epochs` passes de mise à jour PPO sur des mini-batches
3. **save_model()** : Sauvegarde les poids du réseau dans `model_saved_path`
4. **Logging** : Enregistre les métriques dans wandb

> **Point technique** : `rollout_phase()` est surchargée par rapport à `BaseTrain` pour corriger un bug avec les actions continues 6D. La version originale utilisait `.item()` sur les actions, ce qui provoque un crash pour des tableaux multi-dimensionnels. La version corrigée utilise `.cpu().numpy()` et somme les log-probs sur les dimensions d'action avant insertion dans le buffer.

---

## 📊 Évaluation

Le script `eval_agent.py` charge un modèle sauvegardé et génère un GIF animé de l'agent en action dans l'environnement.

### Commandes

```bash
# Évaluer le dernier modèle sauvegardé
uv run python eval_agent.py

# Spécifier un modèle
uv run python eval_agent.py --model saved_models/agent.pt

# Plusieurs épisodes avec un FPS spécifique
uv run python eval_agent.py --episodes 3 --fps 20

# Chemin de sortie personnalisé
uv run python eval_agent.py --model checkpoints/agriDrone.pt --output demo.gif
```

### Arguments CLI

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--model` | dernier modèle trouvé | Chemin vers le fichier `.pt` du modèle |
| `--episodes` | 1 | Nombre d'épisodes à enregistrer |
| `--max-steps` | 500 | Nombre maximal de pas par épisode |
| `--frameskip` | 2 | Capturer une image tous les N pas |
| `--fps` | 15 | Images par seconde du GIF |
| `--output` | `eval_output.gif` | Nom du fichier GIF de sortie |
| `--width` | 640 | Largeur de l'image en pixels |
| `--height` | 480 | Hauteur de l'image en pixels |

### Modes de rendu

L'environnement supporte trois modes de rendu :

| Mode | Description | Usage |
|------|-------------|-------|
| `render_mode="rgb_array"` | Capture headless via PyBullet DIRECT (pas de fenêtre) | Évaluation, génération GIF/video |
| `render_mode="human"` | Fenêtre PyBullet GUI temps réel | Débogage interactif |
| `render_mode=None` | Aucun rendu | Entraînement (max performance) |

Le mode `rgb_array` utilise un client PyBullet **séparé** du client physique. Les appels API PyBullet dans `_render_rgb_array` spécifient `physicsClientId=self._client` pour éviter les conflits. Les frames sont capturées à 640×480 via `getCameraImage()`.

---

## 🧪 Tests

Le projet contient **323 tests pytest** couvrant l'intégralité de la codebase : dynamique du drone, modèle de vent, gestion des obstacles, fonction de récompense, environnement Gymnasium, utilsitaires de normalisation, modèle agent, pipeline d'entraînement et cas limites.

### Commandes

```bash
# Exécuter tous les tests
uv run python -m pytest environment/tests/ -v

# Tests spécifiques
uv run python -m pytest environment/tests/test_reward_function.py -v
uv run python -m pytest environment/tests/test_agri_drone_env.py -v
uv run python -m pytest environment/tests/test_drone_dynamics.py -v
uv run python -m pytest environment/tests/test_integration.py -v
uv run python -m pytest environment/tests/test_edge_cases.py -v
uv run python -m pytest environment/tests/test_agent_model.py -v
uv run python -m pytest environment/tests/test_training_pipeline.py -v
uv run python -m pytest environment/tests/test_trainer_integration.py -v
```

### Couverture des tests

| Fichier de test | Sujet |
|-----------------|-------|
| `test_drone_dynamics.py` | Dynamique physique, intégration, limites |
| `test_wind_model.py` | Modèle de vent et rafales |
| `test_obstacles.py` | Génération, collision, distance |
| `test_normalization.py` | Fonctions normalize/denormalize |
| `test_reward_function.py` | RewardCalculator, RewardConfig, compute_water_task |
| `test_agri_drone_env.py` | Reset, step, observation, espace d'action |
| `test_field_cell.py` | États de la cellule du champ |
| `test_minimal_fly_env.py` | Environnement simplifiée |
| `test_demo_config.py` | Configuration de démonstration |
| `test_integration.py` | Flux d'intégration bout en bout |
| `test_edge_cases.py` | Cas limites et conditions extrêmes |
| `test_base_env_interface.py` | Conformité à l'interface BaseEnv |
| `test_agent_model.py` | Architecture et forward pass de l'agent |
| `test_training_pipeline.py` | Rollout, update, sauvegarde du Trainer |
| `test_trainer_integration.py` | Boucle d'entraînement complète, loss, métriques |

---

## 📦 Dépendances

Les dépendances sont pinées dans `requirements.txt`. Les packages principaux sont :

| Package | Version | Usage |
|---------|---------|-------|
| `gymnasium` | 1.3.0 | API standard pour les environnements RL |
| `pybullet` | 3.2.7 | Simulation physique et rendu 3D |
| `numpy` | 2.5.1 | Calcul numérique |
| `torch` | 2.13.0 | Réseaux de neurones et optimisation |
| `rl-template` | 0.1.2 | Framework RL (BaseEnv, BaseAgent, BaseTrain, PPO) |
| `stable-baselines3` | 2.9.0 | Algorithmes RL (disponible mais non utilisé directement) |
| `wandb` | 0.28.0 | Suivi des expériences et logging |
| `tqdm` | 4.68.4 | Barres de progression |
| `pillow` | 12.3.0 | Création de GIFs pour l'évaluation |
| `pytest` | 9.1.1 | Framework de tests |

### Installation

```bash
# Utiliser uv (recommandé — gère le .venv automatiquement)
uv pip install -r requirements.txt
```

> **Important** : Utilisez `uv run` pour toutes les commandes Python du projet. L'exécution avec `python` directement risque d'utiliser l'interpréteur système et de manquer des dépendances.

---

## 📝 Notes techniques

### Physique du drone

Le drone est modélisé comme un corps rigide unique avec une commande en attitude de premier ordre. Le modèle inclut :

- **Poussée** : projetée dans le repère monde via les angles d'Euler pour produire l'accélération horizontale (mécanisme physique réel des multirotors)
- **Traînée** : force aérodynamique linéaire `F_drag = -k × v`
- **Saturation** : vitesse et angle d'inclinaison limités aux valeurs physiques du drone
- **Gravité** : constante 9.81 m/s² appliquée en permanence
- **Sol** : le drone ne peut pas descendre sous z = 0.05 m

### Modèle de vent

Le `WindModel` implémente un modèle de rafales aléatoires. Par défaut, le vent est **désactivé** (`enabled=False`). Lorsqu'activé :

- Une vitesse et direction initiales sont choisies aléatoirement
- À chaque pas, une rafale aléatoire se produit avec une probabilité de 5 %
- Entre les rafales, le vent dérive légèrement (bruit gaussien)
- La vitesse est bornée à `max_speed` (défaut : 5.0 m/s)

### Gestion des obstacles

L'`ObstacleManager` génère des obstacles sphériques aléatoires dans la carte. La détection de collision est basée sur la distance surface-à-surface (distance euclidienne centres moins somme des rayons).

### Normalisation

Toutes les observations sont ramenées dans `[-1, 1]` via une normalisation min-max avec clipping. Le booléen `is_watered` est transformé de {0, 1} vers {-1, +1}. Cette normalisation est essentielle pour :

- Stabiliser la convergence du réseau de neurones
- Équilibrer les gradients entre les différentes composantes
- Améliorer l'estimation de l'avantage (GAE) et de la fonction de valeur

### Rendu PyBullet

PyBurn est **découplé** de la physique d'entraînement. Le rendu n'est activé que si `render_mode` est `"human"` ou `"rgb_array"`. Le mode headless (`rgb_array`) utilise un client PyBullet DIRECT séparé du client de simulation pour éviter les interférences.

### Points d'attention

1. **`rl-template`** est le framework RL sous-jacent. L'environment hérite de `BaseEnv`, l'agent de `BaseAgent`, et le trainer de `BaseTrain`
2. **Actions continues** : le `rollout_phase()` et `update_weights()` du `Trainer` sont surchargés pour gérer correctement les actions 6D continues (sommation des log-probs sur les dimensions d'action)
3. **Pas de système de build** : il n'y a pas de `pyproject.toml` ni de `setup.py`. Les imports utilisent `sys.path.insert` dans certains scripts

---

## 📚 Documentation complémentaire

Pour une documentation exhaustive de l'architecture, des choix de design et des détails d'implémentation, consultez :

**[Documentation détaillée](docs/DOCUMENTATION_DETAILED.md)** (1531 lignes)

La specification MDP (espace d'observation, espace d'action, fonctions de transition) est décrite dans :

**[Spécification MDP](specify.md)**
