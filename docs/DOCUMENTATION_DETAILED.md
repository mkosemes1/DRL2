# 📚 Documentation Détaillée — Environnement DRL2

> **Version** : 1.0.0
> **Dernière mise à jour** : Juillet 2026
> **Langue** : Français
> **Auteur** : Senior Technical Writer

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture logicielle](#2-architecture-logicielle)
3. [Environnement AgriDroneEnv](#3-environnement-agridroneenv)
4. [Physique du drone (DroneDynamics)](#4-physique-du-drone-dronedynamics)
5. [Fonction de récompense (RewardCalculator)](#5-fonction-de-récompense-rewardcalculator)
6. [Gestion des obstacles (ObstacleManager)](#6-gestion-des-obstacles-obstaclemanager)
7. [Normalisation des observations](#7-normalisation-des-observations)
8. [Pipeline d'entraînement](#8-pipeline-dentraînement)
9. [Guide de configuration](#9-guide-de-configuration)
10. [Glossaire technique](#10-glossaire-technique)

---

## 1. Vue d'ensemble du projet

### 1.1 Objectif et contexte

DRL2 est un **environnement d'apprentissage par renforcement (RL)** conçu pour entraîner un drone agricole hexacoptère à accomplir des missions complexes dans une parcelle cultivée. Le drone doit :

- 🌱 **Arroser** des groupes de plantes répartis aléatoirement
- 💧 **Gérer un réservoir d'eau** limité avec remplissage à une bassine
- 🗺️ **Cartographier** la parcelle en survolant les cellules
- 🦠 **Détecter et traiter** les plantes malades (pulvérisation)
- 🚧 **Éviter** des obstacles sphériques
- 🏠 **Retourner à la base** en cas de batterie faible

Le projet utilise l'API **Gymnasium v1** pour l'interface RL standard et **PyBullet** pour la simulation physique et le rendu 3D.

### 1.2 Stack technique

| Composant | Technologie | Version | Rôle |
|-----------|-------------|---------|------|
| API RL | Gymnasium | ≥ 1.0 | Interface standard (reset, step, spaces) |
| Simulation 3D | PyBullet | ≥ 3.2 | Physique + rendu 3D |
| Calcul numérique | NumPy | ≥ 2.0 | Opérations matricielles |
| Réseaux de neurones | PyTorch | ≥ 2.0 | Agent PPO (acteur-critique) |
| Algorithme RL | PPO | — | Politique-proximale |
| Suivi d'expériences | Weights & Biases | ≥ 0.15 | Logging des métriques |
| Images | Pillow | ≥ 10.0 | Création de GIFs d'évaluation |

### 1.3 Spécification MDP

Le problème est formulé comme un **Processus de Décision Markovien (MDP)** :

```
MDP = (S, A, P, R, γ)

  S : Espace d'observation (39 dimensions par défaut)
  A : Espace d'action continu (6 dimensions dans [-1, 1])
  P : Dynamique physique (DroneDynamics, pas dt = 0.02s)
  R : Fonction de récompense composite (navigation + water task)
  γ : Facteur de discount (0.99 par défaut)
```

---

## 2. Architecture logicielle

### 2.1 Diagramme des fichiers

```
DRL2/
├── environment/                          # 📦 Package principal
│   ├── env.py                            # 🔧 AgriDroneEnv (732 lignes)
│   │   └── FieldCell                     #   └── Cellule de grille du champ
│   ├── minimal_fly_env.py                # 🔧 MinimalFlyEnv (env simplifiée)
│   ├── demo_env.py                       # 🎮 Script de démonstration
│   ├── obstacles.py                      # 🚧 ObstacleManager (60 lignes)
│   ├── physics/
│   │   ├── drone_dynamics.py             # 🚁 DroneDynamics (179 lignes)
│   │   │   ├── DroneState                #   └── État à 12 composantes
│   │   │   └── DroneParams               #   └── Paramètres physiques
│   │   └── wind_model.py                 # 🌬️ WindModel (50 lignes)
│   ├── reward/
│   │   └── reward_function.py            # 💰 RewardCalculator (339 lignes)
│   │       └── RewardConfig              #   └── Configuration paramétrable
│   └── utils/
│       └── normalization.py              # 📐 normalize/denormalize (33 lignes)
├── agent/
│   └── model.py                          # 🧠 Agent PPO (126 lignes)
├── train.py                              # 🏋️ Trainer PPO (96 lignes)
├── eval_agent.py                         # 📊 Évaluation + GIF
├── run_train.py                          # 🚀 Script CLI d'entraînement
├── specify.md                            # 📄 Spécification MDP
├── README.md                             # 📖 README existant
└── docs/
    └── DOCUMENTATION_DETAILED.md          # 📚 Ce document
```

### 2.2 Diagramme du flux de données

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BOUCLE D'ENTRAÎNEMENT PPO                    │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────┐      ┌───────────┐      ┌──────────────┐
  │  Agent   │      │  Trainer  │      │   Buffer     │
  │  (PPO)   │◄────►│  (PPO)    │◄────►│ (rollout)    │
  └────┬─────┘      └─────┬─────┘      └──────┬───────┘
       │                  │                    │
       │  get_distribution│     buffer.insert  │
       │                  │                    │
       ▼                  ▼                    │
  ┌──────────────────────────────────────────┐ │
  │           AgriDroneEnv                   │ │
  │  ┌──────────┐  ┌────────────┐            │ │
  │  │ step()   │  │ _get_obs() │            │ │
  │  └────┬─────┘  └────────────┘            │ │
  │       │                                   │ │
  │       ▼                                   │ │
  │  ┌──────────────────┐                    │ │
  │  │  DroneDynamics    │                    │ │
  │  │  step(flight_act) │                    │ │
  │  └────────┬─────────┘                    │ │
  │           │                              │ │
  │           ▼                              │ │
  │  ┌──────────────────┐                    │ │
  │  │ RewardCalculator  │                    │ │
  │  │ compute_water_task│────────────────────┘ │
  │  └──────────────────┘                      │
  └───────────────────────────────────────────┘

  Données échangées :
    Agent → Env   : action[6] (throttle, roll, pitch, yaw, spray, irrigate)
    Env → Agent   : observation[39] (état normalisé)
    Env → Trainer : reward (float), terminated, truncated, info
```

### 2.3 Flux détaillé d'un pas de simulation

```
┌─────────────────────────────────────────────────────────────────┐
│                     step(action[6])                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. action = np.clip(action, -1, 1)                             │
│  2. prev_dist = _distance_to_nearest_unwatered()                │
│  3. state = dynamics.step(action[:4])                           │
│     ├── throttle → poussée (Newtons)                            │
│     ├── attitude 1er ordre → angles désirés                     │
│     ├── projection monde → accélération                         │
│     ├── traînée aérodynamique                                   │
│     ├── intégration vitesse (Euler)                             │
│     └── intégration position (Euler)                            │
│  4. cell = _get_cell_under_drone()                              │
│     └── mise à jour visuelle (visited, sprayed, watered)        │
│  5. Logique d'arrosage des groupes de plantes                   │
│     └── si action[5]>0, distance<proximité, tank>=conso         │
│  6. Logique de remplissage du réservoir                         │
│     └── si distance_bassine < rayon → tank=100                  │
│  7. curr_dist = _distance_to_nearest_unwatered()                │
│  8. all_watered = np.all(groups[:,3] >= 0.5)                    │
│  9. reward, terms = reward_calc.compute_water_task(...)          │
│ 10. terminated = all_watered                                    │
│ 11. truncated = (step_count >= max_steps)                       │
│ 12. return (obs, reward, terminated, truncated, info)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Environnement AgriDroneEnv

### 3.1 Initialisation (__init__)

L'environnement `AgriDroneEnv` hérite de `BaseEnv` (rl_template) et configure l'ensemble des composants du système.

#### 3.1.1 Structure du dictionnaire de configuration

Le dictionnaire `config` contient **5 sections** obligatoires :

```python
config = {
    "world":        { ... },   # Géométrie du monde
    "drone":        { ... },   # Paramètres physiques du drone
    "simulation":   { ... },   # Paramètres temporels
    "normalization": { ... },  # Limites de normalisation
    "water_task":   { ... },   # Paramètres de la tâche d'irrigation
}
```

#### 3.1.2 Tableau complet des paramètres

| Section | Paramètre | Type | Défaut | Description |
|---------|-----------|------|--------|-------------|
| `world` | `size_x` | float | 60.0 | Largeur du monde (mètres) |
| `world` | `size_y` | float | 60.0 | Profondeur du monde (mètres) |
| `world` | `ground_z` | float | 0.0 | Altitude du sol (mètres) |
| `world` | `size_z` | float | 50.0 | Altitude maximale (mètres) |
| `world` | `field_cells_x` | int | 20 | Nombre de cellules en X |
| `world` | `field_cells_y` | int | 20 | Nombre de cellules en Y |
| `drone` | `dry_mass` | float | 10.0 | Masse à vide (kg) |
| `drone` | `payload_mass_full` | float | 5.0 | Masse max de charge utile (kg) |
| `drone` | `gravity` | float | 9.81 | Accélération gravitationnelle (m/s²) |
| `drone` | `max_thrust_total` | float | 350.0 | Poussée maximale (N) |
| `drone` | `drag_coefficient` | float | 0.08 | Coefficient de traînée linéaire |
| `drone` | `max_tilt_angle_rad` | float | 0.5236 | Angle d'inclinaison max (rad, ≈30°) |
| `drone` | `max_angular_rate` | float | 3.0 | Vitesse angulaire max (rad/s) |
| `drone` | `attitude_time_constant` | float | 0.08 | Constante de temps attitude (s) |
| `drone` | `urdf_path` | str | — | Chemin vers le fichier URDF |
| `simulation` | `dt` | float | 0.02 | Pas de temps (secondes) |
| `simulation` | `max_episode_steps` | int | 1000 | Nombre max de pas par épisode |
| `normalization` | `max_velocity` | float | 50.0 | Vitesse max pour normalisation (m/s) |
| `normalization` | `max_distance` | float | 100.0 | Distance max pour normalisation (m) |
| `water_task` | `basin_position` | list[float] | [15, 15, 0.5] | Position (x,y,z) de la bassine |
| `water_task` | `basin_refill_radius` | float | 3.0 | Rayon de remplissage (m) |
| `water_task` | `water_consumption` | float | 2.0 | Eau consommée par arrosage (unités) |
| `water_task` | `watering_proximity` | float | 2.0 | Distance d'arrosage max (m) |
| `water_task` | `num_plant_groups` | int | 5 | Nombre de groupes de plantes |

#### 3.1.3 Espaces Gymnasium

```python
# Espace d'observation : 17 + 3 + 1 + N*4 dimensions
# Par défaut : 17 + 3 + 1 + 5*4 = 39 dimensions
observation_space = Box(low=-1.0, high=1.0, shape=(39,), dtype=float32)

# Espace d'action : 6 dimensions continues
action_space = Box(low=-1.0, high=1.0, shape=(6,), dtype=float32)
```

#### 3.1.4 État interne initial

| Attribut | Valeur initiale | Description |
|----------|-----------------|-------------|
| `step_count` | 0 | Compteur de pas |
| `battery_level` | 1e6 | Niveau de batterie (quasi-infini) |
| `water_tank_level` | 100.0 | Réservoir d'eau (0-100) |
| `goal_position` | [20, 20, 2] | Objectif de navigation |
| `returning_home` | False | Indicateur retour à la base |

### 3.2 Boucle principale (step)

La méthode `step(action)` est le cœur de l'environnement. Elle orchestre la physique, la logique métier et le calcul de récompense.

#### 3.2.1 Paramètre d'entrée

```python
action : np.ndarray de forme (6,) et valeurs dans [-1, 1]
  action[0] : throttle   → commande de poussée
  action[1] : roll       → consigne d'inclinaison en roulis
  action[2] : pitch      → consigne d'inclinaison en tangage
  action[3] : yaw        → commande de vitesse de lacet
  action[4] : spray      → pulvérisation (activée si > 0)
  action[5] : irrigate   → irrigation (activée si > 0)
```

#### 3.2.2 Valeurs de retour

```python
Returns:
    obs       : np.ndarray (39,) float32 — observation normalisée
    reward    : float — récompense composite
    terminated: bool  — True si mission accomplie (tous groupes arrosés)
    truncated : bool  — True si timeout (step_count >= max_steps)
    info      : dict  — métriques détaillées de l'étape
```

#### 3.2.3 Contenu du dictionnaire `info`

| Clé | Type | Description |
|-----|------|-------------|
| `distance_to_goal` | float | Distance au point objectif |
| `battery_level` | float | Batterie restante |
| `water_tank_level` | float | Niveau du réservoir d'eau |
| `num_unwatered` | int | Groupes restant à arroser |
| `just_watered` | bool | True si arrosage cette étape |
| `just_refilled` | bool | True si remplissage cette étape |
| `all_watered` | bool | True si tous les groupes arrosés |
| `reward_terms` | dict | Décomposition de chaque terme de récompense |

### 3.3 Réinitialisation (reset)

La méthode `reset(seed=None)` prépare un nouvel épisode :

```
┌─────────────────────────────────────────────────────────────┐
│                     reset(seed)                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Réinitialiser les compteurs :                            │
│     step_count=0, battery=1e6, water_tank=100               │
│                                                              │
│  2. Réinitialiser la physique :                              │
│     dynamics.reset(position=[0, 0, 1])                       │
│                                                              │
│  3. Générer la grille du champ (premier appel) :             │
│     ├── Pour chaque cellule (i, j) :                         │
│     │   ├── healthy = True (85%) / False (15%)               │
│     │   └── wet     = True (85%) / False (15%)               │
│     └── Sinon : ré-aleatiser healthy/wet                     │
│                                                              │
│  4. Réinitialiser les flags de traitement :                  │
│     visited=False, sprayed=False, watered=False              │
│                                                              │
│  5. Générer les groupes de plantes :                         │
│     ├── Pour k = 0..num_plant_groups-1 :                     │
│     │   plant_groups[k] = [x_rand, y_rand, z_rand, 0.0]    │
│     └── is_watered = 0.0 (non arrosé)                        │
│                                                              │
│  6. water_tank_level = 100.0                                 │
│                                                              │
│  7. Retourner (_get_obs(), info)                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Espace d'observation (_get_obs)

Le vecteur d'observation est entièrement normalisé dans `[-1, 1]` via la fonction interne `safe_norm()`.

#### 3.4.1 Tableau détaillé de chaque dimension

```
┌─────────────────────────────────────────────────────────────────┐
│                 VECTEUR D'OBSERVATION (39 dims)                  │
├──────────┬───────┬──────────────────────────────────────────────┤
│ Indices  │ Dims  │ Contenu                                      │
├──────────┼───────┼──────────────────────────────────────────────┤
│  0-2     │  3    │ Position du drone (x, y, z)                  │
│  3-5     │  3    │ Vitesse linéaire (vx, vy, vz)                │
│  6-8     │  3    │ Attitude (roll, pitch, yaw)                  │
│  9-11    │  3    │ Vitesses angulaires (roll_rate, pitch_rate,  │
│          │       │   yaw_rate)                                  │
│  12      │  1    │ Distance à l'objectif (norme euclidienne)    │
│  13      │  1    │ Erreur de cap (heading error)                │
│  14      │  1    │ Niveau de batterie                           │
│  15      │  1    │ Distance à l'obstacle le plus proche         │
│  16      │  1    │ Intensité du vent                            │
├──────────┼───────┼──────────────────────────────────────────────┤
│  17-19   │  3    │ Coordonnées de la bassine d'eau (x, y, z)    │
│  20      │  1    │ Niveau du réservoir d'eau (0-100)            │
├──────────┼───────┼──────────────────────────────────────────────┤
│  21-24   │  4    │ Groupe de plantes #0 (x, y, z, is_watered)   │
│  25-28   │  4    │ Groupe de plantes #1 (x, y, z, is_watered)   │
│  29-32   │  4    │ Groupe de plantes #2 (x, y, z, is_watered)   │
│  33-36   │  4    │ Groupe de plantes #3 (x, y, z, is_watered)   │
│  37-40   │  4    │ Groupe de plantes #4 (x, y, z, is_watered)   │
├──────────┴───────┴──────────────────────────────────────────────┤
│  TOTAL : 17 + 3 + 1 + (5 × 4) = 39 dimensions                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.4.2 Plages de normalisation

| Dimension | Plage physique | Plage normalisée | Formule |
|-----------|---------------|-------------------|---------|
| Position (x,y) | [-30, 30] | [-1, 1] | `2*(v-min)/(max-min) - 1` |
| Position (z) | [0, 50] | [-1, 1] | idem |
| Vitesse (vx,vy,vz) | [-50, 50] | [-1, 1] | idem |
| Attitude (roll,pitch) | [-π, π] | [-1, 1] | idem |
| Attitude (yaw) | [-π, π] | [-1, 1] | idem |
| Vitesses angulaires | [-3, 3] | [-1, 1] | idem |
| Distance objectif | [0, 100] | [-1, 1] | idem |
| Batterie | [0, 1e6] | [-1, 1] | idem |
| Réservoir | [0, 100] | [-1, 1] | idem |
| Bassine (x,y) | [-30, 30] | [-1, 1] | idem |
| Bassine (z) | [0, 50] | [-1, 1] | idem |
| Plant groups (x,y) | [-30, 30] | [-1, 1] | idem |
| Plant groups (z) | [0, 50] | [-1, 1] | idem |
| is_watered | 0.0 / 1.0 | -1.0 / +1.0 | `2*v - 1` |

### 3.5 Rendu (render)

L'environnement supporte **3 modes de rendu** :

| Mode | Description | Client PyBullet | Usage |
|------|-------------|-----------------|-------|
| `"human"` | Fenêtre GUI temps réel | GUI (interactif) | Débogage visuel |
| `"rgb_array"` | Image RGB statique (640×480) | DIRECT (headless) | Évaluation, GIF |
| `None` | Pas de rendu | Aucun | Entraînement rapide |

#### 3.5.1 Architecture des clients PyBullet

```
┌──────────────────────────────────────────────────────┐
│                    AgriDroneEnv                       │
│                                                       │
│  ┌────────────────────┐  ┌────────────────────────┐  │
│  │ Client GUI          │  │ Client DIRECT           │  │
│  │ (render="human")    │  │ (render="rgb_array")    │  │
│  │                     │  │                         │  │
│  │  - fenêtre temps    │  │  - pas de fenêtre       │  │
│  │    réel             │  │  - capture d'image      │  │
│  │  - interaction      │  │  - pour GIF/video       │  │
│  │    utilisateur      │  │                         │  │
│  └────────────────────┘  └────────────────────────┘  │
│                                                       │
│  NOTE : clients SÉPARÉS, pas de conflit d'IDs        │
└──────────────────────────────────────────────────────┘
```

#### 3.5.2 Codage couleur des cellules du champ

| État de la cellule | Couleur | Code RGBA |
|---------------------|---------|-----------|
| Saine et humide | 🟢 Vert | [0.2, 0.8, 0.2, 0.8] |
| Malade | 🔴 Rouge | [0.9, 0.1, 0.1, 0.8] |
| Sèche | 🟡 Jaune | [0.8, 0.6, 0.2, 0.8] |
| Malade + Sèche | 🟤 Rouge foncé | [0.8, 0.2, 0.1, 0.8] |
| Pulvérisée / Arrosée | 🔵 Bleu | [0.1, 0.5, 0.9, 0.8] |

---

## 4. Physique du drone (DroneDynamics)

### 4.1 Paramètres physiques (DroneParams)

Le modèle physique est un **corps rigide simplifié** avec commande en attitude du premier ordre.

| Paramètre | Symbole | Défaut | Unité | Description |
|-----------|---------|--------|-------|-------------|
| `dry_mass` | m_dry | 10.0 | kg | Masse à vide du drone |
| `payload_mass` | m_payload | 5.0 | kg | Masse de charge utile (variable) |
| `gravity` | g | 9.81 | m/s² | Accélération gravitationnelle |
| `max_thrust_total` | T_max | 350.0 | N | Poussée maximale totale |
| `drag_coefficient` | k_drag | 0.15 | — | Coefficient de traînée linéaire |
| `max_tilt_angle_rad` | θ_max | 0.5236 | rad | Angle d'inclinaison max (≈30°) |
| `max_angular_rate` | ω_max | 3.0 | rad/s | Vitesse angulaire maximale |
| `attitude_time_constant` | τ | 0.15 | s | Constante de temps du contrôleur |
| `max_velocity` | v_max | 15.0 | m/s | Vitesse maximale du drone |

**Masse totale** :
```
m_total = m_dry + m_payload = 10.0 + 5.0 = 15.0 kg
```

### 4.2 Modèle d'intégration

Le pipeline physique transforme une action normalisée en mouvement 3D via un modèle d'Euler explicite.

#### 4.2.1 Diagramme du pipeline physique

```
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE PHYSIQUE — Un pas de temps (dt)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT : action[4] = [throttle, roll_cmd, pitch_cmd, yaw_cmd]   │
│          valeurs dans [-1, 1]                                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ ÉTAPE 1 : Poussée totale                            │        │
│  │                                                     │        │
│  │   throttle_norm = (throttle_cmd + 1.0) / 2.0       │        │
│  │   thrust = throttle_norm × T_max                    │        │
│  │                                                     │        │
│  │   Exemple : throttle_cmd = 0.4                       │        │
│  │   → throttle_norm = 0.7                             │        │
│  │   → thrust = 0.7 × 350 = 245.0 N                   │        │
│  └─────────────────────────────────────────────────────┘        │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ ÉTAPE 2 : Consignes d'angle désirées                │        │
│  │                                                     │        │
│  │   desired_roll  = roll_cmd  × θ_max                │        │
│  │   desired_pitch = pitch_cmd × θ_max                │        │
│  │   desired_yaw_rate = yaw_cmd × ω_max               │        │
│  └─────────────────────────────────────────────────────┘        │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ ÉTAPE 3 : Réponse d'attitude du 1er ordre           │        │
│  │                                                     │        │
│  │   roll_rate  = (desired_roll  - current_roll)  / τ │        │
│  │   pitch_rate = (desired_pitch - current_pitch) / τ │        │
│  │                                                     │        │
│  │   → clip dans [-ω_max, +ω_max]                     │        │
│  │                                                     │        │
│  │   roll(t+1)  = roll(t)  + roll_rate  × dt          │        │
│  │   pitch(t+1) = pitch(t) + pitch_rate × dt          │        │
│  │   yaw(t+1)   = yaw(t)   + desired_yaw_rate × dt   │        │
│  │                                                     │        │
│  │   → yaw wrappé dans [-π, +π]                       │        │
│  └─────────────────────────────────────────────────────┘        │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ ÉTAPE 4 : Projection de la poussée dans le repère   │        │
│  │           monde (formule complète multirotor)       │        │
│  │                                                     │        │
│  │   ax = (thrust/m) × (cos(r)·sin(p)·cos(y)         │        │
│  │        + sin(r)·sin(y))                             │        │
│  │   ay = (thrust/m) × (cos(r)·sin(p)·sin(y)         │        │
│  │        - sin(r)·cos(y))                             │        │
│  │   az = (thrust/m) × (cos(r)·cos(p)) - g           │        │
│  │                                                     │        │
│  │   where r=roll, p=pitch, y=yaw, m=mass             │        │
│  └─────────────────────────────────────────────────────┘        │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ ÉTAPE 5 : Traînée aérodynamique linéaire            │        │
│  │                                                     │        │
│  │   F_drag = -k_drag × v                              │        │
│  │   ax -= (k_drag / m) × vx                           │        │
│  │   ay -= (k_drag / m) × vy                           │        │
│  │   az -= (k_drag / m) × vz                           │        │
│  └─────────────────────────────────────────────────────┘        │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ ÉTAPE 6 : Intégration de la vitesse (Euler)          │        │
│  │                                                     │        │
│  │   v(t+1) = v(t) + a × dt                           │        │
│  │                                                     │        │
│  │   → clip vitesse dans [0, v_max]                    │        │
│  └─────────────────────────────────────────────────────┘        │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ ÉTAPE 7 : Intégration de la position (Euler)         │        │
│  │                                                     │        │
│  │   x(t+1) = x(t) + vx × dt                          │        │
│  │   y(t+1) = y(t) + vy × dt                          │        │
│  │   z(t+1) = z(t) + vz × dt                          │        │
│  │                                                     │        │
│  │   → si z < 0.05 : z = 0.05, vz = 0 (contact sol)   │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                 │
│  OUTPUT : DroneState (position, vitesse, attitude)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 Équations clés

**Poussée normalisée** :
```
throttle_norm = (throttle_cmd + 1.0) / 2.0
thrust = throttle_norm × T_max
```

**Projection de la poussée (formule exacte multirotor)** :
```
ax = (thrust / m_total) × (cos(roll)·sin(pitch)·cos(yaw)
     + sin(roll)·sin(yaw))

ay = (thrust / m_total) × (cos(roll)·sin(pitch)·sin(yaw)
     - sin(roll)·cos(yaw))

az = (thrust / m_total) × (cos(roll)·cos(pitch)) - g
```

**Traînée aérodynamique** :
```
a_drag = -(k_drag / m_total) × v
```

**Intégration d'Euler explicite** :
```
v(t+1) = v(t) + a_total × dt
x(t+1) = x(t) + v(t+1) × dt
```

**Réponse d'attitude (1er ordre)** :
```
ω_roll  = (desired_roll  - current_roll)  / τ
ω_pitch = (desired_pitch - current_pitch) / τ

roll(t+1)  = roll(t)  + clip(ω_roll,  -ω_max, ω_max) × dt
pitch(t+1) = pitch(t) + clip(ω_pitch, -ω_max, ω_max) × dt
yaw(t+1)   = wrap(yaw(t) + ω_yaw × dt,  -π, π)
```

### 4.3 DroneState

L'état du drone est un dataclass à **12 composantes** :

| Indice | Attribut | Type | Unité | Description |
|--------|----------|------|-------|-------------|
| 0 | `x` | float | m | Position X |
| 1 | `y` | float | m | Position Y |
| 2 | `z` | float | m | Position Z (altitude) |
| 3 | `vx` | float | m/s | Vitesse linéaire X |
| 4 | `vy` | float | m/s | Vitesse linéaire Y |
| 5 | `vz` | float | m/s | Vitesse linéaire Z |
| 6 | `roll` | float | rad | Angle de roulis |
| 7 | `pitch` | float | rad | Angle de tangage |
| 8 | `yaw` | float | rad | Angle de lacet |
| 9 | `roll_rate` | float | rad/s | Vitesse de roulis |
| 10 | `pitch_rate` | float | rad/s | Vitesse de tangage |
| 11 | `yaw_rate` | float | rad/s | Vitesse de lacet |

**Méthodes utilitaires** :
- `position()` → `np.ndarray([x, y, z])`
- `velocity()` → `np.ndarray([vx, vy, vz])`

### 4.4 Conditions de fin

| Condition | Méthode | Seuil | Résultat |
|-----------|---------|-------|----------|
| Hors limites | `is_out_of_bounds()` | Carte [x,y,z] | `out_of_bounds = True` |
| Retourné | `is_flipped()` | |roll/pitch| > 80° | `flipped = True` |
| Contact sol | intégration position | z < 0.05 | z = 0.05, vz = 0 |

---

## 5. Fonction de récompense (RewardCalculator)

> **C'est le composant le plus critique du projet.** La fonction de récompense guide entièrement le comportement de l'agent.

### 5.1 RewardConfig

Le `RewardConfig` est un dataclass qui regroupe **toutes les pondérations et seuils** de la fonction de récompense.

#### 5.1.1 Tableau complet des paramètres

| Catégorie | Paramètre | Défaut | Description |
|-----------|-----------|--------|-------------|
| **Navigation** | `k_progress` | 5.0 | Poids de la progression vers l'objectif |
| | `goal_reward` | 300.0 | Récompense si objectif atteint |
| | `goal_radius` | 0.5 | Rayon de détection objectif (m) |
| | `heading_weight` | 0.5 | Poids de l'erreur de cap |
| | `smooth_weight` | 0.1 | Poids du lissage des actions |
| | `energy_alpha` | 0.05 | Coefficient pénalité énergétique |
| | `stability_weight` | 0.3 | Poids de la stabilité angulaire |
| | `time_penalty` | 0.01 | Pénalité temporelle constante |
| | `collision_penalty` | 300.0 | Pénalité collision |
| | `out_of_bounds_penalty` | 200.0 | Pénalité sortie limites |
| | `flip_penalty` | 250.0 | Pénalité retournement |
| **Agriculture** | `health_bonus` | 2.0 | Bonus par % de maladies traitées |
| | `irrigation_bonus` | 2.0 | Bonus par % de zones arrosées |
| | `exploration_bonus` | 0.5 | Bonus par % de cartographie |
| | `waste_penalty` | 0.5 | Pénalité gaspillage ressources |
| | `low_battery_penalty` | 1.0 | Pénalité batterie faible |
| **Water Task** | `watering_reward` | 5.0 | Récompense arrosage réussi |
| | `refill_reward` | 1.0 | Récompense remplissage réservoir |
| | `refill_threshold` | 98.0 | Seuil anti-farming pour refill |
| | `time_penalty_per_group` | 0.02 | Pénalité par groupe non arrosé |
| | `distance_shaping_reward` | 0.05 | Bonus rapprochement groupe |
| | `mission_complete_reward` | 100.0 | Bonus fin de mission |

### 5.2 compute() — Navigation de base

La méthode `compute()` implémente la récompense de navigation pure, utilisée pour les environnements de vol simple.

#### 5.2.1 Flux de calcul

```
┌─────────────────────────────────────────────────────────────┐
│              compute() — Navigation de base                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ENTRÉES :                                                   │
│    distance_old, distance_new, heading_error, action,        │
│    angular_rates, collided, out_of_bounds, flipped,          │
│    reached_goal                                              │
│                                                              │
│  ┌───────────────────────────────────────────────────┐       │
│  │ terms["progress"]   = k_progress × (d_old - d_new)│      │
│  │ terms["goal"]       = goal_reward si atteint       │      │
│  │ terms["heading"]    = heading_weight × ...          │      │
│  │ terms["smooth"]     = -smooth_weight × ||Δa||      │      │
│  │ terms["energy"]     = -α × throttle_norm²           │      │
│  │ terms["stability"]  = -ω × Σ(ω_i²)                │      │
│  │ terms["time"]       = -time_penalty                 │      │
│  │ terms["collision"]  = -collision_penalty si True    │      │
│  │ terms["out_of_bounds"] = -OOB_penalty si True      │      │
│  │ terms["flip"]       = -flip_penalty si True         │      │
│  └───────────────────────────────────────────────────┘       │
│                                                              │
│  total = Σ(terms)                                            │
│  return (total, terms)                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 5.2.2 Détail de chaque terme

| Terme | Formule | Valeur typique | Objectif |
|-------|---------|----------------|----------|
| `progress` | `k_progress × (d_old - d_new)` | [-25, +25] | Guider vers l'objectif |
| `goal` | `goal_reward` si atteint | 0 ou 300 | Récompense terminal |
| `heading` | `heading_weight × (1 - h/π) × 2 - heading_weight` | [-0.5, +0.5] | Aligner le cap |
| `smooth` | `-smooth_weight × ‖action - prev_action‖` | [-0.3, 0] | Lisser les mouvements |
| `energy` | `-α × ((throttle+1)/2)²` | [-0.05, 0] | Économiser l'énergie |
| `stability` | `-ω × Σ(rates²)` | [-2.7, 0] | Stabiliser l'attitude |
| `time` | `-time_penalty` | -0.01 | Inciter à l'efficacité |
| `collision` | `-collision_penalty` | -300 | Éviter les obstacles |
| `out_of_bounds` | `-OOB_penalty` | -200 | Rester dans la carte |
| `flip` | `-flip_penalty` | -250 | Éviter le retournement |

### 5.3 compute_agri() — Récompense agricole étendue

La méthode `compute_agri()` étend la navigation de base avec des **bonus agricoles** pour guider l'agent vers des tâches spécifiques.

#### 5.3.1 Flux de calcul

```
┌─────────────────────────────────────────────────────────────┐
│              compute_agri() — Récompense étendue             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ENTRÉES :                                                   │
│    (navigation) + battery, maladies, sec, pesticide,         │
│    water_used, visited%, returning_home                      │
│                                                              │
│  ┌───────────────────────────────────────────────────┐       │
│  │ Navigation (identique à compute) :                 │      │
│  │   progress, heading, smooth, energy, stability,    │      │
│  │   time, collision, out_of_bounds, flip              │      │
│  │                                                     │      │
│  │ + Bonus agriculture :                               │      │
│  │   health     = health_bonus × (% maladies)         │      │
│  │   irrigation = irrigation_bonus × (% sec arrosées) │      │
│  │   exploration = exploration_bonus × (% visitées)   │      │
│  │                                                     │      │
│  │ + Pénalités ressources :                            │      │
│  │   waste_pesticide = -waste_penalty si gaspillage    │      │
│  │   waste_water     = -waste_penalty si gaspillage    │      │
│  │   low_battery     = -low_battery_penalty si bat<10  │      │
│  └───────────────────────────────────────────────────┘       │
│                                                              │
│  total = Σ(terms)                                            │
│  return (total, terms)                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 5.3.2 Termes supplémentaires

| Terme | Formule | Condition | Objectif |
|-------|---------|-----------|----------|
| `health` | `health_bonus × (maladies_traitées / total_malades)` | total_malades > 0 | Traiter les maladies |
| `irrigation` | `irrigation_bonus × (sec_arrosées / total_sec)` | total_sec > 0 | Arroser les zones sèches |
| `exploration` | `exploration_bonus × visited_percentage` | — | Cartographier la parcelle |
| `waste_pesticide` | `-waste_penalty` | Pesticide utilisé + aucune maladie restante | Éviter le gaspillage |
| `waste_water` | `-waste_penalty` | Eau utilisée + aucune zone sèche restante | Éviter le gaspillage |
| `low_battery` | `-low_battery_penalty` | batterie < 10 et pas en retour | Retourner à la base |

### 5.4 compute_water_task() — Tâche d'irrigation (LA PLUS IMPORTANTE)

C'est **la méthode utilisée par l'environnement principal** `AgriDroneEnv`. Elle combine 5 termes de récompense pour guider l'agent dans la gestion de l'irrigation.

#### 5.4.1 Flux de calcul détaillé

```
┌─────────────────────────────────────────────────────────────────┐
│            compute_water_task() — Tâche d'irrigation             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ENTRÉES :                                                       │
│    tank_level    : niveau réservoir (0-100)                      │
│    prev_dist     : distance au groupe le plus proche (étape -1)  │
│    curr_dist     : distance au groupe le plus proche (étape t)   │
│    just_watered  : True si arrosage réussi cette étape           │
│    just_refilled : True si remplissage réussi cette étape        │
│    all_watered   : True si tous les groupes sont arrosés         │
│    num_unwatered : nombre de groupes restant à arroser           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ TERME 1 : Arrosage (watering)                          │      │
│  │                                                        │      │
│  │   Si just_watered = True :                             │      │
│  │     terms["watering"] = +5.0                           │      │
│  │   Sinon :                                              │      │
│  │     terms["watering"] = 0.0                            │      │
│  │                                                        │      │
│  │   Condition déclenchante :                             │      │
│  │     1. action[5] > 0 (irrigation activée)              │      │
│  │     2. distance_drone_groupe < watering_proximity (2m) │      │
│  │     3. water_tank_level >= water_consumption (2.0)     │      │
│  │     4. Le groupe n'est pas encore arrosé               │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ TERME 2 : Remplissage (refill)                         │      │
│  │                                                        │      │
│  │   Si just_refilled = True :                            │      │
│  │     terms["refill"] = +1.0                             │      │
│  │   Sinon :                                              │      │
│  │     terms["refill"] = 0.0                              │      │
│  │                                                        │      │
│  │   Condition déclenchante :                             │      │
│  │     1. distance_drone_bassine < basin_refill_radius(3m)│      │
│  │     2. water_tank_level < refill_threshold (98.0)      │      │
│  │        (anti-farming : éviter les remplissages infinis)│      │
│  │     3. Après remplissage, tank = 100.0                 │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ TERME 3 : Pénalité temporelle (time_penalty)           │      │
│  │                                                        │      │
│  │   terms["time_penalty"] = -0.02 × num_unwatered        │      │
│  │                                                        │      │
│  │   Logique :                                            │      │
│  │   - Chaque pas coûte proportionnellement au travail    │      │
│  │     restant                                            │      │
│  │   - Plus il reste de groupes, plus la pénalité est     │      │
│  │     forte                                              │      │
│  │   - Incite l'agent à agir rapidement                   │      │
│  │   - Sans être trop punitif (max = -0.10 pour 5 groupes)│      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ TERME 4 : Shaping de distance (distance_shaping)       │      │
│  │                                                        │      │
│  │   Si num_unwatered > 0 ET prev_dist < ∞ ET curr_dist <∞│     │
│  │     SI curr_dist < prev_dist (rapprochement) :         │      │
│  │       terms["distance_shaping"] = +0.05                │      │
│  │     SINON :                                            │      │
│  │       terms["distance_shaping"] = 0.0                  │      │
│  │                                                        │      │
│  │   Objectif :                                          │      │
│  │   - Guider l'agent vers le groupe le plus proche       │      │
│  │   - Compenser partiellement la pénalité temporelle     │      │
│  │   - Créer un gradient de récompense continu            │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ TERME 5 : Mission accomplie (mission_complete)         │      │
│  │                                                        │      │
│  │   Si all_watered = True :                              │      │
│  │     terms["mission_complete"] = +100.0                 │      │
│  │   Sinon :                                              │      │
│  │     terms["mission_complete"] = 0.0                    │      │
│  │                                                        │      │
│  │   Condition :                                          │      │
│  │     TOUTES les valeurs plant_groups[:,3] >= 0.5        │      │
│  │   Conséquences :                                       │      │
│  │     - terminated = True (arrêt anticipé)               │      │
│  │     - Bonus massif pour récompenser la complétion      │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  RETOUR :                                                        │
│    total = Σ(terms)                                              │
│    return (total, terms_dict)                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.4.2 Tableau récapitulatif des termes

| # | Terme | Formule | Déclencheur | Valeur typique | Impact |
|---|-------|---------|-------------|----------------|--------|
| 1 | `watering` | `+5.0` si arrosage réussi | Groupe arrosé à distance < 2m avec tank suffisant | 0 ou +5.0 | Récompense sparse principale |
| 2 | `refill` | `+1.0` si remplissage | Drone à < 3m de la bassine, tank < 98 | 0 ou +1.0 | Récompense de ravitaillement |
| 3 | `time_penalty` | `-0.02 × num_unwatered` | À chaque pas | -0.10 à 0.0 | Pression temporelle graduée |
| 4 | `distance_shaping` | `+0.05` si rapprochement | curr_dist < prev_dist, groupes restants | 0 ou +0.05 | Guidage continu |
| 5 | `mission_complete` | `+100.0` si tous arrosés | Tous les groupes arrosés | 0 ou +100.0 | Objectif terminal |

#### 5.4.3 Cycle de vie d'un épisode (water task)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CYCLE DE VIE D'UN ÉPISODE                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐                                                        │
│  │  RESET   │  tank=100, N groupes non arrosés, position=[0,0,1]    │
│  └────┬─────┘                                                        │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    BOUCLE DE PAS (max 1000)                   │    │
│  │                                                              │    │
│  │  ┌─────────────────────────────────────────────────────────┐ │    │
│  │  │  ÉTAPE A : Se déplacer vers un groupe non arrosé        │ │    │
│  │  │  - distance_shaping = +0.05 si rapprochement            │ │    │
│  │  │  - time_penalty = -0.02 × N (N = groupes restants)      │ │    │
│  │  └───────────────────────┬─────────────────────────────────┘ │    │
│  │                          │                                    │    │
│  │                          ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐ │    │
│  │  │  ÉTAPE B : Arroser le groupe (si à proximité)           │ │    │
│  │  │  - condition : distance < 2m, action[5]>0, tank ≥ 2    │ │    │
│  │  │  - résultat : group arrosé, tank -= 2                   │ │    │
│  │  │  - récompense : +5.0                                    │ │    │
│  │  └───────────────────────┬─────────────────────────────────┘ │    │
│  │                          │                                    │    │
│  │                          ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐ │    │
│  │  │  ÉTAPE C : Vérifier le réservoir                        │ │    │
│  │  │  Si tank < 2 (plus assez d'eau) :                       │ │    │
│  │  │    → Aller à la bassine pour remplir                    │ │    │
│  │  │    → refill_reward = +1.0 (si tank < 98)                │ │    │
│  │  │    → tank = 100                                         │ │    │
│  │  └───────────────────────┬─────────────────────────────────┘ │    │
│  │                          │                                    │    │
│  │                          ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐ │    │
│  │  │  ÉTAPE D : Vérifier la mission                          │ │    │
│  │  │  Si tous les groupes arrosés :                          │ │    │
│  │  │    → mission_complete = +100.0                          │ │    │
│  │  │    → terminated = True                                  │ │    │
│  │  │    → FIN DE L'ÉPISODE                                   │ │    │
│  │  └───────────────────────┬─────────────────────────────────┘ │    │
│  │                          │                                    │    │
│  │                          └─────► retour à l'ÉTAPE A          │    │
│  │                                  (si pas terminé)             │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  FIN : terminated (mission) ou truncated (timeout)           │    │
│  │  Récompense totale = Σ(récompenses cumulées)                │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

#### 5.4.4 Exemple numérique détaillé

Considérons un scénario avec **5 groupes de plantes** et un réservoir initial de **100.0**.

**Scénario pas à pas :**

```
INITIALISATION :
  plant_groups = [
    [5.0, 5.0, 1.0, 0.0],    # Groupe 0 : non arrosé
    [10.0, 10.0, 1.0, 0.0],  # Groupe 1 : non arrosé
    [15.0, 5.0, 1.0, 0.0],   # Groupe 2 : non arrosé
    [20.0, 15.0, 1.0, 0.0],  # Groupe 3 : non arrosé
    [25.0, 10.0, 1.0, 0.0],  # Groupe 4 : non arrosé
  ]
  water_tank_level = 100.0
  num_unwatered = 5
```

**Étape 1 — Déplacement vers le groupe 0** :
```
  drone_position = [3.0, 4.0, 2.0]
  prev_dist = 2.24 (distance au groupe 0 à l'étape précédente)
  curr_dist = 2.24 (distance au groupe 0 à cette étape)
  
  Résultat :
    watering      = 0.0   (pas d'arrosage)
    refill        = 0.0   (pas de remplissage)
    time_penalty  = -0.02 × 5 = -0.10
    dist_shaping  = 0.0   (pas de rapprochement, dist identique)
    mission       = 0.0
    ─────────────────────
    TOTAL ÉTAPE 1 = -0.10
```

**Étape 2 — Rapprochement du groupe 0** :
```
  drone_position = [4.0, 4.5, 2.0]
  prev_dist = 2.24
  curr_dist = 1.12
  
  Résultat :
    watering      = 0.0
    refill        = 0.0
    time_penalty  = -0.02 × 5 = -0.10
    dist_shaping  = +0.05  (curr_dist < prev_dist ✓)
    mission       = 0.0
    ─────────────────────
    TOTAL ÉTAPE 2 = -0.05
```

**Étape 3 — Arrosage du groupe 0** :
```
  drone_position = [5.0, 5.0, 2.0]  (à < 2m du groupe 0)
  action[5] = 0.8 (irrigation activée)
  water_tank_level = 100.0 ≥ 2.0 ✓
  
  Résultat :
    plant_groups[0,3] = 1.0  (marqué arrosé)
    water_tank_level = 100.0 - 2.0 = 98.0
    num_unwatered = 4
    
    watering      = +5.0  (arrosage réussi ✓)
    refill        = 0.0
    time_penalty  = -0.02 × 4 = -0.08
    dist_shaping  = 0.0   (pas de calcul de distance cette étape)
    mission       = 0.0
    ─────────────────────
    TOTAL ÉTAPE 3 = +4.92
```

**Étape 4 — Retour à la bassine pour remplir** :
```
  drone_position = [15.0, 15.0, 1.0]  (à < 3m de la bassine [15,15,0.5])
  water_tank_level = 98.0 < 98.0 ? Non (98.0 n'est PAS < 98.0)
  
  Résultat :
    just_refilled = False  (seuil non atteint, pas de reward)
    water_tank_level = 100.0  (rempli quand même)
    
    watering      = 0.0
    refill        = 0.0   (tank = 98.0 ≥ 98.0, pas de reward)
    time_penalty  = -0.02 × 4 = -0.08
    dist_shaping  = 0.0
    mission       = 0.0
    ─────────────────────
    TOTAL ÉTAPE 4 = -0.08
    
  NOTE : Si le tank avait été à 95.0 (< 98.0), le refill serait +1.0
```

**Scénario complet avec tous les groupes arrosés** :
```
  APRÈS AVOIR ARROSÉ LES 5 GROUPES :
  
  Étape finale (5ème arrosage) :
    plant_groups = [
      [5.0, 5.0, 1.0, 1.0],   ✓ arrosé
      [10.0, 10.0, 1.0, 1.0], ✓ arrosé
      [15.0, 5.0, 1.0, 1.0],  ✓ arrosé
      [20.0, 15.0, 1.0, 1.0], ✓ arrosé
      [25.0, 10.0, 1.0, 1.0], ✓ arrosé
    ]
    all_watered = True
    
    watering      = +5.0
    refill        = 0.0
    time_penalty  = -0.02 × 0 = 0.0
    dist_shaping  = 0.0
    mission       = +100.0  (TOUS arrosés !)
    ─────────────────────
    TOTAL ÉTAPE FINALE = +105.0
    
    terminated = True → ÉPISODE TERMINÉ
```

**Bilan de l'épisode** :
```
  RÉCOMPENSE CUMULÉE :
    Étape 1  :  -0.10
    Étape 2  :  -0.05
    Étape 3  :  +4.92  (1er arrosage)
    Étapes 4-8 :  -0.08 × 5 = -0.40  (déplacements)
    Étape 9  :  +4.92  (2ème arrosage)
    Étapes 10-14 : -0.08 × 5 = -0.40
    Étape 15 :  +4.92  (3ème arrosage)
    ...etc...
    Étape 25 :  +105.0 (dernier arrosage + mission)
    ─────────────────────
    TOTAL ≈ +120.0 (estimation pour un épisode efficient)
    
  RÉCOMPENSE MAXIMALE POSSIBLE :
    5 arrosages × 5.0  =  +25.0
    1 mission complete = +100.0
    ≈ +125.0 (sans pénalités temporelles)
    
  RÉCOMPENSE MINIMALE (timeout) :
    1000 pas × (-0.10) = -100.0 (si aucun arrosage)
```

### 5.5 Interactions entre les termes

#### 5.5.1 Comment les termes se combinent

```
┌─────────────────────────────────────────────────────────────────┐
│           ÉQUILIBRE ENTRE EXPLORATION ET EXPLOITATION           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EXPLORATION (aller voir les groupes) :                          │
│    distance_shaping (+0.05) compense partiellement               │
│    time_penalty (-0.10 max) → incite à se dépêcher              │
│                                                                  │
│  EXPLOITATION (arroser quand on est près) :                      │
│    watering (+5.0) >> time_penalty (-0.10) → récompense nette    │
│    +4.90 par arrosage réussi                                     │
│                                                                  │
│  GESTION DES RESSOURCES (remplir le réservoir) :                 │
│    refill (+1.0) si tank < 98 → incite au retour à la bassine   │
│    refill est PLUS PETIT que watering (+5.0) →                   │
│    l'agent priorise l'arrosage quand il a de l'eau               │
│                                                                  │
│  OBJECTIF TERMINAL (mission accomplished) :                      │
│    mission_complete (+100.0) >> tout autre terme                 │
│    terminated = True → arrêt anticipé → récompense élevée        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.5.2 Pourquoi ce design équilibre exploration/exploitation

1. **Le distance_shaping (+0.05)** fournit un **gradient continu** qui guide l'agent vers le groupe le plus proche, même sans arrosage immédiat. Cela évite l'exploration aléatoire pure.

2. **La time_penalty proportionnelle (-0.02 × N)** crée une **urgence graduée** : plus il reste de travail, plus la pénalité est forte. Cela empêche l'agent de traîner inutilement.

3. **Le watering (+5.0) dominant** rend l'arrosage **très attractif** comparé aux alternatives. Un arrosage réussi rapporte net +4.90 (après pénalité temporelle), soit 49× la distance_shaping.

4. **Le refill modeste (+1.0)** rend le retour à la bassine **utile mais pas prioritaire**. L'agent doit gérer son réservoir sans en faire sa tâche principale.

5. **Le mission_complete massif (+100.0)** crée un **objectif terminal clair** : l'agent est fortement incentivé à compléter TOUS les arrosages.

#### 5.5.3 Graphique ASCII de la récompense cumulative typique

```
Récompense
cumulative
    │
+120┤                                                    ╱
    │                                                   ╱
+100┤                                                  ╱
    │                                                ╱
 +80┤                                              ╱
    │                                           ╱
 +60┤                                         ╱
    │                                      ╱
 +40┤                                   ╱
    │                                ╱
 +20┤                            ╱╱
    │                      ╱╱╱
   0┤─────────────╱╱╱╱╱╱╱╱
    │         ╱╱╱
 -20┤     ╱╱╱
    │  ╱╱
 -40┤╱
    │
 -60┤
    │
 -80┤
    │
-100┤
    └─────┬──────┬──────┬──────┬──────┬──────┬──────┬───► Temps (pas)
          0    100    200    300    400    500    600

    Légende :
    ╱╱╱ = Récompense croissante (arrosages successifs)
    ─── = Pénalités temporelles (attente entre arrosages)
    ╱   = Saut final (+100 mission_complete)
```

---

## 6. Gestion des obstacles (ObstacleManager)

### 6.1 Architecture

Le `ObstacleManager` gère des **obstacles sphériques** dans l'environnement 3D.

```python
class ObstacleManager:
    obstacles: list[dict]  # [{"pos": np.array([x,y,z]), "radius": r}, ...]
```

### 6.2 Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `min_radius` | 0.5 m | Rayon minimal des obstacles |
| `max_radius` | 2.5 m | Rayon maximal des obstacles |
| `seed` | None | Graine aléatoire pour la reproductibilité |

### 6.3 Méthodes

| Méthode | Description | Retour |
|---------|-------------|--------|
| `generate(count, exclude_zone)` | Génère `count` obstacles aléatoirement en évitant les zones d'exclusion | None |
| `nearest_distance(drone_pos)` | Calcule la distance surface-à-surface au plus proche obstacle | float (+inf si aucun) |
| `check_collision(drone_pos, drone_radius)` | Vérifie si le drone touche un obstacle | bool |

### 6.4 Algorithme de génération

```
┌──────────────────────────────────────────────────────┐
│  generate(count, exclude_zone)                        │
├──────────────────────────────────────────────────────┤
│  Pour chaque obstacle à générer :                     │
│    1. Position aléatoire dans 80% de la carte          │
│    2. Rayon aléatoire dans [min_radius, max_radius]    │
│    3. Vérifier les zones d'exclusion :                  │
│       - Pour chaque zone (pos, radius) :                │
│         distance(pos, zone_pos) < radius + zone_r + 2  │
│         → trop proche, régénérer                        │
│    4. Ajouter l'obstacle à la liste                     │
│  Maximum de tentatives : count × 50                     │
└──────────────────────────────────────────────────────┘
```

### 6.5 Détection de collision

La collision est calculée en **surface-à-surface** :

```python
# Distance centre-à-centre moins les deux rayons
distance_surface = ‖drone_pos - obstacle_pos‖ - obstacle_radius

# Collision si distance_surface < drone_radius (0.3m par défaut)
collision = distance_surface < drone_radius
```

> **Note** : Dans l'environnement principal (`AgriDroneEnv`), les obstacles sont désactivés par défaut (`obstacle_manager.obstacles = []`).

---

## 7. Normalisation des observations

### 7.1 Pourquoi normaliser ?

La normalisation est **critique** pour la convergence de PPO :

1. **Équilibre des gradients** : Sans normalisation, une composante en mètres (0-150) écraserait le gradient d'une composante en radians (-π, π).

2. **Convergence plus rapide** : Le réseau de neurones converge plus vite quand toutes les entrées ont le même ordre de grandeur.

3. **Stabilité de l'estimation de valeur** : Le Critic (estimateur de valeur) est plus stable quand les observations sont bornées dans [-1, 1].

4. **Stabilité du GAE** : Le Generalized Advantage Estimation bénéficie d'entrées normalisées.

### 7.2 Fonctions

#### 7.2.1 normalize()

```python
def normalize(value: float, min_val: float, max_val: float) -> float:
    """Ramène value de [min_val, max_val] vers [-1, 1], avec clipping."""
    if max_val - min_val == 0:
        return 0.0  # Évite la division par zéro
    normalized = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
    return float(np.clip(normalized, -1.0, 1.0))
```

**Exemples** :
```
normalize(30.0, -30.0, 30.0) = 2*(30-(-30))/(30-(-30)) - 1 = 1.0
normalize(0.0,  -30.0, 30.0) = 2*(0-(-30))/(30-(-30)) - 1  = 0.0
normalize(-30.0, -30.0, 30.0) = 2*(-30-(-30))/(30-(-30)) - 1 = -1.0
normalize(50.0, -30.0, 30.0) = 1.0 (clippé, car > 1.0)
```

#### 7.2.2 denormalize()

```python
def denormalize(value: float, min_val: float, max_val: float) -> float:
    """Ramène value de [-1, 1] vers [min_val, max_val]."""
    return min_val + (value + 1.0) / 2.0 * (max_val - min_val)
```

**Exemples** :
```
denormalize(1.0, -30.0, 30.0) = -30 + (1+1)/2 * 60 = 30.0
denormalize(0.0, -30.0, 30.0) = -30 + (0+1)/2 * 60 = 0.0
denormalize(-1.0, -30.0, 30.0) = -30 + (-1+1)/2 * 60 = -30.0
```

### 7.3 Tableau des plages de normalisation

| Composante | Plage physique | Plage norm. | Formule |
|------------|---------------|-------------|---------|
| Position (x, y) | [-size_x/2, size_x/2] | [-1, 1] | `2*(v-min)/(max-min) - 1` |
| Position (z) | [ground_z, size_z] | [-1, 1] | idem |
| Vitesse | [-max_velocity, max_velocity] | [-1, 1] | idem |
| Attitude | [-π, π] | [-1, 1] | idem |
| Rates | [-3.0, 3.0] | [-1, 1] | idem |
| Distance | [0, max_distance] | [-1, 1] | idem |
| Batterie | [0, 1e6] | [-1, 1] | idem |
| Réservoir | [0, 100] | [-1, 1] | idem |
| is_watered | {0.0, 1.0} | {-1.0, +1.0} | `2*v - 1` |

---

## 8. Pipeline d'entraînement

### 8.1 Agent PPO (BaseAgent)

L'agent est un **réseau de neurones acteur-critique** à double tête.

#### 8.1.1 Architecture du réseau

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE DE L'AGENT PPO                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT : observation[39] (vecteur normalisé)                     │
│                                                                  │
│  ┌─────────────────────────────────────┐                        │
│  │        RÉSEAU PARTAGÉ               │                        │
│  │                                     │                        │
│  │  Linear(39 → 128) + Tanh           │                        │
│  │  Linear(128 → 64) + Tanh           │                        │
│  └──────────┬──────────┬───────────────┘                        │
│             │          │                                         │
│    ┌────────┘          └────────┐                                │
│    │                            │                                │
│    ▼                            ▼                                │
│  ┌──────────────┐    ┌──────────────────┐                       │
│  │ TÊTE ACTEUR  │    │ TÊTE CRITIQUE    │                       │
│  │ (Politique)  │    │ (Valeur)         │                       │
│  │              │    │                  │                       │
│  │ Linear(64→64)│    │ Linear(64→64)    │                       │
│  │ + ReLU       │    │ + ReLU           │                       │
│  │ Linear(64→6) │    │ Linear(64→1)     │                       │
│  └──────┬───────┘    └────────┬─────────┘                       │
│         │                     │                                  │
│         ▼                     ▼                                  │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │ mean[6]      │    │ V(s) :       │                           │
│  │ log_std[6]   │    │ scalaire     │                           │
│  │   ↓          │    │              │                           │
│  │ Normal(mean, │    │              │                           │
│  │   std)       │    │              │                           │
│  └──────┬───────┘    └──────────────┘                           │
│         │                                                       │
│         ▼                                                       │
│  OUTPUT : action[6] (échantillonnée de la distribution)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 8.1.2 Initialisation des poids

| Couche | Méthode | Gain | Justification |
|--------|---------|------|---------------|
| Couches cachées | Orthogonale | √2 | Standard pour ReLU |
| Tête acteur | Orthogonale | 0.01 | Distributions étroites au début |
| Tête critique | Orthogonale | 1.0 | Valeurs neutres au début |
| Biais | Constante | 0.0 | Zéro initial |

### 8.2 Trainer PPO (BaseTrain)

Le `Trainer` dans `train.py` étend `BaseTrain` de `rl_template` et surcharge **uniquement 2 méthodes** pour gérer correctement les actions continues 6D.

#### 8.2.1 Surcharges critiques

| Méthode | Problème | Solution |
|---------|----------|----------|
| `rollout_phase()` | `.item()` crash pour actions 6D | `.cpu().numpy()` + `sum(log_probs, dim=-1)` |
| `update_weights()` | Shape mismatch log_prob (B,6) vs advantages (B,) | `new_log_probs.sum(dim=-1)` |

#### 8.2.2 Métriques loggées

| Métrique | Description |
|----------|-------------|
| `train/loss` | Loss totale PPO |
| `train/policy_loss` | Loss de politique (actor) |
| `train/value_loss` | Loss de valeur (critic) |
| `train/entropy` | Entropie de la politique |
| `train/cumulative_reward` | Récompense cumulative de l'épisode |
| `train/learning_rate` | Taux d'apprentissage courant |

#### 8.2.3 Hyperparamètres PPO

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `lr` | 3e-4 | Taux d'apprentissage |
| `gamma` | 0.99 | Facteur de discount |
| `gae_lambda` | 0.95 | Lambda pour le GAE |
| `clip_eps` | 0.2 | Epsilon de clipping PPO |
| `ent_coef` | 0.01 | Coefficient d'entropie |
| `value_coef` | 0.5 | Coefficient de la loss de valeur |
| `rollout_steps` | 128 | Pas par rollout |
| `batch_size` | 64 | Taille des mini-batches |

### 8.3 Boucle d'entraînement

```
┌─────────────────────────────────────────────────────────────────┐
│                 BOUCLE D'ENTRAÎNEMENT PPO                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Pour chaque update (epoch) :                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  1. state, _ = env.reset()                                  │ │
│  │                                                              │ │
│  │  2. ROLLOUT PHASE (128 pas) :                               │ │
│  │     Pour chaque pas :                                        │ │
│  │       a. distribution, value = agent.get_distribution(state) │ │
│  │       b. action = distribution.sample()                      │ │
│  │       c. log_prob = distribution.log_prob(action).sum(-1)    │ │
│  │       d. new_state, reward, term, trunc, info = env.step()   │ │
│  │       e. buffer.insert(state, action, log_prob, reward, ...) │ │
│  │       f. state = new_state                                   │ │
│  │                                                              │ │
│  │  3. UPDATE WEIGHTS (PPO) :                                   │ │
│  │     a. Calculer les avantages (GAE)                          │ │
│  │     b. Pour chaque mini-batch :                              │ │
│  │        i.   new_dist, new_value = agent(state)               │ │
│  │        ii.  ratio = exp(new_log_prob - old_log_prob)         │ │
│  │        iii. clipped = clip(ratio, 1-ε, 1+ε)                 │ │
│  │        iv.  policy_loss = -min(ratio×A, clipped×A)           │ │
│  │        v.   value_loss = (V(s) - returns)²                   │ │
│  │        vi.  entropy_loss = -entropy                          │ │
│  │        vii. loss = policy_loss + 0.5×value_loss + 0.01×entropy│ │
│  │        viii. loss.backward() → optimizer.step()              │ │
│  │                                                              │ │
│  │  4. LOG métriques (wandb optionnel)                          │ │
│  │  5. SAVE model                                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Guide de configuration

### 9.1 Exemple complet de configuration

```python
config = {
    # ============================================
    # GÉOMÉTRIE DU MONDE
    # ============================================
    "world": {
        "size_x": 60.0,         # Largeur de la carte (mètres)
        "size_y": 60.0,         # Profondeur de la carte (mètres)
        "ground_z": 0.0,        # Altitude du sol (mètres)
        "size_z": 50.0,         # Altitude maximale (mètres)
        "field_cells_x": 20,    # Nombre de cellules en X (grille d'affichage)
        "field_cells_y": 20,    # Nombre de cellules en Y (grille d'affichage)
    },
    
    # ============================================
    # PARAMÈTRES PHYSIQUES DU DRONE
    # ============================================
    "drone": {
        "dry_mass": 10.0,                    # Masse à vide (kg)
        "payload_mass_full": 5.0,            # Masse max de charge utile (kg)
        "gravity": 9.81,                     # Gravité (m/s²)
        "max_thrust_total": 350.0,           # Poussée max (N)
        "drag_coefficient": 0.08,            # Traînée aérodynamique
        "max_tilt_angle_rad": 0.5236,        # Inclinaison max (≈30°)
        "max_angular_rate": 3.0,             # Vitesse angulaire max (rad/s)
        "attitude_time_constant": 0.08,      # Réponse du contrôleur (s)
        "urdf_path": "environment/agri_hexacopter_pro.urdf",
    },
    
    # ============================================
    # SIMULATION
    # ============================================
    "simulation": {
        "dt": 0.02,              # Pas de temps (secondes) → 50 Hz
        "max_episode_steps": 1000, # Durée max d'un épisode
    },
    
    # ============================================
    # NORMALISATION
    # ============================================
    "normalization": {
        "max_velocity": 50.0,    # Vitesse max pour normalisation (m/s)
        "max_distance": 100.0,   # Distance max pour normalisation (m)
    },
    
    # ============================================
    # TÂCHE D'IRRIGATION (WATER TASK)
    # ============================================
    "water_task": {
        "basin_position": [15.0, 15.0, 0.5],  # Position (x,y,z) de la bassine
        "basin_refill_radius": 3.0,            # Rayon de remplissage (m)
        "water_consumption": 2.0,              # Eau par arrosage (unités)
        "watering_proximity": 2.0,             # Distance d'arrosage max (m)
        "num_plant_groups": 5,                 # Nombre de groupes de plantes
    },
}
```

### 9.2 Paramètres les plus ajustables

| Paramètre | Impact sur l'entraînement | Recommandation |
|-----------|---------------------------|----------------|
| `num_plant_groups` | Plus de groupes = tâche plus complexe | Commencer à 3, augmenter progressivement |
| `watering_proximity` | Plus petit = plus difficile d'arroser | 2.0m par défaut, réduire pour augmenter la difficulté |
| `basin_refill_radius` | Plus grand = plus facile de remplir | 3.0m par défaut |
| `water_consumption` | Plus grand = réservoir vide plus vite | 2.0 par défaut, ajuster selon num_plant_groups |
| `dt` | Plus petit = plus précis mais plus lent | 0.02s par défaut (50 Hz) |
| `max_episode_steps` | Plus grand = plus de temps pour finir | 1000 par défaut, réduire pour accélérer |
| `k_progress` | Plus grand = exploration plus agressive | 5.0 par défaut |
| `time_penalty_per_group` | Plus grand = plus d'urgence | 0.02 par défaut |

### 9.3 Relation entre réservoir et nombre de groupes

Avec les valeurs par défaut :
- **Réservoir** : 100 unités
- **Consommation** : 2.0 unités/arrosage
- **Capacité max** : 100 / 2.0 = **50 arrosages** sans remplissage
- **Avec 5 groupes** : 5 arrosages nécessaires, réservoir largement suffisant
- **Avec 25 groupes** : 50 arrosages nécessaires, remplissage obligatoire

> **Recommandation** : Pour `num_plant_groups > 20`, augmenter `water_consumption` ou diminuer la capacité initiale du réservoir pour forcer la gestion des ressources.

---

## 10. Glossaire technique

| Terme | Définition |
|-------|------------|
| **MDP** | Markov Decision Process — modèle mathématique pour les problèmes de RL |
| **PPO** | Proximal Policy Optimization — algorithme RL à politique proximale |
| **Actor-Critic** | Architecture avec deux têtes : une politique (acteur) et un estimateur de valeur (critique) |
| **GAE** | Generalized Advantage Estimation — estimation de l'avantage combinant biais et variance |
| **Rollout** | Séquence de pas de simulation collectée pour l'entraînement |
| **Buffer** | Mémoire qui stocke les transitions (état, action, réward, etc.) |
| **Log-prob** | Logarithme de la probabilité de l'action sous la politique courante |
| **Entropy bonus** | Régularisation qui encourage l'exploration en maximisant l'entropie de la politique |
| **Clipping** | Limiter le ratio de probabilités PPO dans [1-ε, 1+ε] pour éviter les mises à jour trop grandes |
| **Domain Randomization** | Variation aléatoire des paramètres physiques pour améliorer la généralisation |
| **Curriculum Learning** | Entraînement progressif de complexité croissante |
| **Water Task** | Tâche principale d'irrigation avec gestion de réservoir |
| **Distance Shaping** | Récompense bonus pour se rapprocher de l'objectif (guidage continu) |
| **Sparse Reward** | Récompense rare et discrète (ex: +5.0 par arrosage réussi) |
| **Time Penalty** | Pénalité par pas de temps, incite à l'efficacité |
| **Refill** | Remplissage du réservoir d'eau à la bassine |
| **Basin** | Point de ravitaillement en eau fixe dans l'environnement |
| **Plant Group** | Groupe de plantes à arroser, défini par position (x,y,z) et statut is_watered |
| **Field Cell** | Cellule individuelle de la grille d'affichage du champ |
| **Euler Integration** | Méthode numérique d'intégration : x(t+1) = x(t) + v(t)×dt |
| **First-Order Attitude** | Modèle simplifié : l'atteinte de l'angle désiré suit une exponentielle decay |
| **Throttle** | Commande de poussée totale (normalisée [-1, 1]) |
| **Spray** | Action de pulvérisation (activée si action[4] > 0) |
| **Irrigate** | Action d'irrigation (activée si action[5] > 0) |
| **Terminated** | Indicateur de fin d'épisode naturelle (mission accomplie) |
| **Truncated** | Indicateur de fin d'épisode artificielle (timeout) |
| **Observation Space** | Espace des états observables par l'agent |
| **Action Space** | Espace des actions disponibles à l'agent |
| **Reward Shaping** | Technique d'ajout de récompenses intermédiaires pour guider l'apprentissage |
| **Anti-farming** | Mécanisme pour éviter l'exploitation abusive d'un terme de récompense |
| **PyBullet** | Moteur de simulation physique et rendu 3D open-source |
| **Gymnasium** | API standard pour les environnements RL (successeur de OpenAI Gym) |
| **URDF** | Unified Robot Description Format — format de description de robots 3D |
| **wandb** | Weights & Biases — plateforme de suivi d'expériences ML |
| **tqdm** | Barre de progression Python pour le suivi visuel de l'entraînement |
| **MLP** | Multi-Layer Perceptron — réseau de neurones entièrement connecté |

---

> **Fin de la documentation détaillée DRL2.**
>
> Ce document couvre l'intégralité de l'architecture, de la physique, de la récompense et du pipeline d'entraînement de l'environnement drone agricole. Pour toute question, consulter le code source dans `environment/` ou le README existant.
