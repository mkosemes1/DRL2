# Environnement de Drone Agricole pour Reinforcement Learning

## 1. Présentation

Ce dépôt implémente un **environnement d’apprentissage par renforcement (RL)** pour un drone agricole hexacoptère. L’environnement, nommé `AgriDroneEnv`, respecte l’API Gymnasium v1 et est conçu pour entraîner un agent à accomplir des missions agricoles complexes :

- **Survol et cartographie** d’une parcelle
- **Détection et traitement** des plantes malades (pulvérisation)
- **Irrigation** des zones sèches
- **Retour automatique** à la base lorsque la batterie est faible
- **Évitement d’obstacles** et gestion du vent

L’environnement a été étendu par rapport à une version initiale pour intégrer ces fonctionnalités agricoles, tout en conservant une physique réaliste et une récompense composite bien calibrée.

---

## 2. Architecture générale

L’environnement est organisé en modules :
environment/
├── agri_drone_env.py # Environnement principal (hérite de BaseEnv)
├── demo_env.py # Script de démonstration avec vol en zigzag
├── agri_hexacopter_pro.urdf # Modèle URDF du drone (pour rendu 3D)
├── obstacles.py # Gestion des obstacles sphériques
├── pybullet_renderer.py # Rendu 3D temps réel (PyBullet)
├── physics/
│ ├── drone_dynamics.py # Dynamique du drone (équations physiques)
│ └── wind_model.py # Modèle de vent avec rafales
├── reward/
│ └── reward_function.py # Récompense composite (navigation + agricole)
└── utils/
└── normalization.py # Fonctions de normalisation [-1,1]


L’environnement hérite de la classe abstraite `BaseEnv` (fournie) qui impose les méthodes `reset()`, `step()`, `close()`. Il est donc compatible avec tous les algorithmes RL (PPO, SAC, etc.).

---

## 3. Physique du drone

Le modèle physique est implémenté dans `drone_dynamics.py`. Il s’agit d’un **modèle simplifié de corps rigide** avec commande en attitude.

### 3.1 Équations d’état

L’état du drone est défini par :

- Position : `(x, y, z)`
- Vitesse linéaire : `(vx, vy, vz)`
- Attitude (angles d’Euler) : `(roll, pitch, yaw)`
- Vitesses angulaires : `(roll_rate, pitch_rate, yaw_rate)`

### 3.2 Entrée de commande

L’action (4 premières dimensions) est :

- `throttle` ∈ [-1, 1] → converti en poussée totale `T = ((throttle+1)/2) * T_max`
- `roll_cmd` ∈ [-1, 1] → consigne d’inclinaison en roulis `desired_roll = roll_cmd * max_tilt`
- `pitch_cmd` ∈ [-1, 1] → consigne d’inclinaison en tangage `desired_pitch = pitch_cmd * max_tilt`
- `yaw_cmd` ∈ [-1, 1] → commande de vitesse de lacet `desired_yaw_rate = yaw_cmd * max_angular_rate`

### 3.3 Évolution temporelle

**Attitude** : les angles de roulis et tangage suivent un modèle du premier ordre vers leur consigne :
roll_rate = (desired_roll - roll) / tau
pitch_rate = (desired_pitch - pitch) / tau

**Accélération** : la poussée est projetée dans le repère monde :
ax = (T/m) * (cos(roll)sin(pitch)cos(yaw) + sin(roll)sin(yaw))
ay = (T/m) * (cos(roll)sin(pitch)sin(yaw) - sin(roll)cos(yaw))
az = (T/m) * cos(roll)*cos(pitch) - g

## 4. Modélisation du champ agricole

Le champ est discrétisé en une grille de cellules (ex : 20×20). Chaque cellule possède les attributs :

- `healthy` : booléen (True = plante saine, False = malade)
- `wet` : booléen (True = humide, False = sec)
- `sprayed` : booléen (True = déjà pulvérisée)
- `watered` : booléen (True = déjà arrosée)
- `visited` : booléen (True = survolée au moins une fois)

À chaque épisode, l’état initial des cellules est généré aléatoirement avec les probabilités `disease_probability` et `dry_probability`.

Le drone peut agir sur une cellule lorsqu’il la survole (position projetée sur le plan horizontal) :

- **Pulvérisation** (`spray_on > 0`) : si la cellule est malade et non encore traitée, elle devient saine et la quantité de pesticide diminue.
- **Irrigation** (`irrigate_on > 0`) : si la cellule est sèche et non encore arrosée, elle devient humide et le niveau d’eau diminue.

**Cartographie** : chaque cellule survolée est marquée `visited = True`.

---

## 5. Espaces d’observation et d’action

### 5.1 Espace d’observation (24 dimensions, normalisées [-1,1])

| Indice | Description |
|--------|-------------|
| 0–2    | Position (x, y, z) |
| 3–5    | Vitesse linéaire (vx, vy, vz) |
| 6–8    | Attitude (roll, pitch, yaw) |
| 9–11   | Vitesses angulaires (roll_rate, pitch_rate, yaw_rate) |
| 12     | Distance à l’objectif (cible courante) |
| 13     | Erreur de cap (heading) |
| 14     | Niveau de batterie |
| 15     | Distance à l’obstacle le plus proche |
| 16     | Intensité du vent |
| 17     | Niveau de pesticide restant |
| 18     | Niveau d’eau restant |
| 19     | Pourcentage de maladies traitées |
| 20     | Pourcentage de zones sèches arrosées |
| 21     | Pourcentage de cartographie (cellules visitées) |
| 22     | Distance à la base |
| 23     | Booléen : retour à la base (1) ou non (-1) |

### 5.2 Espace d’action (6 dimensions continues)

- `[throttle, roll_cmd, pitch_cmd, yaw_cmd]` : commandes de vol (∈ [-1,1])
- `spray_on` : ∈ [-1,1] ; interprété comme activé si > 0
- `irrigate_on` : ∈ [-1,1] ; interprété comme activé si > 0

---

## 6. Fonction de récompense

La récompense est composite et conçue pour guider l’agent vers un comportement efficace et sécurisé. Elle est calculée à chaque pas par la méthode `compute_agri()` de `reward_function.py`.

### 6.1 Termes de navigation (hérités)

| Terme | Formule |
|-------|---------|
| **Progression** | `k_progress * (distance_old - distance_new)` |
| **Cap** | `heading_weight * (2*(1 - heading_error/π) - 1)` |
| **Lissage** | `-smooth_weight * ||action - action_prev||` |
| **Énergie** | `-energy_alpha * throttle_normalized²` |
| **Stabilité** | `-stability_weight * sum(angular_rates²)` |
| **Temps** | `-time_penalty` (constant) |
| **Pénalités terminales** | `-collision_penalty` (crash), `-out_of_bounds_penalty`, `-flip_penalty` |

### 6.2 Termes agricoles (ajoutés)

| Terme | Formule |
|-------|---------|
| **Santé** | `health_bonus * (maladies_traitees / max(1, total_malades))` |
| **Irrigation** | `irrigation_bonus * (sec_arrosees / max(1, total_sec))` |
| **Exploration** | `exploration_bonus * visited_percentage` |
| **Gaspillage pesticide** | `-waste_penalty` si pulvérisation sans maladie |
| **Gaspillage eau** | `-waste_penalty` si irrigation sans zone sèche |
| **Batterie faible** | `-low_battery_penalty` si batterie < 10 Wh et pas en retour |
| **Objectif atteint** | `+goal_reward` (mission accomplie) |

La récompense totale est la somme de tous ces termes.

---

## 7. Curriculum Learning et Domain Randomization

Pour faciliter l’apprentissage et améliorer la généralisation, l’environnement supporte :

- **Nombre d’obstacles** variable (via `curriculum_stage["obstacle_count"]`)
- **Activation du vent** (via `curriculum_stage["wind_enabled"]`)
- **Randomisation de la masse** du drone (masse de la cuve variable)

Ces paramètres peuvent être modifiés par un `CurriculumManager` externe entre les épisodes.

---

## 8. Rendu 3D

Le rendu 3D est assuré par `PyBulletRenderer`, qui :

- Charge le modèle URDF du drone (`agri_hexacopter_pro.urdf`)
- Affiche le drone, les obstacles, l’objectif et la base
- Anime les hélices proportionnellement au throttle
- Est **découplé** de la physique d’entraînement : le rendu n’est activé que si `render_mode="human"` et ne recalcule aucune physique, ce qui permet un entraînement rapide.

---

## 9. Installation et utilisation

### 9.1 Dépendances

Le fichier `requirements.txt` (ou `pyproject.toml`) doit contenir :
numpy
gymnasium
pybullet
matplotlib (optionnel pour logs)
stable-baselines3 (pour l’entraînement)

### 9.2 Lancement de la démonstration

```bash
cd environment
uv run demo_env.py