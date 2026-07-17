# Environnement de Drone Agricole pour Reinforcement Learning

## 1. Présentation

Ce dossier implémente un **environnement d'apprentissage par renforcement (RL)** pour un drone agricole hexacoptère. L'environnement `AgriDroneEnv`, qui hérite directement de `gym.Env`, respecte l'API Gymnasium v1 et est conçu pour entraîner un agent à accomplir des missions agricoles complexes :

- **Tâche d'irrigation** : arroser des groupes de plantes avec gestion de réservoir d'eau
- **Survol et cartographie** d'une parcelle
- **Détection et traitement** des plantes malades (pulvérisation)
- **Retour automatique** à la base lorsque la batterie est faible
- **Évitement d'obstacles** et gestion du vent

---

## 2. Architecture

```
environment/
├── agri_drone_env.py            # AgriDroneEnv (gym.Env) — environnement principal
├── minimal_fly_env.py           # MinimalFlyEnv — env simplifiée (vol uniquement)
├── demo_env.py                  # Script de démonstration avec rendu PyBullet
├── obstacles.py                 # ObstacleManager — gestion des obstacles sphériques
├── pybullet_renderer.py         # Rendu 3D PyBullet (découplé de la physique)
├── physics/
│   ├── drone_dynamics.py        # DroneDynamics + DroneParams + DroneState
│   └── wind_model.py            # WindModel — modèle de vent avec rafales
├── reward/
│   └── reward_function.py       # RewardCalculator + RewardConfig
└── utils/
    └── normalization.py         # Fonctions normalize() / denormalize()
```

---

## 3. Classes principales

### 3.1 `AgriDroneEnv` (`agri_drone_env.py`)

Environnement principal héritant de `gym.Env`. Gère la dynamique du drone, la grille du champ agricole, la tâche d'irrigation avec groupes de plantes et réservoir d'eau, ainsi que le rendu PyBullet.

**Paramètres de construction** :
- `config` : dictionnaire de configuration (voir section 6)
- `render_mode` : `"human"`, `"rgb_array"` ou `None`

**Méthodes principales** :
- `reset()` : réinitialise l'environnement, positionne les groupes de plantes aléatoirement
- `step(action)` : exécute un pas de simulation (6 dims : 4 vol + 1 pulvérisation + 1 irrigation)
- `render()` : affiche l'état via PyBullet GUI
- `close()` : libère les ressources PyBullet

### 3.2 `FieldCell` (`agri_drone_env.py`)

Représente une cellule individuelle de la grille du champ agricole. Chaque cellule possède les attributs booléens : `healthy`, `wet`, `sprayed`, `watered`, `visited`.

### 3.3 `DroneDynamics` (`physics/drone_dynamics.py`)

Modèle physique simplifié de corps rigide avec commande en attitude. Intègre la dynamique pas de temps par pas de temps (`dt`) :

- **État** : position (x, y, z), vitesse (vx, vy, vz), attitude (roll, pitch, yaw), vitesses angulaires
- **Commande** : poussée totale (throttle), consignes d'inclinaison (roll, pitch), vitesse de lacet (yaw)
- **Évolution** : modèle du premier ordre pour l'attitude, projection de la poussée dans le repère monde, traînée aérodynamique linéaire, saturation de vitesse

### 3.4 `WindModel` (`physics/wind_model.py`)

Modèle de vent simple avec rafales aléatoires. Le vent est désactivé par défaut (`enabled=False`) et peut être activé pour le Domain Randomization.

### 3.5 `ObstacleManager` (`obstacles.py`)

Gestion des obstacles sphériques de la parcelle. Génère des obstacles aléatoirement dans la carte, calcule la distance de surface la plus proche et détecte les collisions.

### 3.6 `RewardCalculator` (`reward/reward_function.py`)

Calculateur de récompense composite avec trois méthodes :
- `compute()` : récompense de navigation de base
- `compute_agri()` : récompense étendue avec fonctions agricoles
- `compute_water_task()` : récompense dédiée à la tâche d'irrigation

---

## 4. Physique du drone

Le modèle physique est implémenté dans `drone_dynamics.py`. Il s'agit d'un **modèle simplifié de corps rigide** avec commande en attitude.

### 4.1 Équations d'état

L'état du drone est défini par :

- Position : `(x, y, z)`
- Vitesse linéaire : `(vx, vy, vz)`
- Attitude (angles d'Euler) : `(roll, pitch, yaw)`
- Vitesses angulaires : `(roll_rate, pitch_rate, yaw_rate)`

### 4.2 Entrée de commande

L'action (4 premières dimensions) est :

- `throttle` ∈ [-1, 1] → converti en poussée totale `T = ((throttle+1)/2) * T_max`
- `roll_cmd` ∈ [-1, 1] → consigne d'inclinaison en roulis `desired_roll = roll_cmd * max_tilt`
- `pitch_cmd` ∈ [-1, 1] → consigne d'inclinaison en tangage `desired_pitch = pitch_cmd * max_tilt`
- `yaw_cmd` ∈ [-1, 1] → commande de vitesse de lacet `desired_yaw_rate = yaw_cmd * max_angular_rate`

### 4.3 Évolution temporelle

**Attitude** : les angles de roulis et tangage suivent un modèle du premier ordre vers leur consigne :
```
roll_rate = (desired_roll - roll) / tau
pitch_rate = (desired_pitch - pitch) / tau
```

**Accélération** : la poussée est projetée dans le repère monde :
```
ax = (T/m) * (cos(roll)*sin(pitch)*cos(yaw) + sin(roll)*sin(yaw))
ay = (T/m) * (cos(roll)*sin(pitch)*sin(yaw) - sin(roll)*cos(yaw))
az = (T/m) * cos(roll)*cos(pitch) - g
```

---

## 5. Modélisation du champ agricole

Le champ est discrétisé en une grille de cellules (par défaut 20×20). Chaque cellule possède les attributs :

- `healthy` : True si la plante est saine, False si malade
- `wet` : True si le sol est humide, False s'il est sec
- `sprayed` : True si la cellule a déjà été pulvérisée
- `watered` : True si la cellule a déjà été arrosée
- `visited` : True si le drone a survolé cette cellule

À chaque épisode, l'état initial des cellules est généré aléatoirement (15 % de maladies, 15 % de zones sèches).

Le drone peut agir sur une cellule lorsqu'il la survole :
- **Pulvérisation** (`action[4] > 0`) : si la cellule est malade et non traitée, elle devient saine
- **Irrigation** (`action[5] > 0`) : si la cellule est sèche et non arrosée, elle devient humide

---

## 6. Espaces d'observation et d'action

### 6.1 Espace d'observation

Le vecteur d'observation est normalisé dans `[-1, 1]`. Sa dimension totale est **17 + 3 + 1 + N×4** (par défaut N=5 → 39 dimensions).

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

### 6.2 Espace d'action (6 dimensions continues)

| Indice | Nom | Description |
|--------|-----|-------------|
| 0 | `throttle` | Poussée totale (0–100 %) |
| 1 | `roll_cmd` | Consigne d'inclinaison en roulis |
| 2 | `pitch_cmd` | Consigne d'inclinaison en tangage |
| 3 | `yaw_cmd` | Commande de vitesse de lacet |
| 4 | `spray_on` | Pulvérisation (activée si > 0) |
| 5 | `irrigate_on` | Irrigation (activée si > 0) |

---

## 7. Tâche d'irrigation (Water Task)

La tâche principale de l'environnement est d'arroser N groupes de plantes répartis aléatoirement dans la carte.

### Mécanique

1. **Arrosage** : lorsqu'un groupe non arrosé est à moins de 2 m du drone et que le réservoir contient assez d'eau, le groupe est marqué comme arrosé et `water_consumption` unités sont déduites du réservoir.
2. **Remplissage** : lorsque le drone se trouve dans le rayon `basin_refill_radius` de la bassine, le réservoir est remis à 100.
3. **Fin de mission** : l'épisode se termine avec succès (`terminated=True`) lorsque tous les groupes sont arrosés.

### Configuration

```python
"water_task": {
    "basin_position": [15.0, 15.0, 0.5],  # Position (x, y, z) de la bassine
    "basin_refill_radius": 3.0,            # Rayon de remplissage (m)
    "water_consumption": 2.0,              # Eau consommée par arrosage
    "num_plant_groups": 5,                 # Nombre de groupes de plantes
}
```

---

## 8. Fonction de récompense

La récompense est composite et calculée par `RewardCalculator`. Pour la tâche d'irrigation, elle utilise la méthode `compute_water_task()` :

| Terme | Valeur par défaut | Description |
|-------|-------------------|-------------|
| `watering_reward` | +5.0 | Arrosage réussi d'un groupe |
| `refill_reward` | +1.0 | Remplissage du réservoir à la bassine |
| `time_penalty_per_group` | −0.02 | Pénalité par groupe non arrosé |
| `distance_shaping_reward` | +0.05 | Bonus si le drone se rapproche du groupe le plus proche |
| `mission_complete_reward` | +100.0 | Bonus terminal quand tous les groupes sont arrosés |

---

## 9. Rendu 3D

Le rendu 3D est assuré par PyBullet, intégré directement dans `AgriDroneEnv` :

- Charge le modèle URDF du drone (`agri_hexacopter_pro.urdf`)
- Affiche la grille du champ avec des couleurs dynamiques selon l'état des cellules
- Anime les hélices proportionnellement au throttle
- Est **découplé** de la physique d'entraînement : le rendu n'est activé que si `render_mode="human"`

### Couleurs des cellules

| État | Couleur |
|------|---------|
| Malade et sec | Rouge foncé |
| Malade | Rouge |
| Sec | Jaune |
| Pulvérisée ou arrosée | Bleu |
| Saine et humide | Vert |

---

## 10. Utilisation

### Lancement de la démonstration

```bash
cd environment
uv run demo_env.py          # Env principale avec grille de champs
uv run minimal_fly_env.py   # Env simplifiée (vol uniquement)
```

### Tests manuels

```bash
cd environment
python test_dynamics.py     # Script de test de la dynamique (pas pytest)
```

### Dépendances

```
numpy
gymnasium
pybullet
stable-baselines3
torch
wandb
```
