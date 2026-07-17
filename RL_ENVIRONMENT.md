# Guide Technique — Environnement RL Agricole

Ce document explique en détail l'architecture, la physique, les espaces d'action/observation, la fonction de récompense, et l'intégration avec le framework `rl_template`. Il sert de référence technique pour comprendre et contribuer au code.

---

## 1. Vue d'ensemble

Le projet implémente un environnement Gymnasium pour entraîner un agent RL (PPO) à piloter un drone agricole hexacoptère. L'agent doit accomplir des missions :
- **Cartographie** d'une parcelle (survol de chaque cellule)
- **Pulvérisation** des plantes malades
- **Irrigation** des zones sèches
- **Retour à la base** quand la batterie est faible
- **Évitement d'obstacles**

### Fichiers clés

| Fichier | Rôle | Statut |
|---------|------|--------|
| `environment/agri_drone_env.py` | Environnement principal (`AgriDroneEnv`) | ✅ Implémenté |
| `environment/minimal_fly_env.py` | Environnement minimal de vol (`MinimalFlyEnv`) | ✅ Implémenté |
| `environment/physics/drone_dynamics.py` | Modèle physique du drone (`DroneDynamics`) | ✅ Implémenté |
| `environment/physics/wind_model.py` | Modèle de vent (`WindModel`) | ✅ Implémenté |
| `environment/obstacles.py` | Gestionnaire d'obstacles (`ObstacleManager`) | ✅ Implémenté |
| `environment/reward/reward_function.py` | Fonction de récompense (`RewardCalculator`) | ✅ Implémenté |
| `environment/utils/normalization.py` | Utilitaires de normalisation | ✅ Implémenté |
| `environment/pybullet_renderer.py` | Rendu 3D PyBullet | ✅ Implémenté |
| `agent/model.py` | Agent RL (réseau de neurones) | ⚠️ Skeleton partiellement cassé |
| `train.py` | Pipeline d'entraînement | ❌ Vide |

---

## 2. Modèle Physique — DroneDynamics

### 2.1 État du drone (DroneState)

L'état est un dataclass avec 12 composantes :

```
Position :     x, y, z          (mètres)
Vitesse :      vx, vy, vz       (m/s)
Attitude :     roll, pitch, yaw  (radians)
Vitesses ang : roll_rate, pitch_rate, yaw_rate  (rad/s)
```

Méthodes utilitaires :
- `position()` → `np.array([x, y, z])`
- `velocity()` → `np.array([vx, vy, vz])`

### 2.2 Paramètres physiques (DroneParams)

```
dry_mass           = 10.0 kg     (masse à vide)
payload_mass       = 5.0 kg      (masse cargo, variable 0 → payload_mass_full)
gravity            = 9.81 m/s²
max_thrust_total   = 350.0 N     (poussée maximale)
drag_coefficient   = 0.15        (traînée linéaire)
max_tilt_angle_rad = 0.5236      (30°, inclinaison max)
max_angular_rate   = 3.0 rad/s   (vitesse angulaire max)
attitude_time_const= 0.15        (constante de temps du contrôleur)
max_velocity       = 15.0 m/s    (vitesse max)
```

La masse totale est `dry_mass + payload_mass` (propriété `total_mass`).

### 2.3 Boucle de simulation (DroneDynamics.step())

À chaque pas de temps `dt`, l'action `[throttle, roll_cmd, pitch_cmd, yaw_cmd]` est appliquée :

```
1. Poussée totale :
   throttle_normalized = (throttle + 1) / 2     # [-1,1] → [0,1]
   thrust = throttle_normalized * max_thrust_total

2. Consignes d'attitude :
   desired_roll  = roll_cmd * max_tilt_angle_rad
   desired_pitch = pitch_cmd * max_tilt_angle_rad
   desired_yaw_rate = yaw_cmd * max_angular_rate

3. Réponse du 1er ordre (contrôleur bas-niveau) :
   roll_rate  = (desired_roll - current_roll) / tau
   pitch_rate = (desired_pitch - current_pitch) / tau
   # saturé à ±max_angular_rate

4. Mise à jour des angles :
   roll  += roll_rate * dt
   pitch += pitch_rate * dt
   yaw   += desired_yaw_rate * dt
   yaw = wrap(yaw, -π, π)

5. Projection de la poussée dans le repère monde :
   ax = (thrust/m) * (cos(roll)*sin(pitch)*cos(yaw) + sin(roll)*sin(yaw))
   ay = (thrust/m) * (cos(roll)*sin(pitch)*sin(yaw) - sin(roll)*cos(yaw))
   az = (thrust/m) * cos(roll)*cos(pitch) - gravity

6. Traînée linéaire :
   ax -= (drag_coefficient / m) * vx
   ay -= (drag_coefficient / m) * vy
   az -= (drag_coefficient / m) * vz

7. Intégration (Euler explicite) :
   vx += ax * dt
   vy += ay * dt
   vz += az * dt
   # Saturation de vitesse
   x += vx * dt
   y += vy * dt
   z += vz * dt
   # Sol : z >= 0.05
```

### 2.4 Limites de sécurité

- `is_out_of_bounds()` : vérifie si le drone sort des limites du monde
- `is_flipped()` : retour True si roll ou pitch dépasse 80°
- Sol : z ne descend jamais sous 0.05m (collision avec le sol)

---

## 3. Espaces d'action et d'observation

### 3.1 Espace d'action

**AgriDroneEnv** : `Box(-1, 1, shape=(6,), float32)`

| Index | Dimension | Interprétation |
|-------|-----------|----------------|
| 0 | `throttle` | Poussée [-1,1] → [0%, 100%] |
| 1 | `roll_cmd` | Consigne d'inclinaison latérale [-1,1] |
| 2 | `pitch_cmd` | Consigne d'inclinaison avant [-1,1] |
| 3 | `yaw_cmd` | Vitesse de rotation [-1,1] |
| 4 | `spray_on` | Pulvérisation si > 0 |
| 5 | `irrigate_on` | Irrigation si > 0 |

**MinimalFlyEnv** : `Box(-1, 1, shape=(4,), float32)`

| Index | Dimension | Interprétation |
|-------|-----------|----------------|
| 0 | `throttle` | Poussée |
| 1 | `roll_cmd` | Inclinaison latérale |
| 2 | `pitch_cmd` | Inclinaison avant |
| 3 | `yaw_cmd` | Rotation |

### 3.2 Espace d'observation

**AgriDroneEnv** : `Box(-1, 1, shape=(17,), float32)` — normalisé [-1, 1]

| Index | Description | Normalisation |
|-------|-------------|---------------|
| 0–2 | Position (x, y, z) | [world_min, world_max] → [-1,1] |
| 3–5 | Vitesse (vx, vy, vz) | [-max_v, max_v] → [-1,1] |
| 6–8 | Attitude (roll, pitch, yaw) | [-π, π] → [-1,1] |
| 9–11 | Vitesses angulaires | [-3, 3] → [-1,1] |
| 12 | Distance à l'objectif | [0, max_distance] → [-1,1] |
| 13 | Erreur de cap | fixé à 0 (non implémenté) |
| 14 | Niveau de batterie | [0, 1e6] → [-1,1] |
| 15 | Distance obstacle le plus proche | fixé à 0 (non implémenté) |
| 16 | Intensité du vent | fixé à 0 (vent désactivé) |

> ⚠️ **Note** : `environment/README.md` décrit un espace à 24 dimensions et un parent `BaseEnv`. Le code utilise réellement 17 dimensions et `gym.Env` directement. Fiez-vous au code.

**MinimalFlyEnv** : `Box(-inf, inf, shape=(9,), float32)` — non normalisé

| Index | Description |
|-------|-------------|
| 0–2 | Position (x, y, z) |
| 3–5 | Vitesse (vx, vy, vz) |
| 6–8 | Attitude (roll, pitch, yaw) |

---

## 4. Champ Agricole

### 4.1 Grille de cellules

Le champ est discrétisé en une grille `field_cells_x × field_cells_y` (par défaut 20×20). Chaque cellule (`FieldCell`) possède :

```
healthy  : bool   (True = plante saine)
wet      : bool   (True = humide)
sprayed  : bool   (True = déjà pulvérisée)
watered  : bool   (True = déjà arrosée)
visited  : bool   (True = survolée)
```

### 4.2 Initialisation

À chaque épisode, les cellules sont générées aléatoirement :
- 15% de chance d'être malade (`healthy = False`)
- 15% de chance d'être sèche (`wet = False`)
- Les flags `visited`, `sprayed`, `watered` sont réinitialisés

### 4.3 Interaction drone-cellule

La méthode `_get_cell_under_drone()` projette la position du drone sur la grille :
```python
i = int((x - x_min) / (x_max - x_min) * nx)
j = int((y - y_min) / (y_max - y_min) * ny)
```

Si `action[4] > 0` (spray) et la cellule est malade → elle devient saine
Si `action[5] > 0` (irrigate) et la cellule est sèche → elle devient humide

### 4.4 Rendu visuel

En mode `render_mode="human"`, PyBullet affiche :
- Cellules vertes = saines et humides
- Cellules rouges = malades
- Cellules jaunes = sèches
- Cellules bleues = traitées (pulvérisées ou arrosées)

---

## 5. Fonction de Récompense

### 5.1 RewardConfig

```python
# Navigation
k_progress        = 5.0    # Bonus de progression vers l'objectif
goal_reward       = 300.0  # Bonus si objectif atteint
goal_radius       = 0.5    # Rayon de considération de l'objectif
heading_weight    = 0.5    # Poids du cap
smooth_weight     = 0.1    # Pénalité de lissage d'action
energy_alpha      = 0.05   # Pénalité d'énergie
stability_weight  = 0.3    # Pénalité d'instabilité
time_penalty      = 0.01   # Pénalité temporelle
collision_penalty = 300.0  # Pénalité de collision
out_of_bounds_penalty = 200.0
flip_penalty      = 250.0

# Agricole
health_bonus      = 2.0    # Bonus par % de maladies traitées
irrigation_bonus  = 2.0    # Bonus par % de zones sèches arrosées
exploration_bonus = 0.5    # Bonus par % de cartographie
waste_penalty     = 0.5    # Pénalité pour gaspillage
low_battery_penalty = 1.0  # Pénalité batterie faible sans retour
```

### 5.2 Méthode compute() — Navigation simple

```
R = k_progress * (d_old - d_new)           # Progression
  + goal_reward si objectif atteint
  + heading_weight * (1 - heading_error/π)  # Cap
  - smooth_weight * ||Δaction||             # Lissage
  - energy_alpha * throttle²                # Énergie
  - stability_weight * Σ(ω²)               # Stabilité
  - time_penalty                            # Temps
  - collision_penalty si collision          # Crash
  - out_of_bounds_penalty si hors limites   # Limites
  - flip_penalty si retourné                # Retournement
```

### 5.3 Méthode compute_agri() — Récompense étendue

Ajoute les termes agricoles :
```
  + health_bonus * (maladies_traitées / total_malades)
  + irrigation_bonus * (sec_arrosées / total_sec)
  + exploration_bonus * visited_percentage
  - waste_penalty si pulvérisation inutile
  - waste_penalty si irrigation inutile
  - low_battery_penalty si batterie < 10 Wh et pas en retour
```

---

## 6. Intégration avec rl_template

### 6.1 Architecture du framework

`rl_template` fournit les classes abstraites et le training loop :

```
rl_template/
├── agent.py      → BaseAgent (ABC + nn.Module)
│                   forward(state) → (logits, value)
│                   get_distribution(state) → (dist, value)
│                   get_action(state, action=None) → (action, log_prob, entropy, value)
│
├── env.py        → BaseEnv (ABC + gym.Env)
│                   reset() → (obs, info)
│                   step(action) → (obs, reward, terminated, truncated, info)
│                   close()
│
├── train.py      → BaseTrain (ABC)
│                   rollout_phase(state) → collect experience
│                   update_weights(step) → PPO update
│                   save_model()
│
├── common.py     → Buffer (rollout buffer pré-alloué en numpy)
│                   insert(state, action, log_prob, reward, value, done)
│                   insert_returns(returns, adv)
│                   get_all() → tuple de 8 tensors PyTorch
│
├── config.py     → PPOConfig (lr, gamma, gae_lambda, clip_eps, ent_coef, value_coef)
│                   TrainConfig (model_name, device, timestamp, batch_size, rollout_steps)
│
└── algorithms/ppo/ppo.py → PPOTrainer
                             compute_gae(rewards, values, last_value, dones)
                             update(memory, total_steps, step, batch_size, epochs)
                             lr_decay(lr, total_steps, step)
```

### 6.2 Intégration actuelle

**`agent/model.py`** importe `BaseAgent` et commence un skeleton :

```python
from torch import nn
from torch.distribitions import Normal   # ← typo : "distribitions"
from rl_template import BaseAgent

class Agent(BaseAgent)                    # ← manque ":" et le corps
```

**Ce qui manque pour que ça fonctionne :**
1. Corriger `torch.distribitions` → `torch.distributions`
2. Implémenter `forward(state)` → `(logits, value)`
3. Implémenter `get_distribution(state)` → `(Normal, value)` pour actions continues
4. Ajouter un réseau (MLP) pour le policy head et le value head

### 6.3 Problème de compatibilité : actions continues

**Problème critique** : `Buffer.insert()` attend `action: int` (scalaire, discret). Or `AgriDroneEnv.action_space` est `Box(-1, 1, shape=(6,))` — **continu**.

Le `Buffer` et le `PPOTrainer` tels qu'écrits ne fonctionnent que pour les espaces d'action discrets. Pour ce projet, il faudra adapter :
- Le `Buffer` pour accepter des actions continues (tableaux numpy)
- Le `PPOTrainer.update()` pour gérer les distributions continues (Normal au lieu de Categorical)
- L'agent à implémenter une tête de politique qui sort `(mean, std)` pour une distribution `Normal`

---

## 7. Statut d'implémentation

### ✅ Fonctionnel
- Modèle physique DroneDynamics avec intégration Euler
- Modèle de vent WindModel (gustes, domain randomization)
- Gestionnaire d'obstacles ObstacleManager (sphériques, exclusion zones)
- Fonction de récompense composite (navigation + agricole)
- Normalisation des observations [-1, 1]
- Rendu 3D PyBullet (découplé de la physique)
- Environments Gymnasium (AgriDroneEnv, MinimalFlyEnv)

### ⚠️ Partiellement implémenté
- `agent/model.py` : skeleton cassé (typo + classe vide)

### ❌ Non implémenté
- `train.py` : fichier vide
- Adaptation du Buffer pour actions continues
- Boucle d'entraînement complète
- Système de curriculum learning
- Gestion de la batterie réaliste (consommation variable)

### 🐛 Bugs connus
- `agent/model.py` ligne 2 : `torch.distribitions` → `torch.distributions`
- `agent/model.py` ligne 5 : `class Agent(BaseAgent)` → manque `:` et corps
- `environment/README.md` : décrit 24-dim obs et `BaseEnv` parent (incorrect)
- `reward/physics/` : duplicate de `physics/` avec `DEBUG=True` (ne pas utiliser)
- `requirements.txt` : pinned `rl-template==0.1.1` mais v0.1.2 installé

---

## 8. Commandes Utiles

```bash
# Démo environnement principal (nécessite display)
uv run environment/demo_env.py

# Démo vol minimal (nécessite display)
uv run environment/minimal_fly_env.py

# Test physique manuel (pas pytest)
cd environment && python test_dynamics.py

# Installer les dépendances
uv pip install -r requirements.txt

# Vérifier l'installation de rl_template
uv run python -c "from rl_template import BaseAgent; print('OK')"
```
