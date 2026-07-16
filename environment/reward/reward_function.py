"""

Fonction de récompense composite, entièrement paramétrable.

R = R_progress + R_goal + R_heading + R_smooth + R_energy
    + R_stability + R_time - R_collision - R_out - R_flip
    + (bonus agricoles : santé, irrigation, exploration, etc.)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardConfig:
    # Termes de navigation
    k_progress: float = 5.0
    goal_reward: float = 300.0
    goal_radius: float = 0.5
    heading_weight: float = 0.5
    smooth_weight: float = 0.1
    energy_alpha: float = 0.05
    stability_weight: float = 0.3
    time_penalty: float = 0.01
    collision_penalty: float = 300.0
    out_of_bounds_penalty: float = 200.0
    flip_penalty: float = 250.0

    # Termes agricoles (nouveaux)
    health_bonus: float = 2.0          # par unité de pourcentage de maladies traitées
    irrigation_bonus: float = 2.0      # par unité de pourcentage de zones sèches arrosées
    exploration_bonus: float = 0.5     # par unité de pourcentage de cartographie
    waste_penalty: float = 0.5         # pénalité pour gaspillage de ressources
    low_battery_penalty: float = 1.0   # pénalité si batterie faible sans retour


class RewardCalculator:
    def __init__(self, config: RewardConfig):
        self.cfg = config
        self._prev_action = np.zeros(4)   # pour le lissage

    def reset(self):
        """Réinitialise la mémoire de l'action précédente."""
        self._prev_action = np.zeros(4)

    # ------------------------------------------------------------------
    # Méthode originale (navigation simple, conservée pour compatibilité)
    def compute(
        self,
        distance_old: float,
        distance_new: float,
        heading_error: float,
        action: np.ndarray,
        angular_rates: np.ndarray,
        collided: bool,
        out_of_bounds: bool,
        flipped: bool,
        reached_goal: bool,
    ) -> tuple[float, dict]:
        """Récompense de base (sans fonctions agricoles)."""
        c = self.cfg
        terms = {}

        terms["progress"] = c.k_progress * (distance_old - distance_new)
        terms["goal"] = c.goal_reward if reached_goal else 0.0
        terms["heading"] = c.heading_weight * (1.0 - heading_error / np.pi) * 2.0 - c.heading_weight

        action_delta = np.linalg.norm(action - self._prev_action)
        terms["smooth"] = -c.smooth_weight * action_delta
        self._prev_action = action.copy()

        throttle_normalized = (action[0] + 1.0) / 2.0
        terms["energy"] = -c.energy_alpha * (throttle_normalized ** 2)
        terms["stability"] = -c.stability_weight * float(np.sum(np.square(angular_rates)))
        terms["time"] = -c.time_penalty
        terms["collision"] = -c.collision_penalty if collided else 0.0
        terms["out_of_bounds"] = -c.out_of_bounds_penalty if out_of_bounds else 0.0
        terms["flip"] = -c.flip_penalty if flipped else 0.0

        total = sum(terms.values())
        return float(total), terms

    # ------------------------------------------------------------------
    # Nouvelle méthode pour l'environnement agricole étendu
    def compute_agri(
        self,
        distance_old: float,
        distance_new: float,
        heading_error: float,
        action: np.ndarray,
        angular_rates: np.ndarray,
        collided: bool,
        out_of_bounds: bool,
        flipped: bool,
        reached_goal: bool,
        battery_level: float,
        maladies_traitees: int,
        sec_arrosees: int,
        total_malades: int,
        total_sec: int,
        pesticide_used: float,
        water_used: float,
        visited_percentage: float,
        returning_home: bool,
    ) -> tuple[float, dict]:
        """
        Récompense étendue pour les tâches agricoles :
          - pulvérisation (traitement des maladies)
          - irrigation (arrosage des zones sèches)
          - cartographie (exploration du champ)
          - gestion de la batterie et retour à la base
        """
        c = self.cfg
        terms = {}

        # --- 1. Progression vers la cible (navigation) ---
        terms["progress"] = c.k_progress * (distance_old - distance_new)

        # --- 2. Bonus de santé (maladies traitées) ---
        if total_malades > 0:
            pct_maladies = maladies_traitees / total_malades
            terms["health"] = c.health_bonus * pct_maladies
        else:
            terms["health"] = 0.0

        # --- 3. Bonus d'irrigation (zones sèches arrosées) ---
        if total_sec > 0:
            pct_sec = sec_arrosees / total_sec
            terms["irrigation"] = c.irrigation_bonus * pct_sec
        else:
            terms["irrigation"] = 0.0

        # --- 4. Bonus d'exploration (cartographie) ---
        terms["exploration"] = c.exploration_bonus * visited_percentage

        # --- 5. Bonus terminal si objectif atteint ---
        terms["goal"] = c.goal_reward if reached_goal else 0.0

        # --- 6. Pénalités pour gaspillage de ressources ---
        # Pulvérisation inutile (plus de maladies ou aucune)
        if pesticide_used > 0 and (total_malades == 0 or maladies_traitees >= total_malades):
            terms["waste_pesticide"] = -c.waste_penalty
        else:
            terms["waste_pesticide"] = 0.0

        # Arrosage inutile (plus de zones sèches ou aucune)
        if water_used > 0 and (total_sec == 0 or sec_arrosees >= total_sec):
            terms["waste_water"] = -c.waste_penalty
        else:
            terms["waste_water"] = 0.0

        # --- 7. Pénalité si batterie faible et pas en retour ---
        if battery_level < 10.0 and not returning_home:
            terms["low_battery"] = -c.low_battery_penalty
        else:
            terms["low_battery"] = 0.0

        # --- 8. Termes de contrôle (identique à compute) ---
        terms["heading"] = c.heading_weight * (1.0 - heading_error / np.pi) * 2.0 - c.heading_weight

        action_delta = np.linalg.norm(action - self._prev_action)
        terms["smooth"] = -c.smooth_weight * action_delta
        self._prev_action = action.copy()

        throttle_normalized = (action[0] + 1.0) / 2.0
        terms["energy"] = -c.energy_alpha * (throttle_normalized ** 2)
        terms["stability"] = -c.stability_weight * float(np.sum(np.square(angular_rates)))
        terms["time"] = -c.time_penalty

        # --- 9. Pénalités terminales (crash, sortie, retournement) ---
        terms["collision"] = -c.collision_penalty if collided else 0.0
        terms["out_of_bounds"] = -c.out_of_bounds_penalty if out_of_bounds else 0.0
        terms["flip"] = -c.flip_penalty if flipped else 0.0

        # Somme de tous les termes
        total = sum(terms.values())
        return float(total), terms