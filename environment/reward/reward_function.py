"""
reward/reward_function.py
===========================
Fonction de récompense composite, entièrement paramétrable.

R = R_progress + R_goal + R_heading + R_smooth + R_energy
    + R_stability + R_time - R_collision - R_out - R_flip

Chaque terme est expliqué en détail dans le README.
Une mauvaise conception de la reward est la cause n°1 d'échec
d'entraînement PPO : signal trop clairsemé (sparse), échelles
incohérentes entre les termes, ou récompenses contradictoires
empêchent le gradient de politique de converger vers un
comportement utile.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardConfig:
    k_progress: float = 5.0          # gain sur le rapprochement de la cible
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


class RewardCalculator:
    def __init__(self, config: RewardConfig):
        self.cfg = config
        self._prev_action = np.zeros(4)

    def reset(self):
        self._prev_action = np.zeros(4)

    def compute(
        self,
        distance_old: float,
        distance_new: float,
        heading_error: float,       # angle entre le cap du drone et la direction cible, en radians [0, pi]
        action: np.ndarray,
        angular_rates: np.ndarray,  # [roll_rate, pitch_rate, yaw_rate]
        collided: bool,
        out_of_bounds: bool,
        flipped: bool,
        reached_goal: bool,
    ) -> tuple[float, dict]:
        """
        Retourne (reward_totale, dict_détaillé) pour permettre le logging
        de chaque composante séparément dans TensorBoard.
        """
        c = self.cfg
        terms = {}

        # --- Progression vers la cible : encourage tout rapprochement ---
        terms["progress"] = c.k_progress * (distance_old - distance_new)

        # --- Bonus terminal si la cible est atteinte ---
        terms["goal"] = c.goal_reward if reached_goal else 0.0

        # --- Cap vers la cible : 1 quand aligné, -1 quand opposé ---
        terms["heading"] = c.heading_weight * (1.0 - heading_error / np.pi) * 2.0 - c.heading_weight

        # --- Lissage : pénalise les changements brusques d'action (jerk) ---
        action_delta = np.linalg.norm(action - self._prev_action)
        terms["smooth"] = -c.smooth_weight * action_delta
        self._prev_action = action.copy()

        # --- Énergie : pénalise le throttle élevé (consommation ∝ throttle²) ---
        throttle_normalized = (action[0] + 1.0) / 2.0
        terms["energy"] = -c.energy_alpha * (throttle_normalized ** 2)

        # --- Stabilité : pénalise les vitesses angulaires élevées (oscillations) ---
        terms["stability"] = -c.stability_weight * float(np.sum(np.square(angular_rates)))

        # --- Coût temporel : pousse à finir la mission rapidement ---
        terms["time"] = -c.time_penalty

        # --- Pénalités terminales fortes ---
        terms["collision"] = -c.collision_penalty if collided else 0.0
        terms["out_of_bounds"] = -c.out_of_bounds_penalty if out_of_bounds else 0.0
        terms["flip"] = -c.flip_penalty if flipped else 0.0

        total = sum(terms.values())
        return float(total), terms