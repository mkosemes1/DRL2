"""
physics/wind_model.py
======================
Modèle de vent simple avec rafales aléatoires.
Utilisé pour le Domain Randomization et le Curriculum Learning
(le vent n'est activé qu'à partir du stage_3).
"""

import numpy as np


class WindModel:
    def __init__(self, max_speed: float = 5.0, gust_probability: float = 0.05,
                 enabled: bool = False, seed: int | None = None):
        self.max_speed = max_speed
        self.gust_probability = gust_probability
        self.enabled = enabled
        self.rng = np.random.default_rng(seed)
        self.current_wind = np.zeros(2)  # (wx, wy)

    def reset(self):
        if self.enabled:
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(0, self.max_speed * 0.5)
            self.current_wind = speed * np.array([np.cos(angle), np.sin(angle)])
        else:
            self.current_wind = np.zeros(2)

    def step(self) -> np.ndarray:
        """Retourne le vecteur vent (wx, wy) courant et le fait évoluer légèrement."""
        if not self.enabled:
            return self.current_wind

        # Rafale aléatoire
        if self.rng.random() < self.gust_probability:
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(0, self.max_speed)
            self.current_wind = speed * np.array([np.cos(angle), np.sin(angle)])
        else:
            # dérive légère
            self.current_wind += self.rng.normal(0, 0.05, size=2)
            speed = np.linalg.norm(self.current_wind)
            if speed > self.max_speed:
                self.current_wind *= self.max_speed / speed

        return self.current_wind

    def magnitude(self) -> float:
        return float(np.linalg.norm(self.current_wind))