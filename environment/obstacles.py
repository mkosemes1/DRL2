"""
environment/obstacles.py
==========================
Gestion des obstacles sphériques de la parcelle agricole.
"""

import numpy as np


class ObstacleManager:
    def __init__(self, world_bounds: dict, min_radius: float = 0.5,
                 max_radius: float = 2.5, seed: int | None = None):
        self.world_bounds = world_bounds
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.rng = np.random.default_rng(seed)
        self.obstacles: list[dict] = []  # [{"pos": np.array([x,y,z]), "radius": r}, ...]

    def generate(self, count: int, exclude_zone: list[tuple] | None = None) -> None:
        """
        Génère `count` obstacles aléatoirement dans la carte.
        exclude_zone : liste de (position, rayon_exclusion) à éviter
        (typiquement le point de départ et l'objectif).
        """
        self.obstacles = []
        xb, yb = self.world_bounds["x"], self.world_bounds["y"]

        attempts = 0
        while len(self.obstacles) < count and attempts < count * 50:
            attempts += 1
            pos = np.array([
                self.rng.uniform(xb[0] * 0.8, xb[1] * 0.8),
                self.rng.uniform(yb[0] * 0.8, yb[1] * 0.8),
                self.rng.uniform(1.0, 8.0),
            ])
            radius = self.rng.uniform(self.min_radius, self.max_radius)

            if exclude_zone:
                too_close = any(
                    np.linalg.norm(pos[:2] - zone_pos[:2]) < (zone_radius + radius + 2.0)
                    for zone_pos, zone_radius in exclude_zone
                )
                if too_close:
                    continue

            self.obstacles.append({"pos": pos, "radius": radius})

    def nearest_distance(self, drone_pos: np.ndarray) -> float:
        """Distance (surface à surface) au plus proche obstacle. +inf si aucun obstacle."""
        if not self.obstacles:
            return float("inf")
        dists = [
            np.linalg.norm(drone_pos - obs["pos"]) - obs["radius"]
            for obs in self.obstacles
        ]
        return float(min(dists))

    def check_collision(self, drone_pos: np.ndarray, drone_radius: float = 0.3) -> bool:
        return self.nearest_distance(drone_pos) < drone_radius