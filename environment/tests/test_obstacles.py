"""
tests/test_obstacles.py
========================
Tests unitaires pour ObstacleManager.

Vérifie la gestion des obstacles sphériques :
  - Génération de nombre correct
  - Zones d'exclusion
  - Distance au plus proche
  - Détection de collision
  - Bornes de la carte
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from obstacles import ObstacleManager


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def world_bounds():
    """Bornes du monde standard."""
    return {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}


@pytest.fixture
def manager(world_bounds):
    """ObstacleManager avec seed fixe."""
    return ObstacleManager(world_bounds, min_radius=0.5, max_radius=2.5, seed=42)


# ─── Tests obstacles vides ────────────────────────────────────────

class TestEmptyObstacles:
    """Tests quand aucun obstacle n'est généré."""

    def test_obstacles_empty(self, manager):
        """Sans obstacles, nearest_distance = inf."""
        drone_pos = np.array([0.0, 0.0, 1.0])
        dist = manager.nearest_distance(drone_pos)
        assert dist == float("inf")

    def test_obstacles_empty_no_collision(self, manager):
        """Sans obstacles, pas de collision."""
        drone_pos = np.array([0.0, 0.0, 1.0])
        assert manager.check_collision(drone_pos) is False


# ─── Tests génération ─────────────────────────────────────────────

class TestGenerate:
    """Tests pour la génération d'obstacles."""

    def test_obstacles_generate_count(self, manager):
        """generate(N) crée N obstacles."""
        manager.generate(5)
        assert len(manager.obstacles) == 5

    def test_obstacles_generate_zero(self, manager):
        """generate(0) crée aucun obstacle."""
        manager.generate(0)
        assert len(manager.obstacles) == 0

    def test_obstacles_generate_positions_in_bounds(self, manager):
        """Les positions générées sont dans les bornes du monde."""
        manager.generate(20)
        xb, yb = manager.world_bounds["x"], manager.world_bounds["y"]
        for obs in manager.obstacles:
            pos = obs["pos"]
            # Les positions sont dans 80% des bornes (voir code source)
            assert xb[0] * 0.8 <= pos[0] <= xb[1] * 0.8
            assert yb[0] * 0.8 <= pos[1] <= yb[1] * 0.8
            assert 1.0 <= pos[2] <= 8.0

    def test_obstacles_generate_radius_range(self, manager):
        """Les rayons sont dans [min_radius, max_radius]."""
        manager.generate(20)
        for obs in manager.obstacles:
            assert manager.min_radius <= obs["radius"] <= manager.max_radius


# ─── Tests zones d'exclusion ──────────────────────────────────────

class TestExcludeZone:
    """Tests pour les zones d'exclusion."""

    def test_obstacles_generate_exclude(self, manager):
        """La zone d'exclusion empêche les obstacles à proximité."""
        exclude_pos = np.array([0.0, 0.0, 1.0])
        exclude_radius = 10.0
        manager.generate(5, exclude_zone=[(exclude_pos, exclude_radius)])
        # Aucun obstacle ne doit être dans la zone d'exclusion
        for obs in manager.obstacles:
            dist = np.linalg.norm(obs["pos"][:2] - exclude_pos[:2])
            assert dist >= exclude_radius + obs["radius"] + 2.0 - 0.1


# ─── Tests distance au plus proche ────────────────────────────────

class TestNearestDistance:
    """Tests pour nearest_distance()."""

    def test_obstacles_nearest_distance(self):
        """nearest_distance retourne la bonne distance surface-à-surface."""
        bounds = {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}
        mgr = ObstacleManager(bounds, seed=42)
        # Créer un obstacle manuellement
        mgr.obstacles = [{"pos": np.array([5.0, 0.0, 1.0]), "radius": 1.0}]
        drone_pos = np.array([3.0, 0.0, 1.0])
        # Distance euclidienne = 2.0, rayon = 1.0 → surface = 1.0
        dist = mgr.nearest_distance(drone_pos)
        assert dist == pytest.approx(1.0)

    def test_obstacles_nearest_distance_multiple(self):
        """nearest_distance avec plusieurs obstacles retourne le minimum."""
        bounds = {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}
        mgr = ObstacleManager(bounds, seed=42)
        mgr.obstacles = [
            {"pos": np.array([5.0, 0.0, 1.0]), "radius": 1.0},  # surface dist = 4.0
            {"pos": np.array([2.0, 0.0, 1.0]), "radius": 0.5},  # surface dist = 1.5
        ]
        drone_pos = np.array([0.0, 0.0, 1.0])
        dist = mgr.nearest_distance(drone_pos)
        assert dist == pytest.approx(1.5)


# ─── Tests collision ──────────────────────────────────────────────

class TestCollision:
    """Tests pour check_collision()."""

    def test_obstacles_collision_true(self):
        """check_collision retourne True quand le drone est proche."""
        bounds = {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}
        mgr = ObstacleManager(bounds, seed=42)
        mgr.obstacles = [{"pos": np.array([1.0, 0.0, 1.0]), "radius": 1.0}]
        drone_pos = np.array([1.5, 0.0, 1.0])  # dist = 0.5 < drone_radius=0.3 + 0
        # surface-to-surface = 0.5 - 1.0 = -0.5 < 0.3 → collision
        assert mgr.check_collision(drone_pos, drone_radius=0.3) is True

    def test_obstacles_collision_false(self):
        """check_collision retourne False quand le drone est loin."""
        bounds = {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}
        mgr = ObstacleManager(bounds, seed=42)
        mgr.obstacles = [{"pos": np.array([10.0, 0.0, 1.0]), "radius": 1.0}]
        drone_pos = np.array([0.0, 0.0, 1.0])  # dist euclidienne = 10.0
        # surface-to-surface = 9.0 > 0.3 → pas de collision
        assert mgr.check_collision(drone_pos, drone_radius=0.3) is False
