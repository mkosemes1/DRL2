"""
tests/test_demo_config.py
=========================
Tests unitaires pour la configuration du fichier demo_env.py.

Vérifie la structure, les clés et les valeurs du dictionnaire
de configuration utilisé pour la démonstration de l'environnement
AgriDroneEnv.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agri_drone_env import AgriDroneEnv


@pytest.fixture
def demo_config():
    """Retourne la configuration telle que définie dans demo_env.py.

    La configuration est reproduite ici directement car le script
    demo_env.py contient du code d'exécution qui ne peut pas être
    importé sans initializer PyBullet.
    """
    return {
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
            "urdf_path": os.path.join(os.path.dirname(__file__), "..", "agri_hexacopter_pro.urdf")
        },
        "simulation": {"dt": 0.02, "max_episode_steps": 1000},
        "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
        "water_task": {
            "basin_position": [15.0, 15.0, 0.5],
            "basin_refill_radius": 3.0,
            "water_consumption": 2.0,
            "num_plant_groups": 5,
        },
    }


# ─── Tests Structure de la Config ───────────────────────────────

class TestDemoConfigStructure:
    """Vérifie que la config de démo contient toutes les clés obligatoires."""

    def test_demo_config_structure(self, demo_config):
        """Vérifie que la config contient les clés world, drone, simulation, normalization, water_task."""
        required_keys = {"world", "drone", "simulation", "normalization", "water_task"}
        assert required_keys.issubset(set(demo_config.keys())), (
            f"Clés manquantes : {required_keys - set(demo_config.keys())}"
        )

    def test_demo_config_world_keys(self, demo_config):
        """Vérifie que la section world contient toutes les clés nécessaires."""
        world_cfg = demo_config["world"]
        required_world_keys = {"size_x", "size_y", "ground_z", "size_z"}
        assert required_world_keys.issubset(set(world_cfg.keys())), (
            f"Clés world manquantes : {required_world_keys - set(world_cfg.keys())}"
        )

    def test_demo_config_drone_keys(self, demo_config):
        """Vérifie que la section drone contient les clés de base."""
        drone_cfg = demo_config["drone"]
        assert "dry_mass" in drone_cfg
        assert "payload_mass_full" in drone_cfg

    def test_demo_config_simulation_keys(self, demo_config):
        """Vérifie que la section simulation contient dt et max_episode_steps."""
        sim_cfg = demo_config["simulation"]
        assert "dt" in sim_cfg
        assert "max_episode_steps" in sim_cfg

    def test_demo_config_normalization_keys(self, demo_config):
        """Vérifie que la section normalization contient max_velocity et max_distance."""
        norm_cfg = demo_config["normalization"]
        assert "max_velocity" in norm_cfg
        assert "max_distance" in norm_cfg


# ─── Tests World Bounds ─────────────────────────────────────────

class TestDemoConfigWorldBounds:
    """Vérifie que la configuration world produit les bonnes limites."""

    def test_demo_config_world_bounds(self, demo_config):
        """Vérifie les dimensions du monde de la config de démo."""
        world_cfg = demo_config["world"]
        expected_x = (-world_cfg["size_x"] / 2, world_cfg["size_x"] / 2)
        expected_y = (-world_cfg["size_y"] / 2, world_cfg["size_y"] / 2)
        expected_z = (world_cfg["ground_z"], world_cfg["size_z"])

        e = AgriDroneEnv(demo_config, render_mode=None)
        assert e.world_bounds["x"] == expected_x
        assert e.world_bounds["y"] == expected_y
        assert e.world_bounds["z"] == expected_z
        e.close()

    def test_demo_config_field_cells(self, demo_config):
        """Vérifie que field_cells_x et field_cells_y sont correctement lus."""
        world_cfg = demo_config["world"]
        e = AgriDroneEnv(demo_config, render_mode=None)
        assert e.field_size == (world_cfg["field_cells_x"], world_cfg["field_cells_y"])
        e.close()

    def test_demo_config_max_episode_steps(self, demo_config):
        """Vérifie que max_episode_steps est correctement lu."""
        sim_cfg = demo_config["simulation"]
        e = AgriDroneEnv(demo_config, render_mode=None)
        assert e.max_steps == sim_cfg["max_episode_steps"]
        e.close()

    def test_demo_config_dt(self, demo_config):
        """Vérifie que le pas de temps dt est correctement lu."""
        sim_cfg = demo_config["simulation"]
        e = AgriDroneEnv(demo_config, render_mode=None)
        assert e.dt == pytest.approx(sim_cfg["dt"])
        e.close()


# ─── Tests Water Task Config ────────────────────────────────────

class TestDemoConfigWaterTask:
    """Vérifie que la configuration water_task contient tous les champs requis."""

    def test_demo_config_water_task(self, demo_config):
        """Vérifie que water_task contient basin_position, basin_refill_radius, water_consumption, num_plant_groups."""
        water_cfg = demo_config["water_task"]
        required_keys = {"basin_position", "basin_refill_radius", "water_consumption", "num_plant_groups"}
        assert required_keys.issubset(set(water_cfg.keys())), (
            f"Clés water_task manquantes : {required_keys - set(water_cfg.keys())}"
        )

    def test_demo_config_basin_position_values(self, demo_config):
        """Vérifie que la position de la bassine est correcte."""
        water_cfg = demo_config["water_task"]
        e = AgriDroneEnv(demo_config, render_mode=None)
        np.testing.assert_array_almost_equal(e.water_basin_position, water_cfg["basin_position"])
        e.close()

    def test_demo_config_num_plant_groups(self, demo_config):
        """Vérifie que num_plant_groups affecte la dimension d'observation."""
        water_cfg = demo_config["water_task"]
        e = AgriDroneEnv(demo_config, render_mode=None)
        expected_dim = 17 + 3 + 1 + water_cfg["num_plant_groups"] * 4
        assert e.observation_space.shape == (expected_dim,)
        assert e.num_plant_groups == water_cfg["num_plant_groups"]
        e.close()

    def test_demo_config_basin_refill_radius(self, demo_config):
        """Vérifie que le rayon de remplissage de la bassine est correct."""
        water_cfg = demo_config["water_task"]
        e = AgriDroneEnv(demo_config, render_mode=None)
        assert e.basin_refill_radius == water_cfg["basin_refill_radius"]
        e.close()

    def test_demo_config_water_consumption(self, demo_config):
        """Vérifie que la consommation d'eau est correcte."""
        water_cfg = demo_config["water_task"]
        e = AgriDroneEnv(demo_config, render_mode=None)
        assert e.water_consumption == water_cfg["water_consumption"]
        e.close()
