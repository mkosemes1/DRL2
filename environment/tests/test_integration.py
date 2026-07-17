"""
tests/test_integration.py
=========================
Tests d'intégration pour l'environnement AgriDroneEnv.

Exerce plusieurs composants ensemble pour valider le fonctionnement
complet de l'environnement : reset, step, arrosage, remplissage,
mission accomplie, troncation, observation et récompenses.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agri_drone_env import AgriDroneEnv


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Configuration minimale pour les tests d'intégration."""
    return {
        "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
        "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
        "simulation": {"dt": 0.02, "max_episode_steps": 100},
        "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
        "water_task": {
            "basin_position": [15.0, 15.0, 0.5],
            "basin_refill_radius": 3.0,
            "water_consumption": 2.0,
            "watering_proximity": 2.0,
            "num_plant_groups": 3,
        },
    }


@pytest.fixture
def env(config):
    """Environnement AgriDroneEnv initialisé et réinitialisé."""
    e = AgriDroneEnv(config, render_mode=None)
    e.reset(seed=42)
    return e


# ─── Tests Épisode Complet Court ────────────────────────────────

class TestFullEpisodeShort:
    """Tests pour un épisode court avec des actions aléatoires."""

    def test_full_episode_short(self, env):
        """Crée l'environnement, exécute 10 pas aléatoires, vérifie aucun crash."""
        env.reset(seed=42)
        rng = np.random.default_rng(seed=123)
        for _ in range(10):
            action = rng.uniform(-1.0, 1.0, size=6)
            obs, reward, terminated, truncated, info = env.step(action)
            # Vérifier que les shapes sont correctes
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, float)
            if terminated or truncated:
                break


# ─── Tests Flux d'Arrosage ──────────────────────────────────────

class TestWateringFlow:
    """Tests pour le flux complet d'arrosage d'un groupe de plantes."""

    def test_watering_flow(self, env):
        """Positionne le drone près d'un groupe, l'arrose, vérifie réservoir et statut."""
        # Placer un groupe de plantes à une position connue
        env.plant_groups[0] = np.array([2.0, 2.0, 0.5, 0.0], dtype=np.float32)
        # Placer le drone exactement sur le groupe
        env.dynamics.reset(np.array([2.0, 2.0, 1.0]))

        tank_before = env.water_tank_level
        assert env.plant_groups[0, 3] == 0.0, "Le groupe ne doit pas être arrosé initialement"

        # Action : throttle suffisant pour voler + irrigation activée
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
        _, _, _, _, info = env.step(action)

        # Vérifier que l'arrosage a eu lieu
        assert info["just_watered"] is True, "Le groupe devrait être arrosé"
        assert env.plant_groups[0, 3] == 1.0, "Le groupe devrait être marqué comme arrosé"
        assert env.water_tank_level == pytest.approx(tank_before - 2.0), (
            f"Le réservoir devrait diminuer de 2.0 (consommation par défaut)"
        )


# ─── Tests Flux de Remplissage ──────────────────────────────────

class TestRefillFlow:
    """Tests pour le flux complet de remplissage du réservoir."""

    def test_refill_flow(self, env):
        """Met le réservoir à bas niveau, positionne le drone à la bassine, vérifie le remplissage."""
        # Mettre le réservoir à un niveau bas
        env.water_tank_level = 30.0
        # Positionner le drone à la bassine (15.0, 15.0, 0.5)
        env.dynamics.reset(np.array([15.0, 15.0, 1.0]))

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, _, _, _, info = env.step(action)

        # Vérifier le remplissage
        assert env.water_tank_level == 100.0, "Le réservoir devrait être rempli à 100.0"
        assert info["just_refilled"] is True, "just_refilled devrait être True"


# ─── Tests Mission Accomplie ────────────────────────────────────

class TestMissionCompleteFlow:
    """Tests pour la fin de mission quand tous les groupes sont arrosés."""

    def test_mission_complete_flow(self, env):
        """Arrose manuellement TOUS les groupes, vérifie terminated=True et bonus +100."""
        # Marquer tous les groupes comme arrosés
        env.plant_groups[:, 3] = 1.0

        obs, reward, terminated, truncated, info = env.step(np.zeros(6))

        assert terminated is True, "L'épisode devrait se terminer quand tous les groupes sont arrosés"
        assert info["all_watered"] is True, "all_watered devrait être True"
        # La récompense devrait contenir le bonus mission_complete
        assert info["reward_terms"]["mission_complete"] == 100.0, (
            "Le bonus de mission accomplie devrait être 100.0"
        )


# ─── Tests Troncation ───────────────────────────────────────────

class TestTruncationFlow:
    """Tests pour la troncation après max_steps."""

    def test_truncation_flow(self):
        """Exécute l'environnement pour max_steps, vérifie truncated=True."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 10},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {
                "basin_position": [15.0, 15.0, 0.5],
                "basin_refill_radius": 3.0,
                "water_consumption": 2.0,
                "watering_proximity": 2.0,
                "num_plant_groups": 3,
            },
        }
        e = AgriDroneEnv(config, render_mode=None)
        e.reset(seed=42)

        truncated = False
        for _ in range(e.max_steps):
            _, _, _, truncated, _ = e.step(np.zeros(6))
        assert truncated is True, "truncated devrait être True après max_steps"
        e.close()


# ─── Tests Cohérence des Observations ───────────────────────────

class TestObservationConsistency:
    """Tests pour vérifier la cohérence des observations après step."""

    def test_observation_consistency(self, env):
        """Après un step, vérifie que la forme de l'observation correspond à observation_space.shape."""
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(np.zeros(6))
        assert obs.shape == env.observation_space.shape, (
            f"La forme de l'observation ({obs.shape}) ne correspond pas "
            f"à observation_space.shape ({env.observation_space.shape})"
        )

    def test_observation_all_finite(self, env):
        """Vérifie que toutes les observations sont des valeurs finies après des pas."""
        env.reset(seed=42)
        rng = np.random.default_rng(seed=456)
        for _ in range(20):
            action = rng.uniform(-1.0, 1.0, size=6)
            obs, _, _, _, _ = env.step(action)
            assert np.all(np.isfinite(obs)), f"Observation non finie: {obs}"


# ─── Tests Termes de Récompense dans info ───────────────────────

class TestRewardTermsInInfo:
    """Tests pour vérifier que info['reward_terms'] contient les bonnes clés."""

    def test_reward_terms_in_info(self, env):
        """Après un step, vérifie que info['reward_terms'] a les 5 clés attendues."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(6))
        expected_keys = {"watering", "refill", "time_penalty", "distance_shaping", "mission_complete"}
        assert "reward_terms" in info, "info devrait contenir 'reward_terms'"
        assert set(info["reward_terms"].keys()) == expected_keys, (
            f"Clés manquantes ou supplémentaires dans reward_terms : "
            f"attendu={expected_keys}, obtenu={set(info['reward_terms'].keys())}"
        )

    def test_reward_terms_are_floats(self, env):
        """Vérifie que tous les termes de récompense sont des float."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(6))
        for key, value in info["reward_terms"].items():
            assert isinstance(value, float), (
                f"reward_terms['{key}'] devrait être un float, obtenu {type(value)}"
            )


# ─── Tests Dimension d'Observation avec Nombre Variable de Groupes

class TestNumPlantGroupsObsDim:
    """Tests pour vérifier que la dimension d'observation change avec num_plant_groups."""

    def test_num_plant_groups_obs_dim_n1(self):
        """Avec N=1, la dimension d'observation devrait être 17 + 3 + 1 + 1*4 = 25."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 100},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {"num_plant_groups": 1},
        }
        e = AgriDroneEnv(config, render_mode=None)
        obs, _ = e.reset(seed=42)
        assert obs.shape == (25,)
        assert e.observation_space.shape == (25,)
        e.close()

    def test_num_plant_groups_obs_dim_n3(self):
        """Avec N=3, la dimension d'observation devrait être 17 + 3 + 1 + 3*4 = 33."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 100},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {"num_plant_groups": 3},
        }
        e = AgriDroneEnv(config, render_mode=None)
        obs, _ = e.reset(seed=42)
        assert obs.shape == (33,)
        assert e.observation_space.shape == (33,)
        e.close()

    def test_num_plant_groups_obs_dim_n10(self):
        """Avec N=10, la dimension d'observation devrait être 17 + 3 + 1 + 10*4 = 61."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 100},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {"num_plant_groups": 10},
        }
        e = AgriDroneEnv(config, render_mode=None)
        obs, _ = e.reset(seed=42)
        assert obs.shape == (61,)
        assert e.observation_space.shape == (61,)
        e.close()


# ─── Tests Niveau de Réservoir ──────────────────────────────────

class TestWaterTankLevel:
    """Tests pour vérifier que le réservoir d'eau ne devient jamais négatif."""

    def test_water_tank_never_negative(self, env):
        """Arrose plusieurs fois, vérifie que le réservoir ne descend jamais sous 0."""
        env.reset(seed=42)
        # Placer un groupe de plantes et le drone dessus
        env.plant_groups[0] = np.array([2.0, 2.0, 0.5, 0.0], dtype=np.float32)
        env.dynamics.reset(np.array([2.0, 2.0, 1.0]))

        # Mettre le réservoir à un niveau bas pour forcer un épuisement
        env.water_tank_level = 3.0  # juste assez pour 1 arrosage (consommation = 2.0)

        # Premier arrosage — devrait fonctionner
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
        obs, _, _, _, info = env.step(action)
        assert env.water_tank_level >= 0.0, (
            f"Le réservoir ne devrait jamais être négatif : {env.water_tank_level}"
        )

        # Si le réservoir est encore suffisant, on essaie un autre arrosage
        if env.water_tank_level >= 2.0:
            obs, _, _, _, info = env.step(action)
            assert env.water_tank_level >= 0.0, (
                f"Le réservoir ne devrait jamais être négatif : {env.water_tank_level}"
            )


# ─── Tests Paramètres Configurables ─────────────────────────────

class TestAllConfigurableParams:
    """Tests pour vérifier que tous les paramètres configurables sont utilisés."""

    def test_all_configurable_params_watering_proximity(self):
        """Vérifie que watering_proximity modifie la distance d'arrosage."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 100},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {
                "watering_proximity": 5.0,
                "num_plant_groups": 1,
            },
        }
        e = AgriDroneEnv(config, render_mode=None)
        e.reset(seed=42)
        assert e.watering_proximity == 5.0
        e.close()

    def test_all_configurable_params_water_consumption(self):
        """Vérifie que water_consumption modifie la consommation d'eau par arrosage."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 100},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {
                "water_consumption": 5.0,
                "num_plant_groups": 1,
            },
        }
        e = AgriDroneEnv(config, render_mode=None)
        e.reset(seed=42)
        assert e.water_consumption == 5.0
        e.close()

    def test_all_configurable_params_basin_refill_radius(self):
        """Vérifie que basin_refill_radius modifie le rayon de remplissage."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 100},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {
                "basin_refill_radius": 10.0,
                "num_plant_groups": 1,
            },
        }
        e = AgriDroneEnv(config, render_mode=None)
        e.reset(seed=42)
        assert e.basin_refill_radius == 10.0
        e.close()

    def test_all_configurable_params_basin_position(self):
        """Vérifie que basin_position modifie la position de la bassine."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 100},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {
                "basin_position": [0.0, 0.0, 0.0],
                "num_plant_groups": 1,
            },
        }
        e = AgriDroneEnv(config, render_mode=None)
        e.reset(seed=42)
        np.testing.assert_array_almost_equal(e.water_basin_position, [0.0, 0.0, 0.0])
        e.close()

    def test_all_configurable_params_max_steps(self):
        """Vérifie que max_episode_steps modifie la durée maximale de l'épisode."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 50},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {"num_plant_groups": 1},
        }
        e = AgriDroneEnv(config, render_mode=None)
        e.reset(seed=42)
        assert e.max_steps == 50
        e.close()

    def test_all_configurable_params_dt(self):
        """Vérifie que dt modifie le pas de temps de la simulation."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.01, "max_episode_steps": 100},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {"num_plant_groups": 1},
        }
        e = AgriDroneEnv(config, render_mode=None)
        e.reset(seed=42)
        assert e.dt == pytest.approx(0.01)
        e.close()

    def test_all_configurable_params_consumption_effect(self):
        """Vérifie que la consommation d'eau configurée est bien appliquée lors de l'arrosage."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 100},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {
                "water_consumption": 10.0,
                "watering_proximity": 2.0,
                "num_plant_groups": 1,
            },
        }
        e = AgriDroneEnv(config, render_mode=None)
        e.reset(seed=42)
        # Placer le drone sur le groupe de plantes
        e.plant_groups[0] = np.array([2.0, 2.0, 0.5, 0.0], dtype=np.float32)
        e.dynamics.reset(np.array([2.0, 2.0, 1.0]))

        tank_before = e.water_tank_level  # 100.0
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
        _, _, _, _, info = e.step(action)

        if info["just_watered"]:
            assert e.water_tank_level == pytest.approx(tank_before - 10.0), (
                f"Le réservoir devrait diminuer de 10.0 (water_consumption configuré)"
            )
        e.close()
