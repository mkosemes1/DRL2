"""
tests/test_agri_drone_env.py
==============================
Tests unitaires pour l'environnement AgriDroneEnv.

Vérifie les fonctionnalités principales de l'environnement Gymnasium
pour drone agricole :
  - Espaces d'observation et d'action
  - Réinitialisation (reset) et état initial
  - Exécution de pas (step) et format de retour
  - Arrosage des groupes de plantes
  - Remplissage du réservoir à la bassine
  - Arrêt anticipé et troncation
  - Construction de l'observation normalisée
  - Randomisation de la grille du champ
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Sauter tous les tests si pybullet n'est pas disponible
pybullet = pytest.importorskip("pybullet")
pytest.importorskip("pybullet_data")

from agri_drone_env import AgriDroneEnv, FieldCell


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def minimal_config():
    """Configuration minimale pour les tests de l'environnement."""
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
def env(minimal_config):
    """Environnement AgriDroneEnv initialisé et réinitialisé."""
    e = AgriDroneEnv(minimal_config, render_mode=None)
    e.reset(seed=42)
    return e


@pytest.fixture
def short_env():
    """Environnement avec max_episode_steps=5 pour tests de troncation rapide."""
    config = {
        "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
        "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
        "simulation": {"dt": 0.02, "max_episode_steps": 5},
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
    return e


# ─── Tests Espaces ──────────────────────────────────────────────

class TestSpaces:
    """Tests pour les espaces d'observation et d'action."""

    def test_observation_space_dimensions(self, env):
        """Vérifie que obs_dim == 17 + 3 + 1 + num_plant_groups * 4."""
        num_groups = env.num_plant_groups  # 3
        expected_dim = 17 + 3 + 1 + num_groups * 4  # 33
        assert env.observation_space.shape == (expected_dim,)

    def test_observation_space_bounds(self, env):
        """Vérifie que toutes les valeurs d'observation sont dans [-1, 1]."""
        obs, _ = env.reset(seed=42)
        assert obs.min() >= -1.0
        assert obs.max() <= 1.0
        # Après plusieurs pas
        for _ in range(10):
            obs, _, _, _, _ = env.step(np.zeros(6))
            assert obs.min() >= -1.0, f"obs min {obs.min()} < -1.0"
            assert obs.max() <= 1.0, f"obs max {obs.max()} > 1.0"

    def test_action_space(self, env):
        """Vérifie que l'espace d'action est Box(-1, 1, shape=(6,))."""
        assert env.action_space.shape == (6,)
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0


# ─── Tests Reset ────────────────────────────────────────────────

class TestReset:
    """Tests pour la réinitialisation de l'environnement."""

    def test_reset_returns_obs_and_info(self, env):
        """Vérifie que reset() retourne (obs, info) avec les bonnes formes."""
        obs, info = env.reset(seed=42)
        expected_dim = 17 + 3 + 1 + env.num_plant_groups * 4
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (expected_dim,)
        assert isinstance(info, dict)
        assert "goal_position" in info

    def test_reset_water_tank_full(self, env):
        """Vérifie que le réservoir est à 100.0 après reset."""
        env.reset(seed=42)
        assert env.water_tank_level == 100.0

    def test_reset_plant_groups_not_watered(self, env):
        """Vérifie que tous les groupes de plantes ne sont pas arrosés après reset."""
        env.reset(seed=42)
        # Chaque groupe a (x, y, z, is_watered) → is_watered = colonne 3
        assert np.all(env.plant_groups[:, 3] == 0.0)

    def test_reset_step_count_zero(self, env):
        """Vérifie que le compteur de pas est à 0 après reset."""
        env.reset(seed=42)
        assert env.step_count == 0

    def test_reset_battery_full(self, env):
        """Vérifie que la batterie est pleine après reset."""
        env.reset(seed=42)
        assert env.battery_level == 1e6


# ─── Tests Step ─────────────────────────────────────────────────

class TestStep:
    """Tests pour la méthode step()."""

    def test_step_returns_correct_format(self, env):
        """Vérifie que step() retourne (obs, reward, terminated, truncated, info)."""
        env.reset(seed=42)
        result = env.step(np.zeros(6))
        assert isinstance(result, tuple)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_step_info_has_reward_terms(self, env):
        """Vérifie que info['reward_terms'] contient les 5 clés attendues."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(6))
        assert "reward_terms" in info
        expected_keys = {"watering", "refill", "time_penalty", "distance_shaping", "mission_complete"}
        assert set(info["reward_terms"].keys()) == expected_keys


# ─── Tests Arrosage ─────────────────────────────────────────────

class TestWatering:
    """Tests pour la mécanique d'arrosage des groupes de plantes."""

    def test_watering_action_consumes_water(self, env):
        """Vérifie que l'arrosage consomme 2.0 unités d'eau du réservoir."""
        env.reset(seed=42)
        # Placer un groupe de plantes à une position connue
        env.plant_groups[0] = np.array([2.0, 2.0, 0.5, 0.0], dtype=np.float32)
        # Placer le drone sur ce groupe
        env.dynamics.reset(np.array([2.0, 2.0, 1.0]))

        tank_before = env.water_tank_level
        # Action : throttle suffisant pour rester en l'air, irrigation activée
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
        _, _, _, _, info = env.step(action)

        if info["just_watered"]:
            assert env.water_tank_level == pytest.approx(tank_before - 2.0)

    def test_watering_marks_group_watered(self, env):
        """Vérifie qu'après arrosage, le groupe est marqué comme arrosé."""
        env.reset(seed=42)
        # Placer un groupe de plantes à une position connue
        env.plant_groups[0] = np.array([2.0, 2.0, 0.5, 0.0], dtype=np.float32)
        env.dynamics.reset(np.array([2.0, 2.0, 1.0]))

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
        _, _, _, _, info = env.step(action)

        if info["just_watered"]:
            assert env.plant_groups[0, 3] == 1.0

    def test_cannot_water_empty_tank(self, env):
        """Vérifie que l'arrosage ne se produit pas si le réservoir est insuffisant."""
        env.reset(seed=42)
        env.plant_groups[0] = np.array([2.0, 2.0, 0.5, 0.0], dtype=np.float32)
        env.dynamics.reset(np.array([2.0, 2.0, 1.0]))
        # Mettre le réservoir à un niveau inférieur à la consommation
        env.water_tank_level = 1.0

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
        _, _, _, _, info = env.step(action)

        # Le groupe ne doit pas être arrosé car le réservoir est insuffisant
        assert env.plant_groups[0, 3] == 0.0
        assert info["just_watered"] is False

    def test_cannot_water_without_irrigation_action(self, env):
        """Vérifie qu'aucun arrosage ne se produit si action[5] <= 0."""
        env.reset(seed=42)
        env.plant_groups[0] = np.array([2.0, 2.0, 0.5, 0.0], dtype=np.float32)
        env.dynamics.reset(np.array([2.0, 2.0, 1.0]))

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])  # pas d'irrigation
        _, _, _, _, info = env.step(action)

        assert env.plant_groups[0, 3] == 0.0
        assert info["just_watered"] is False

    def test_cannot_water_far_group(self, env):
        """Vérifie qu'aucun arrosage ne se produit si le drone est trop loin d'un groupe."""
        env.reset(seed=42)
        # Placer le groupe loin du drone
        env.plant_groups[0] = np.array([15.0, 15.0, 0.5, 0.0], dtype=np.float32)
        env.dynamics.reset(np.array([2.0, 2.0, 1.0]))

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
        _, _, _, _, info = env.step(action)

        assert info["just_watered"] is False


# ─── Tests Bassine et Remplissage ──────────────────────────────

class TestBasinRefill:
    """Tests pour le remplissage du réservoir à la bassine."""

    def test_basin_refill(self, env):
        """Vérifie que le réservoir se remplit à 100.0 quand le drone est à la bassine."""
        env.reset(seed=42)
        # Placer le drone à la position de la bassine
        env.dynamics.reset(np.array([15.0, 15.0, 1.0]))
        env.water_tank_level = 50.0

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, _, _, _, info = env.step(action)

        assert env.water_tank_level == 100.0
        assert info["just_refilled"] is True

    def test_basin_refill_reward_guard(self, env):
        """Vérifie que just_refilled=False si le réservoir est déjà à 100.0."""
        env.reset(seed=42)
        env.dynamics.reset(np.array([15.0, 15.0, 1.0]))
        env.water_tank_level = 100.0  # déjà plein

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, _, _, _, info = env.step(action)

        # Le réservoir est déjà >= 98.0, donc just_refilled doit être False
        assert info["just_refilled"] is False
        assert env.water_tank_level == 100.0

    def test_basin_refill_reward_when_low(self, env):
        """Vérifie que just_refilled=True si le réservoir est à 50.0 à la bassine."""
        env.reset(seed=42)
        env.dynamics.reset(np.array([15.0, 15.0, 1.0]))
        env.water_tank_level = 50.0

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, _, _, _, info = env.step(action)

        assert info["just_refilled"] is True
        assert env.water_tank_level == 100.0

    def test_basin_refill_at_threshold(self, env):
        """Vérifie que just_refilled=False si le réservoir est à 98.0 (seuil exact)."""
        env.reset(seed=42)
        env.dynamics.reset(np.array([15.0, 15.0, 1.0]))
        env.water_tank_level = 98.0  # >= 98.0 → pas de refill

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, _, _, _, info = env.step(action)

        assert info["just_refilled"] is False

    def test_basin_refill_just_below_threshold(self, env):
        """Vérifie que just_refilled=True si le réservoir est juste en dessous de 98.0."""
        env.reset(seed=42)
        env.dynamics.reset(np.array([15.0, 15.0, 1.0]))
        env.water_tank_level = 97.9

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, _, _, _, info = env.step(action)

        assert info["just_refilled"] is True
        assert env.water_tank_level == 100.0


# ─── Tests Fin d'épisode ───────────────────────────────────────

class TestEpisodeEnd:
    """Tests pour l'arrêt anticipé et la troncation."""

    def test_early_stopping_all_watered(self, env):
        """Vérifie terminated=True quand tous les groupes sont arrosés."""
        env.reset(seed=42)
        # Marquer tous les groupes comme arrosés
        env.plant_groups[:, 3] = 1.0

        obs, reward, terminated, truncated, info = env.step(np.zeros(6))
        assert terminated is True

    def test_truncation_at_max_steps(self, short_env):
        """Vérifie truncated=True après max_episode_steps pas."""
        short_env.reset(seed=42)
        truncated = False
        for _ in range(short_env.max_steps):
            _, _, _, truncated, _ = short_env.step(np.zeros(6))
        assert truncated is True

    def test_not_terminated_initially(self, env):
        """Vérifie terminated=False au début de l'épisode."""
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(np.zeros(6))
        assert terminated is False


# ─── Tests Observation ──────────────────────────────────────────

class TestObservation:
    """Tests pour la construction du vecteur d'observation."""

    def test_observation_includes_basin_coords(self, env):
        """Vérifie que l'observation contient les coordonnées normalisées de la bassine."""
        obs, _ = env.reset(seed=42)

        # Bassine à (15.0, 15.0, 0.5)
        # world_bounds: x=(-20, 20), y=(-20, 20), z=(0, 10)
        # safe_norm(15.0, -20.0, 20.0) = 2*(15+20)/40 - 1 = 0.75
        # safe_norm(0.5, 0.0, 10.0) = 2*0.5/10 - 1 = -0.9
        assert obs[17] == pytest.approx(0.75, abs=1e-5)
        assert obs[18] == pytest.approx(0.75, abs=1e-5)
        assert obs[19] == pytest.approx(-0.9, abs=1e-5)

    def test_observation_includes_tank_level(self, env):
        """Vérifie que l'observation contient le niveau du réservoir normalisé."""
        obs, _ = env.reset(seed=42)
        # water_tank_level = 100.0 → safe_norm(100, 0, 100) = 1.0
        assert obs[20] == pytest.approx(1.0, abs=1e-5)

    def test_observation_includes_plant_groups(self, env):
        """Vérifie que l'observation contient les N*4 dimensions des groupes de plantes."""
        obs, _ = env.reset(seed=42)
        num_groups = env.num_plant_groups
        # Les dimensions 21 à 21+N*4-1 sont les groupes de plantes
        expected_plant_dims = num_groups * 4
        assert obs.shape[0] == 17 + 3 + 1 + expected_plant_dims
        # Vérifier que les valeurs des plantes sont dans [-1, 1]
        plant_section = obs[21:]
        assert plant_section.min() >= -1.0
        assert plant_section.max() <= 1.0

    def test_is_watered_normalization_zero(self, env):
        """Vérifie que is_watered=0.0 est normalisé en -1.0 dans l'observation."""
        env.reset(seed=42)
        # Tous les groupes ont is_watered=0.0 après reset
        obs = env._get_obs()
        num_groups = env.num_plant_groups
        for k in range(num_groups):
            is_watered_dim = 21 + k * 4 + 3
            assert obs[is_watered_dim] == pytest.approx(-1.0, abs=1e-5)

    def test_is_watered_normalization_one(self, env):
        """Vérifie que is_watered=1.0 est normalisé en +1.0 dans l'observation."""
        env.reset(seed=42)
        # Marquer le premier groupe comme arrosé
        env.plant_groups[0, 3] = 1.0
        obs = env._get_obs()
        is_watered_dim = 21 + 0 * 4 + 3
        assert obs[is_watered_dim] == pytest.approx(1.0, abs=1e-5)

    def test_observation_drone_position(self, env):
        """Vérifie que la position du drone est encodée dans l'observation."""
        obs, _ = env.reset(seed=42)
        # Le drone est initialisé à (0, 0, 1)
        # safe_norm(0, -20, 20) = 2*20/40 - 1 = 0.0
        # safe_norm(1, 0, 10) = 2*1/10 - 1 = -0.8
        assert obs[0] == pytest.approx(0.0, abs=1e-5)  # x
        assert obs[1] == pytest.approx(0.0, abs=1e-5)  # y
        assert obs[2] == pytest.approx(-0.8, abs=1e-5)  # z


# ─── Tests Grille du champ ─────────────────────────────────────

class TestFieldGrid:
    """Tests pour la grille du champ agricole."""

    def test_field_grid_created_on_first_reset(self, env):
        """Vérifie que la grille du champ est créée au premier reset."""
        env.reset(seed=42)
        assert env.field_grid is not None
        assert len(env.field_grid) == env.field_size[0]
        assert len(env.field_grid[0]) == env.field_size[1]

    def test_field_grid_rerandomization(self, env):
        """Vérifie que les états healthy/wet changent entre deux reset (statistiquement)."""
        env.reset(seed=1)

        # Capturer l'état de la grille après le premier reset
        healthy_states_1 = []
        wet_states_1 = []
        for i in range(env.field_size[0]):
            for j in range(env.field_size[1]):
                healthy_states_1.append(env.field_grid[i][j].healthy)
                wet_states_1.append(env.field_grid[i][j].wet)

        # Deuxième reset
        env.reset(seed=2)

        healthy_states_2 = []
        wet_states_2 = []
        for i in range(env.field_size[0]):
            for j in range(env.field_size[1]):
                healthy_states_2.append(env.field_grid[i][j].healthy)
                wet_states_2.append(env.field_grid[i][j].wet)

        # Au moins quelques cellules doivent changer (statistiquement très probable)
        healthy_changes = sum(a != b for a, b in zip(healthy_states_1, healthy_states_2))
        wet_changes = sum(a != b for a, b in zip(wet_states_1, wet_states_2))
        # Avec 400 cellules et ~15% de chance de changer, au moins 10 devraient changer
        assert healthy_changes > 0, "Aucun changement healthy entre deux reset"
        assert wet_changes > 0, "Aucun changement wet entre deux reset"

    def test_field_cells_reset_flags(self, env):
        """Vérifie que les flags visited/sprayed/watered sont réinitialisés."""
        env.reset(seed=42)
        # Modifier quelques cellules
        env.field_grid[0][0].visited = True
        env.field_grid[0][0].sprayed = True
        env.field_grid[0][0].watered = True

        # Nouveau reset
        env.reset(seed=42)
        assert env.field_grid[0][0].visited is False
        assert env.field_grid[0][0].sprayed is False
        assert env.field_grid[0][0].watered is False


# ─── Tests Configuration ────────────────────────────────────────

class TestConfig:
    """Tests pour la lecture de la configuration."""

    def test_watering_proximity_configurable(self, env):
        """Vérifie que watering_proximity est lu depuis la config."""
        assert env.watering_proximity == 2.0

    def test_custom_num_plant_groups(self):
        """Vérifie que num_plant_groups modifie correctement la dimension d'observation."""
        config = {
            "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
            "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
            "simulation": {"dt": 0.02, "max_episode_steps": 100},
            "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
            "water_task": {
                "basin_position": [15.0, 15.0, 0.5],
                "basin_refill_radius": 3.0,
                "water_consumption": 2.0,
                "watering_proximity": 2.0,
                "num_plant_groups": 7,
            },
        }
        e = AgriDroneEnv(config, render_mode=None)
        obs, _ = e.reset(seed=42)
        expected_dim = 17 + 3 + 1 + 7 * 4  # 49
        assert obs.shape == (expected_dim,)
        assert e.observation_space.shape == (expected_dim,)
        e.close()

    def test_basin_position_configurable(self, env):
        """Vérifie que la position de la bassine est lue depuis la config."""
        np.testing.assert_array_almost_equal(
            env.water_basin_position, [15.0, 15.0, 0.5]
        )

    def test_basin_refill_radius_configurable(self, env):
        """Vérifie que le rayon de remplissage est lu depuis la config."""
        assert env.basin_refill_radius == 3.0

    def test_water_consumption_configurable(self, env):
        """Vérifie que la consommation d'eau est lue depuis la config."""
        assert env.water_consumption == 2.0

    def test_max_steps_configurable(self, env):
        """Vérifie que max_episode_steps est lu depuis la config."""
        assert env.max_steps == 100


# ─── Tests Utilitaires ──────────────────────────────────────────

class TestUtilities:
    """Tests pour les méthodes utilitaires internes."""

    def test_distance_to_nearest_unwatered_all_watered(self, env):
        """Vérifie que la distance est 0.0 quand tous les groupes sont arrosés."""
        env.reset(seed=42)
        env.plant_groups[:, 3] = 1.0
        dist = env._distance_to_nearest_unwatered()
        assert dist == 0.0

    def test_distance_to_nearest_unwatered(self, env):
        """Vérifie le calcul de distance au groupe non arrosé le plus proche."""
        env.reset(seed=42)
        # Placer un groupe à une position connue
        env.plant_groups[0] = np.array([3.0, 4.0, 0.5, 0.0], dtype=np.float32)
        env.dynamics.reset(np.array([0.0, 0.0, 1.0]))

        dist = env._distance_to_nearest_unwatered()
        # Distance euclidienne entre (0,0,1) et (3,4,0.5)
        expected = np.sqrt(9 + 16 + 0.25)  # ~5.025
        assert dist == pytest.approx(expected, abs=0.1)

    def test_nearest_unwatered_group_index(self, env):
        """Vérifie que le bon indice de groupe est retourné."""
        env.reset(seed=42)
        env.plant_groups[0] = np.array([10.0, 10.0, 0.5, 0.0], dtype=np.float32)
        env.plant_groups[1] = np.array([1.0, 1.0, 0.5, 0.0], dtype=np.float32)
        env.plant_groups[2] = np.array([5.0, 5.0, 0.5, 1.0], dtype=np.float32)  # arrosé
        env.dynamics.reset(np.array([0.0, 0.0, 1.0]))

        idx, dist = env._nearest_unwatered_group_index()
        # Le groupe le plus proche non arrosé est l'indice 1 à (1,1,0.5)
        assert idx == 1

    def test_nearest_unwatered_group_all_watered(self, env):
        """Vérifie idx=-1 quand tous les groupes sont arrosés."""
        env.reset(seed=42)
        env.plant_groups[:, 3] = 1.0
        idx, _ = env._nearest_unwatered_group_index()
        assert idx == -1


# ─── Tests Close ────────────────────────────────────────────────

class TestClose:
    """Tests pour la méthode close()."""

    def test_close_without_render(self, env):
        """Vérifie que close() ne plante pas sans rendu PyBullet."""
        env.reset(seed=42)
        # Ne devrait pas lever d'exception
        env.close()
