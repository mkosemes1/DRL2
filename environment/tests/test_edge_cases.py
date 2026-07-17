"""
tests/test_edge_cases.py
========================
Tests de cas limites (edge cases) pour la robustesse de l'environnement AgriDroneEnv.

Vérifie le comportement de l'environnement dans des situations inhabituelles :
  - Actions nulles et extrêmes
  - Réinitialisation idempotente
  - Tous les groupes arrosés (distance = 0, index = -1)
  - Forme et type des observations
  - Clip des actions
  - Décrément de la batterie
  - Incrémentation du compteur de pas
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
    """Configuration minimale pour les tests de cas limites."""
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


# ─── Tests Actions Nulles et Extrêmes ───────────────────────────

class TestActionEdgeCases:
    """Tests pour les actions nulles et extrêmes."""

    def test_step_with_zero_action(self, env):
        """Vérifie qu'aucun crash ne se produit avec une action entièrement nulle."""
        env.reset(seed=42)
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(np.zeros(6))
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, float)
            assert np.all(np.isfinite(obs))

    def test_step_with_extreme_action(self, env):
        """Vérifie qu'aucun crash ne se produit avec toutes les actions à ±1.0."""
        env.reset(seed=42)
        # Action maximale positive
        action_pos = np.ones(6, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_pos)
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))

        # Action maximale négative
        action_neg = -np.ones(6, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_neg)
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))

        # Action alternée
        action_alt = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action_alt)
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))

    def test_step_with_action_out_of_bounds(self, env):
        """Vérifie que les actions hors [-1, 1] sont clipées correctement."""
        # Réinitialiser et exécuter avec une action extrême
        env.reset(seed=42)
        action_extreme = np.array([100.0, -100.0, 50.0, -50.0, 10.0, -10.0], dtype=np.float32)
        obs1, _, _, _, _ = env.step(action_extreme)

        # Réinitialiser et exécuter avec l'action équivalente clipée
        env.reset(seed=42)
        action_clipped = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        obs2, _, _, _, _ = env.step(action_clipped)

        # Comparer uniquement les 21 premières dimensions (état du drone + bassine + réservoir)
        # Les dimensions 21+ correspondent aux groupes de plantes dont les positions
        # sont aléatoires à chaque reset (rng non seedé dans le code source)
        np.testing.assert_array_almost_equal(obs1[:21], obs2[:21])


# ─── Tests Réinitialisation Idempotente ──────────────────────────

class TestResetIdempotent:
    """Tests pour vérifier que reset() est idempotente."""

    def test_reset_idempotent(self, config):
        """Appelle reset() deux fois, vérifie que le second reset fonctionne correctement."""
        e = AgriDroneEnv(config, render_mode=None)

        # Premier reset
        obs1, info1 = e.reset(seed=42)
        # Exécuter quelques pas pour modifier l'état
        for _ in range(5):
            e.step(np.zeros(6))

        # Second reset
        obs2, info2 = e.reset(seed=42)

        # Vérifier que les attributs d'état internes sont correctement réinitialisés
        assert e.step_count == 0, "step_count devrait être 0 après reset"
        assert e.water_tank_level == 100.0, "water_tank_level devrait être 100.0 après reset"
        assert e.battery_level == 1e6, "battery_level devrait être 1e6 après reset"
        assert np.all(e.plant_groups[:, 3] == 0.0), "Tous les groupes devraient être non arrosés"

        # Vérifier que la position du drone est réinitialisée
        pos = e.dynamics.state.position()
        np.testing.assert_array_almost_equal(pos, [0.0, 0.0, 1.0])

        # Les 17 premières dimensions de l'observation (état du drone)
        # devraient être identiques après deux resets (même position)
        # Note : les groupes de plantes (dims 21+) sont aléatoires à chaque reset
        # car le rng dans reset() n'est pas seedé
        np.testing.assert_array_almost_equal(obs1[:17], obs2[:17])
        e.close()

    def test_reset_clears_previous_state(self, env):
        """Vérifie que reset efface l'état précédent (batterie, step_count, etc.)."""
        # Modifier l'état
        env.step_count = 500
        env.battery_level = 0.0
        env.water_tank_level = 0.0
        env.plant_groups[:, 3] = 1.0

        # Reset
        env.reset(seed=42)

        assert env.step_count == 0
        assert env.battery_level == 1e6
        assert env.water_tank_level == 100.0
        assert np.all(env.plant_groups[:, 3] == 0.0)


# ─── Tests Distance et Index Tous Arrosés ───────────────────────

class TestAllWateredEdgeCases:
    """Tests pour les cas limites quand tous les groupes sont arrosés."""

    def test_distance_all_watered_returns_zero(self, env):
        """Vérifie que _distance_to_nearest_unwatered retourne 0.0 quand tous les groupes sont arrosés."""
        env.reset(seed=42)
        env.plant_groups[:, 3] = 1.0
        dist = env._distance_to_nearest_unwatered()
        assert dist == 0.0, "La distance devrait être 0.0 quand tous les groupes sont arrosés"

    def test_nearest_unwatered_index_all_watered(self, env):
        """Vérifie que _nearest_unwatered_group_index retourne (-1, inf) quand tous sont arrosés."""
        env.reset(seed=42)
        env.plant_groups[:, 3] = 1.0
        idx, dist = env._nearest_unwatered_group_index()
        assert idx == -1, "L'indice devrait être -1 quand tous les groupes sont arrosés"
        assert dist == float("inf"), "La distance devrait être inf quand tous les groupes sont arrosés"

    def test_watering_when_all_watered(self, env):
        """Vérifie qu'aucun arrosage ne se produit quand tous les groupes sont déjà arrosés."""
        env.reset(seed=42)
        env.plant_groups[:, 3] = 1.0
        # Placer le drone sur un groupe
        env.dynamics.reset(np.array([env.plant_groups[0, 0], env.plant_groups[0, 1], 1.0]))

        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
        _, _, _, _, info = env.step(action)

        assert info["just_watered"] is False, "Aucun arrosage ne devrait se produire"
        assert info["all_watered"] is True, "all_watered devrait être True"


# ─── Tests Forme et Type des Observations ────────────────────────

class TestObservationShapeAndType:
    """Tests pour la forme et le type des vecteurs d'observation."""

    def test_observation_shape_matches_space(self, env):
        """Après reset, vérifie que obs.shape == observation_space.shape."""
        obs, _ = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape, (
            f"La forme de l'observation ({obs.shape}) ne correspond pas "
            f"à observation_space.shape ({env.observation_space.shape})"
        )

    def test_observation_dtype_float32(self, env):
        """Vérifie que le dtype de l'observation est float32."""
        obs, _ = env.reset(seed=42)
        assert obs.dtype == np.float32, f"Le dtype devrait être float32, obtenu {obs.dtype}"

    def test_observation_after_step_shape(self, env):
        """Après un step, vérifie que la forme de l'observation est toujours correcte."""
        env.reset(seed=42)
        for _ in range(10):
            obs, _, _, _, _ = env.step(np.zeros(6))
            assert obs.shape == env.observation_space.shape

    def test_observation_values_in_range(self, env):
        """Vérifie que toutes les valeurs d'observation sont dans [-1, 1]."""
        env.reset(seed=42)
        obs = env._get_obs()
        assert obs.min() >= -1.0, f"La valeur minimale {obs.min()} est inférieure à -1.0"
        assert obs.max() <= 1.0, f"La valeur maximale {obs.max()} est supérieure à 1.0"


# ─── Tests Clip des Actions ─────────────────────────────────────

class TestActionClipping:
    """Tests pour vérifier le clip des actions dans l'environnement."""

    def test_action_clipping(self, env):
        """Envoie une action hors [-1,1], vérifie qu'elle est clipée."""
        # Réinitialiser et exécuter avec une action extrême
        env.reset(seed=42)
        action_extreme = np.array([10.0, -10.0, 5.0, -5.0, 100.0, -100.0], dtype=np.float32)
        obs1, _, _, _, _ = env.step(action_extreme)

        # Réinitialiser et exécuter avec l'action équivalente clipée
        env.reset(seed=42)
        action_normal = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        obs2, _, _, _, _ = env.step(action_normal)

        # Comparer uniquement les 21 premières dimensions (état du drone + bassine + réservoir)
        # Les dimensions 21+ correspondent aux groupes de plantes dont les positions
        # sont aléatoires à chaque reset (rng non seedé dans le code source)
        np.testing.assert_array_almost_equal(obs1[:21], obs2[:21])


# ─── Tests Batterie et Compteur de Pas ──────────────────────────

class TestBatteryAndStepCount:
    """Tests pour la décrémentation de la batterie et l'incrémentation du compteur de pas."""

    def test_battery_decreases_per_step(self, env):
        """Vérifie que la batterie diminue à chaque pas."""
        env.reset(seed=42)
        battery_before = env.battery_level

        env.step(np.zeros(6))
        assert env.battery_level < battery_before, (
            f"La batterie devrait diminuer : avant={battery_before}, après={env.battery_level}"
        )

        battery_after_one = env.battery_level
        env.step(np.zeros(6))
        assert env.battery_level < battery_after_one, (
            f"La batterie devrait continuer à diminuer : avant={battery_after_one}, après={env.battery_level}"
        )

    def test_battery_decreases_linearly(self, env):
        """Vérifie que la batterie diminue de 0.001 à chaque pas."""
        env.reset(seed=42)
        initial_battery = env.battery_level

        for i in range(1, 11):
            env.step(np.zeros(6))
            expected = initial_battery - 0.001 * i
            assert env.battery_level == pytest.approx(expected, abs=1e-7), (
                f"Après {i} pas, la batterie devrait être {expected}, obtenu {env.battery_level}"
            )

    def test_step_count_increments(self, env):
        """Vérifie que step_count s'incrémente à chaque pas."""
        env.reset(seed=42)
        assert env.step_count == 0

        for i in range(1, 11):
            env.step(np.zeros(6))
            assert env.step_count == i, (
                f"Après {i} pas, step_count devrait être {i}, obtenu {env.step_count}"
            )

    def test_step_count_resets_to_zero(self, env):
        """Vérifie que step_count revient à 0 après un reset."""
        env.reset(seed=42)
        for _ in range(20):
            env.step(np.zeros(6))
        assert env.step_count == 20

        env.reset(seed=42)
        assert env.step_count == 0
