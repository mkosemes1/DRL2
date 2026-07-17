"""
tests/test_minimal_fly_env.py
=============================
Tests unitaires pour l'environnement MinimalFlyEnv.

Vérifie les fonctionnalités de l'environnement Gymnasium minimal
sans logique agricole :
  - Espaces d'observation et d'action
  - Réinitialisation (reset) et état initial
  - Exécution de pas (step) et format de retour
  - Bornes des observations
  - Comportement sans troncation (pas de max_steps)
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minimal_fly_env import MinimalFlyEnv


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def env():
    """Environnement MinimalFlyEnv initialisé sans rendu PyBullet."""
    e = MinimalFlyEnv(render_mode=None)
    e.reset(seed=42)
    return e


# ─── Tests Espaces ──────────────────────────────────────────────

class TestMinimalSpaces:
    """Tests pour les espaces d'observation et d'action de MinimalFlyEnv."""

    def test_minimal_obs_space(self):
        """Vérifie que l'espace d'observation a 9 dimensions."""
        e = MinimalFlyEnv(render_mode=None)
        assert e.observation_space.shape == (9,)

    def test_minimal_action_space(self):
        """Vérifie que l'espace d'action a 4 dimensions continues dans [-1, 1]."""
        e = MinimalFlyEnv(render_mode=None)
        assert e.action_space.shape == (4,)
        assert e.action_space.low.min() == -1.0
        assert e.action_space.high.max() == 1.0


# ─── Tests Reset ────────────────────────────────────────────────

class TestMinimalReset:
    """Tests pour la réinitialisation de MinimalFlyEnv."""

    def test_minimal_reset(self, env):
        """Vérifie que reset() retourne (obs, info) avec les bonnes formes."""
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (9,)
        assert isinstance(info, dict)

    def test_minimal_reset_position(self, env):
        """Vérifie que le drone est initialisé à la position (0, 0, 1) après reset."""
        env.reset(seed=42)
        pos = env.dynamics.state.position()
        np.testing.assert_array_almost_equal(pos, [0.0, 0.0, 1.0])

    def test_minimal_reset_zero_velocity(self, env):
        """Vérifie que les vitesses sont nulles après reset."""
        env.reset(seed=42)
        vel = env.dynamics.state.velocity()
        np.testing.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])

    def test_minimal_reset_zero_attitude(self, env):
        """Vérifie que l'attitude est nulle après reset."""
        env.reset(seed=42)
        assert env.dynamics.state.roll == pytest.approx(0.0)
        assert env.dynamics.state.pitch == pytest.approx(0.0)
        assert env.dynamics.state.yaw == pytest.approx(0.0)


# ─── Tests Step ─────────────────────────────────────────────────

class TestMinimalStep:
    """Tests pour la méthode step() de MinimalFlyEnv."""

    def test_minimal_step(self, env):
        """Vérifie que step() retourne (obs, reward, terminated, truncated, info)."""
        result = env.step(np.zeros(4))
        assert isinstance(result, tuple)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (9,)
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_minimal_step_reward_zero(self, env):
        """Vérifie que la récompense est toujours 0.0 (pas de logique de récompense)."""
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.5, 0.0, 0.5, 0.0]))
        assert reward == 0.0

    def test_minimal_step_never_terminated(self, env):
        """Vérifie que terminated est toujours False (pas de condition d'arrêt)."""
        env.reset(seed=42)
        for _ in range(50):
            _, _, terminated, _, _ = env.step(np.array([0.8, 0.0, 0.5, 0.0]))
            assert terminated is False

    def test_minimal_step_action_clipping(self, env):
        """Vérifie que les actions hors [-1, 1] sont clipées."""
        env.reset(seed=42)
        # Action extrême supérieure
        action_extreme = np.array([5.0, 5.0, 5.0, 5.0])
        obs1, _, _, _, _ = env.step(action_extreme)
        # Action clipée équivalente
        env.reset(seed=42)
        action_clipped = np.array([1.0, 1.0, 1.0, 1.0])
        obs2, _, _, _, _ = env.step(action_clipped)
        # Les deux doivent produire le même résultat
        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_minimal_step_zero_action(self, env):
        """Vérifie qu'aucun crash ne se produit avec une action entièrement nulle."""
        env.reset(seed=42)
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(np.zeros(4))
            assert obs.shape == (9,)


# ─── Tests Bornes des Observations ──────────────────────────────

class TestMinimalObsBounds:
    """Tests pour vérifier que les valeurs d'observation sont finies."""

    def test_minimal_obs_finite_after_reset(self, env):
        """Vérifie que les observations sont finies après reset."""
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))

    def test_minimal_obs_finite_after_steps(self, env):
        """Vérifie que les observations restent finies après plusieurs pas."""
        env.reset(seed=42)
        for _ in range(20):
            obs, _, _, _, _ = env.step(np.array([0.8, 0.0, 0.5, 0.0]))
            assert np.all(np.isfinite(obs)), f"Observation non finie: {obs}"


# ─── Tests Troncation ───────────────────────────────────────────

class TestMinimalTruncation:
    """Tests pour vérifier l'absence de troncation dans MinimalFlyEnv."""

    def test_minimal_max_steps_truncation(self):
        """Vérifie que MinimalFlyEnv ne tronque jamais (pas de max_steps)."""
        e = MinimalFlyEnv(render_mode=None)
        e.reset(seed=42)
        # Exécuter 200 pas — ne devrait jamais tronquer
        for _ in range(200):
            _, _, _, truncated, _ = e.step(np.array([0.5, 0.0, 0.3, 0.0]))
            assert truncated is False, "MinimalFlyEnv ne devrait jamais tronquer"


# ─── Tests Configuration ────────────────────────────────────────

class TestMinimalConfig:
    """Tests pour vérifier que les paramètres internes de MinimalFlyEnv sont corrects."""

    def test_minimal_custom_config(self, env):
        """Vérifie que les paramètres physiques par défaut sont utilisés."""
        # Le drone devrait avoir une masse totale de 10.0 + 5.0 = 15.0
        assert env.dynamics.params.total_mass == pytest.approx(15.0)
        assert env.dynamics.params.dry_mass == pytest.approx(10.0)
        assert env.dynamics.params.payload_mass == pytest.approx(5.0)
        assert env.dynamics.params.gravity == pytest.approx(9.81)
        assert env.dynamics.params.max_thrust_total == pytest.approx(350.0)
        assert env.dynamics.dt == pytest.approx(0.02)

    def test_minimal_world_bounds(self, env):
        """Vérifie que les limites du monde sont correctement définies."""
        assert env.dynamics.world_bounds["x"] == (-100, 100)
        assert env.dynamics.world_bounds["y"] == (-100, 100)
        assert env.dynamics.world_bounds["z"] == (0, 100)

    def test_minimal_observation_content(self, env):
        """Vérifie que l'observation contient position, vitesse et attitude."""
        env.reset(seed=42)
        obs = env._get_obs()
        # Les 3 premières dimensions sont la position
        assert obs[0] == pytest.approx(0.0)  # x
        assert obs[1] == pytest.approx(0.0)  # y
        assert obs[2] == pytest.approx(1.0)  # z
        # Les 3 suivantes sont la vitesse (nulle après reset)
        assert obs[3] == pytest.approx(0.0)  # vx
        assert obs[4] == pytest.approx(0.0)  # vy
        assert obs[5] == pytest.approx(0.0)  # vz
        # Les 3 dernières sont l'attitude (nulle après reset)
        assert obs[6] == pytest.approx(0.0)  # roll
        assert obs[7] == pytest.approx(0.0)  # pitch
        assert obs[8] == pytest.approx(0.0)  # yaw
