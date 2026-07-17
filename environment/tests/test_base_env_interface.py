"""
tests/test_base_env_interface.py
=================================
Tests unitaires pour vérifier qu'AgriDroneEnv implémente correctement
l'interface BaseEnv de rl_template.

Vérifie :
  - L'héritage de BaseEnv (et de gym.Env)
  - L'espace d'observation (Box, forme, bornes, dtype)
  - L'espace d'action (Box, forme, bornes)
  - L'interface reset() (type retour, seed, idempotence)
  - L'interface step() (5-tuple, types, forme, actions torch, clipping, troncation)
  - L'interface close() (appelable, idempotent)
  - L'API Gymnasium (metadata, contains)
"""

import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Sauter tous les tests si pybullet n'est pas disponible
pybullet = pytest.importorskip("pybullet")
pytest.importorskip("pybullet_data")

import gymnasium as gym
from rl_template.env import BaseEnv
from agri_drone_env import AgriDroneEnv


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def env():
    """Crée une instance AgriDroneEnv pour les tests."""
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
            "num_plant_groups": 3,
        },
    }
    e = AgriDroneEnv(config)
    yield e
    e.close()


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
    e = AgriDroneEnv(config)
    yield e
    e.close()


# ─── Tests Héritage BaseEnv ──────────────────────────────────────

class TestBaseEnvInheritance:
    """Tests de vérification de l'héritage de BaseEnv."""

    def test_is_subclass_of_base_env(self):
        """Vérifie qu'AgriDroneEnv est une sous-classe de BaseEnv."""
        assert issubclass(AgriDroneEnv, BaseEnv), (
            "AgriDroneEnv doit hériter de BaseEnv depuis rl_template.env"
        )

    def test_is_instance_of_base_env(self, env):
        """Vérifie que l'instance est de type BaseEnv."""
        assert isinstance(env, BaseEnv), (
            "L'instance AgriDroneEnv doit être une instance de BaseEnv"
        )

    def test_is_instance_of_gym_env(self, env):
        """Vérifie que l'instance est aussi de type gym.Env (BaseEnv en hérite)."""
        assert isinstance(env, gym.Env), (
            "L'instance AgriDroneEnv doit être une instance de gym.Env"
        )


# ─── Tests Espace d'Observation ──────────────────────────────────

class TestObservationSpace:
    """Tests pour la vérification de l'espace d'observation."""

    def test_observation_space_is_set(self, env):
        """Vérifie que observation_space est défini (pas None)."""
        assert env.observation_space is not None, (
            "observation_space doit être défini dans __init__"
        )

    def test_observation_space_is_box(self, env):
        """Vérifie que l'espace d'observation est un gym.spaces.Box."""
        assert isinstance(env.observation_space, gym.spaces.Box), (
            "observation_space doit être un gym.spaces.Box"
        )

    def test_observation_space_dtype(self, env):
        """Vérifie que le dtype de l'espace d'observation est float32."""
        assert env.observation_space.dtype == np.float32, (
            f"Le dtype doit être float32, reçu : {env.observation_space.dtype}"
        )

    def test_observation_space_shape(self, env):
        """Vérifie que la forme est 17 + 3 + 1 + N*4 (N = num_plant_groups)."""
        num_groups = env.num_plant_groups  # 3
        expected_dim = 17 + 3 + 1 + num_groups * 4  # 33
        assert env.observation_space.shape == (expected_dim,), (
            f"La forme doit être ({expected_dim},), reçu : {env.observation_space.shape}"
        )

    def test_observation_space_bounds(self, env):
        """Vérifie que les bornes sont low=-1.0 et high=1.0."""
        np.testing.assert_array_equal(
            env.observation_space.low, -1.0,
            err_msg="Les valeurs basses de observation_space doivent être -1.0"
        )
        np.testing.assert_array_equal(
            env.observation_space.high, 1.0,
            err_msg="Les valeurs hautes de observation_space doivent être 1.0"
        )


# ─── Tests Espace d'Action ───────────────────────────────────────

class TestActionSpace:
    """Tests pour la vérification de l'espace d'action."""

    def test_action_space_is_set(self, env):
        """Vérifie que action_space est défini (pas None)."""
        assert env.action_space is not None, (
            "action_space doit être défini dans __init__"
        )

    def test_action_space_is_box(self, env):
        """Vérifie que l'espace d'action est un gym.spaces.Box."""
        assert isinstance(env.action_space, gym.spaces.Box), (
            "action_space doit être un gym.spaces.Box"
        )

    def test_action_space_shape(self, env):
        """Vérifie que la forme de l'espace d'action est (6,)."""
        assert env.action_space.shape == (6,), (
            f"La forme de action_space doit être (6,), reçu : {env.action_space.shape}"
        )

    def test_action_space_bounds(self, env):
        """Vérifie que les bornes de l'action sont [-1, 1]."""
        np.testing.assert_array_equal(
            env.action_space.low, -1.0,
            err_msg="Les valeurs basses de action_space doivent être -1.0"
        )
        np.testing.assert_array_equal(
            env.action_space.high, 1.0,
            err_msg="Les valeurs hautes de action_space doivent être 1.0"
        )


# ─── Tests Interface Reset ───────────────────────────────────────

class TestResetInterface:
    """Tests pour l'interface reset() conformément à BaseEnv."""

    def test_reset_returns_tuple(self, env):
        """Vérifie que reset() retourne un tuple de 2 éléments."""
        result = env.reset()
        assert isinstance(result, tuple), (
            f"reset() doit retourner un tuple, reçu : {type(result)}"
        )
        assert len(result) == 2, (
            f"reset() doit retourner 2 éléments (obs, info), reçu : {len(result)}"
        )

    def test_reset_returns_obs_and_info(self, env):
        """Vérifie que le premier élément est un ndarray et le second un dict."""
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray), (
            f"Le premier élément doit être un np.ndarray, reçu : {type(obs)}"
        )
        assert isinstance(info, dict), (
            f"Le second élément doit être un dict, reçu : {type(info)}"
        )

    def test_reset_obs_shape(self, env):
        """Vérifie que la forme de l'observation correspond à observation_space."""
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape, (
            f"La forme de l'observation {obs.shape} ne correspond pas "
            f"à observation_space {env.observation_space.shape}"
        )

    def test_reset_obs_dtype(self, env):
        """Vérifie que le dtype de l'observation est float32."""
        obs, _ = env.reset()
        assert obs.dtype == np.float32, (
            f"Le dtype de l'observation doit être float32, reçu : {obs.dtype}"
        )

    def test_reset_with_seed(self, env):
        """Vérifie que reset(seed=42) fonctionne sans erreur."""
        obs, info = env.reset(seed=42)
        assert obs is not None, "reset(seed=42) doit retourner une observation"
        assert info is not None, "reset(seed=42) doit retourner des info"

    def test_reset_without_args(self, env):
        """Vérifie que reset() sans argument fonctionne."""
        obs, info = env.reset()
        assert obs is not None, "reset() sans argument doit retourner une observation"
        assert info is not None, "reset() sans argument doit retourner des info"

    def test_reset_idempotent(self, env):
        """Vérifie que reset() peut être appelé deux fois sans erreur."""
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert obs1.shape == obs2.shape, (
            "Les observations des deux reset doivent avoir la même forme"
        )


# ─── Tests Interface Step ────────────────────────────────────────

class TestStepInterface:
    """Tests pour l'interface step() conformément à BaseEnv."""

    def test_step_returns_5_tuple(self, env):
        """Vérifie que step() retourne un tuple de 5 éléments."""
        env.reset()
        result = env.step(np.zeros(6, dtype=np.float32))
        assert isinstance(result, tuple), (
            f"step() doit retourner un tuple, reçu : {type(result)}"
        )
        assert len(result) == 5, (
            f"step() doit retourner 5 éléments, reçu : {len(result)}"
        )

    def test_step_returns_obs(self, env):
        """Vérifie que le premier élément (obs) est un ndarray."""
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert isinstance(obs, np.ndarray), (
            f"L'observation doit être un np.ndarray, reçu : {type(obs)}"
        )

    def test_step_returns_reward(self, env):
        """Vérifie que le deuxième élément (reward) est un float."""
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert isinstance(reward, (float, np.floating)), (
            f"La récompense doit être un float, reçu : {type(reward)}"
        )

    def test_step_returns_terminated(self, env):
        """Vérifie que le troisième élément (terminated) est un bool."""
        env.reset()
        _, _, terminated, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert isinstance(terminated, (bool, np.bool_)), (
            f"terminated doit être un bool, reçu : {type(terminated)}"
        )

    def test_step_returns_truncated(self, env):
        """Vérifie que le quatrième élément (truncated) est un bool."""
        env.reset()
        _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
        assert isinstance(truncated, (bool, np.bool_)), (
            f"truncated doit être un bool, reçu : {type(truncated)}"
        )

    def test_step_returns_info(self, env):
        """Vérifie que le cinquième élément (info) est un dict."""
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert isinstance(info, dict), (
            f"info doit être un dict, reçu : {type(info)}"
        )

    def test_step_obs_shape(self, env):
        """Vérifie que la forme de l'observation de step correspond à observation_space."""
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert obs.shape == env.observation_space.shape, (
            f"La forme de l'observation step {obs.shape} ne correspond pas "
            f"à observation_space {env.observation_space.shape}"
        )

    def test_step_with_numpy_action(self, env):
        """Vérifie que step() accepte un tableau numpy comme action."""
        env.reset()
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None, "L'observation ne doit pas être None"
        assert isinstance(reward, (float, np.floating)), (
            f"La récompense doit être un float, reçu : {type(reward)}"
        )

    def test_step_with_torch_tensor_action(self, env):
        """Vérifie que step() accepte un torch.Tensor comme action.

        CRITIQUE : BaseTrain de rl_template passe des tensors PyTorch
        à step(). L'environnement doit gérer la conversion automatiquement.
        """
        torch = pytest.importorskip("torch")
        env.reset()
        action = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # La méthode step doit supporter les tensors torch
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray), (
            f"L'observation doit être un np.ndarray même avec action torch, "
            f"reçu : {type(obs)}"
        )
        assert isinstance(reward, (float, np.floating)), (
            f"La récompense doit être un float, reçu : {type(reward)}"
        )

    def test_step_action_clipping(self, env):
        """Vérifie que les actions hors de [-1,1] sont correctement gérées (reward fini)."""
        env.reset()
        # Action avec des valeurs largement hors de [-1, 1]
        action = np.array([5.0, -5.0, 10.0, -10.0, 3.0, -3.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        assert np.isfinite(reward), (
            f"La récompense doit être finie même avec des actions hors bornes, "
            f"reçu : {reward}"
        )

    def test_step_multiple(self, env):
        """Vérifie que 10 pas successifs ne provoquent pas d'erreur."""
        env.reset()
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(6, dtype=np.float32)
            )
            assert obs.shape == env.observation_space.shape, (
                f"Erreur au pas {i}: forme {obs.shape} incorrecte"
            )
            assert np.isfinite(reward), (
                f"Erreur au pas {i}: récompense non finie {reward}"
            )
            if terminated or truncated:
                break  # fin d'épisode anticipée, c'est normal

    def test_step_truncation(self, short_env):
        """Vérifie que truncated=True après max_episode_steps pas."""
        short_env.reset()
        truncated = False
        for _ in range(short_env.max_steps):
            _, _, _, truncated, _ = short_env.step(np.zeros(6, dtype=np.float32))
        assert truncated is True, (
            f"truncated doit être True après {short_env.max_steps} pas"
        )


# ─── Tests Interface Close ───────────────────────────────────────

class TestCloseInterface:
    """Tests pour l'interface close() conformément à BaseEnv."""

    def test_close_callable(self, env):
        """Vérifie que close() peut être appelé sans erreur."""
        # Ne doit pas lever d'exception
        env.close()

    def test_close_idempotent(self, env):
        """Vérifie que close() peut être appelé deux fois sans erreur."""
        env.close()
        # Le second appel ne doit pas provoquer d'erreur
        env.close()


# ─── Tests API Gymnasium ─────────────────────────────────────────

class TestGymnasiumAPI:
    """Tests pour les aspects spécifiques à l'API Gymnasium."""

    def test_render_modes(self, env):
        """Vérifie que metadata['render_modes'] est défini."""
        assert hasattr(env, "metadata"), "L'environnement doit avoir un attribut metadata"
        assert "render_modes" in env.metadata, (
            "metadata doit contenir 'render_modes'"
        )
        assert isinstance(env.metadata["render_modes"], list), (
            "render_modes doit être une liste"
        )

    def test_observation_space_contains(self, env):
        """Vérifie que observation_space.contains() accepte une observation valide."""
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), (
            "observation_space.contains(obs) doit être True après reset"
        )

    def test_action_space_contains(self, env):
        """Vérifie que action_space.contains() accepte une action valide."""
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert env.action_space.contains(action), (
            "action_space.contains(action) doit être True pour une action valide"
        )
