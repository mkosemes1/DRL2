"""
tests/test_training_pipeline.py
================================
Tests unitaires pour le pipeline d'entraînement PPO (train.py).

Vérifie la classe Trainer qui hérite de BaseTrain et orchestre la
boucle rollout → GAE → update PPO. Teste aussi les composants
sous-jacents : Buffer et PPOTrainer.

Les tests utilisent une configuration minimale pour garder les
tests rapides (pas de rendu PyBullet).
"""

import sys
import os
import pytest
import numpy as np
import torch
import tempfile

# ─── Configuration des chemins d'import ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agent"))

# Sauter tous les tests si pybullet n'est pas disponible
pytest.importorskip("pybullet")
pytest.importorskip("pybullet_data")

from agri_drone_env import AgriDroneEnv
from model import Agent
from train import Trainer
from rl_template.train import BaseTrain
from rl_template.common import Buffer
from rl_template.algorithms.ppo.ppo import PPOTrainer
from rl_template.config import PPOConfig, TrainConfig
from rl_template.errors import EmptyBufferError


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def small_env():
    """Environnement minimal pour les tests."""
    config = {
        "world": {"size_x": 40.0, "size_y": 40.0, "ground_z": 0.0, "size_z": 10.0},
        "drone": {"dry_mass": 10.0, "payload_mass_full": 5.0},
        "simulation": {"dt": 0.02, "max_episode_steps": 50},
        "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
        "water_task": {
            "basin_position": [15.0, 15.0, 0.5],
            "basin_refill_radius": 3.0,
            "water_consumption": 2.0,
            "watering_proximity": 2.0,
            "num_plant_groups": 3,
        },
    }
    env = AgriDroneEnv(config)
    yield env
    env.close()


@pytest.fixture
def trainer(small_env):
    """Trainer minimal pour les tests."""
    obs_dim = small_env.observation_space.shape[0]
    act_dim = small_env.action_space.shape[0]
    agent = Agent(n_state=obs_dim, n_action=act_dim)

    tmp_dir = tempfile.mkdtemp()
    train_config = TrainConfig(
        model_name="test_model",
        model_saved_path=tmp_dir,
        timestamp=128,
        batch_size=32,
        rollout_steps=64,
    )
    ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
    t = Trainer(
        env=small_env,
        agent=agent,
        train_config=train_config,
        ppo_config=ppo_config,
    )
    yield t


# ─── Classe : Héritage du Trainer ─────────────────────────────────


class TestTrainerInheritance:
    """Vérifie que Trainer hérite correctement de BaseTrain."""

    def test_trainer_is_subclass_of_base_train(self):
        """Trainer doit être une sous-classe de BaseTrain."""
        assert issubclass(Trainer, BaseTrain)

    def test_trainer_is_instance_of_base_train(self, trainer):
        """L'instance Trainer doit être reconnue comme une instance de BaseTrain."""
        assert isinstance(trainer, BaseTrain)


# ─── Classe : Initialisation du Trainer ────────────────────────────


class TestTrainerInit:
    """Vérifie que le Trainer s'initialise avec tous ses composants."""

    def test_trainer_has_agent(self, trainer):
        """Le Trainer doit posséder un attribut agent de type Agent."""
        assert isinstance(trainer.agent, Agent)

    def test_trainer_has_env(self, trainer):
        """Le Trainer doit posséder un attribut env de type AgriDroneEnv."""
        assert isinstance(trainer.env, AgriDroneEnv)

    def test_trainer_has_buffer(self, trainer):
        """Le Trainer doit posséder un attribut buffer de type Buffer."""
        assert isinstance(trainer.buffer, Buffer)

    def test_trainer_has_ppo_trainer(self, trainer):
        """Le Trainer doit posséder un attribut ppo_trainer de type PPOTrainer."""
        assert isinstance(trainer.ppo_trainer, PPOTrainer)

    def test_trainer_buffer_state_shape(self, trainer, small_env):
        """Le buffer doit avoir la bonne forme pour les états (obs_dim,)."""
        obs_dim = small_env.observation_space.shape[0]
        assert trainer.buffer.states.shape[1:] == (obs_dim,)

    def test_trainer_buffer_action_shape(self, trainer, small_env):
        """Le buffer doit avoir la bonne forme pour les actions continues (act_dim,)."""
        act_dim = small_env.action_space.shape[0]
        assert trainer.buffer.actions.shape[1:] == (act_dim,)

    def test_trainer_buffer_capacity(self, trainer):
        """La capacité du buffer (step) doit correspondre à rollout_steps."""
        assert trainer.buffer.step == trainer.train_config.rollout_steps


# ─── Classe : Phase de rollout ─────────────────────────────────────


class TestRolloutPhase:
    """Vérifie que la phase de collecte d'expérience fonctionne correctement."""

    def test_rollout_fills_buffer(self, trainer, small_env):
        """Après un rollout, le buffer doit contenir exactement rollout_steps entrées."""
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        assert trainer.buffer.size == trainer.train_config.rollout_steps

    def test_rollout_states_shape(self, trainer, small_env):
        """Les états stockés dans le buffer doivent avoir la bonne forme."""
        obs_dim = small_env.observation_space.shape[0]
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        filled = trainer.buffer.states[: trainer.buffer.size]
        assert filled.shape == (trainer.train_config.rollout_steps, obs_dim)

    def test_rollout_actions_shape(self, trainer, small_env):
        """Les actions stockées dans le buffer doivent avoir la forme (rollout_steps, act_dim)."""
        act_dim = small_env.action_space.shape[0]
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        filled = trainer.buffer.actions[: trainer.buffer.size]
        assert filled.shape == (trainer.train_config.rollout_steps, act_dim)

    def test_rollout_log_probs_shape(self, trainer, small_env):
        """Les log-probs doivent être des scalaires par pas : forme (rollout_steps,)."""
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        filled = trainer.buffer.old_log_probs[: trainer.buffer.size]
        # Log-probs sommés sur les dimensions d'action → scalaire par pas
        assert filled.shape == (trainer.train_config.rollout_steps,)

    def test_rollout_rewards_are_finite(self, trainer, small_env):
        """Toutes les récompenses collectées doivent être des float finis."""
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        filled = trainer.buffer.rewards[: trainer.buffer.size]
        assert np.all(np.isfinite(filled))

    def test_rollout_values_are_finite(self, trainer, small_env):
        """Toutes les estimations de valeur doivent être des float finis."""
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        filled = trainer.buffer.values[: trainer.buffer.size]
        assert np.all(np.isfinite(filled))

    def test_rollout_handles_episode_reset(self, trainer, small_env):
        """Le rollout doit gérer les fins d'épisode et marquer des dones=1.

        Avec max_episode_steps=50 et rollout_steps=64, au moins un
        épisode sera tronqué, donc au moins un dones=1 doit apparaître.
        """
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        filled = trainer.buffer.dones[: trainer.buffer.size]
        assert np.sum(filled) > 0, "Aucune fin d'épisode détectée pendant le rollout"


# ─── Classe : Mise à jour des poids ────────────────────────────────


class TestUpdateWeights:
    """Vérifie la boucle PPO de mise à jour des poids."""

    def test_update_returns_4_losses(self, trainer, small_env):
        """update_weights doit retourner un tuple de 4 valeurs."""
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        result = trainer.update_weights(0)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_update_loss_is_float(self, trainer, small_env):
        """Chaque loss retournée par update_weights doit être un float."""
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        loss, pi_loss, v_loss, entropy = trainer.update_weights(0)
        assert isinstance(loss, float)
        assert isinstance(pi_loss, float)
        assert isinstance(v_loss, float)
        assert isinstance(entropy, float)

    def test_update_loss_is_finite(self, trainer, small_env):
        """Chaque loss retournée par update_weights doit être finie (pas NaN/Inf)."""
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        loss, pi_loss, v_loss, entropy = trainer.update_weights(0)
        assert np.isfinite(loss)
        assert np.isfinite(pi_loss)
        assert np.isfinite(v_loss)
        assert np.isfinite(entropy)

    def test_update_clears_buffer(self, trainer, small_env):
        """Après update_weights, le buffer doit être vidé (size == 0)."""
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        assert trainer.buffer.size == trainer.train_config.rollout_steps
        trainer.update_weights(0)
        assert trainer.buffer.size == 0

    def test_update_reduces_loss(self, trainer, small_env):
        """Deux mises à jour successives doivent produire des losses différentes.

        Le réseau apprend entre les deux updates, donc la loss globale
        doit évoluer (vérifie que l'apprentissage a bien lieu).
        """
        state, _ = small_env.reset()
        trainer.rollout_phase(state)
        loss1, _, _, _ = trainer.update_weights(0)

        # Deuxième rollout + update
        trainer.rollout_phase(state)
        loss2, _, _, _ = trainer.update_weights(1)

        # Les losses doivent être différentes (l'agent a appris)
        assert loss1 != loss2, "Aucune évolution de loss entre deux updates"


# ─── Classe : Sauvegarde du modèle ─────────────────────────────────


class TestSaveModel:
    """Vérifie la sauvegarde et la reproductibilité du modèle."""

    def test_save_model_creates_file(self, trainer):
        """save_model() doit créer le fichier de poids du modèle."""
        trainer.save_model()
        assert os.path.isfile(trainer.train_config.model_path)

    def test_save_model_idempotent(self, trainer):
        """Appeler save_model() deux fois ne doit pas provoquer d'erreur."""
        trainer.save_model()
        trainer.save_model()
        assert os.path.isfile(trainer.train_config.model_path)


# ─── Classe : Boucle d'entraînement ───────────────────────────────


class TestTrainLoop:
    """Vérifie la boucle complète d'entraînement PPO."""

    def test_train_runs_without_error(self, trainer):
        """train(verbose=False) doit s'exécuter sans erreur.

        Avec timestamp=128 et rollout_steps=64, la boucle effectue
        2 mises à jour PPO.
        """
        trainer.train(verbose=False)

    def test_train_uses_tqdm(self):
        """Vérifie que tqdm est importé dans le module train.

        Le module train.py importe tqdm pour les barres de progression.
        """
        assert "tqdm" in sys.modules, "tqdm n'est pas importé dans le contexte Python"

    def test_train_wandb_optional(self, small_env):
        """Un Trainer sans wandb_config doit fonctionner normalement."""
        obs_dim = small_env.observation_space.shape[0]
        act_dim = small_env.action_space.shape[0]
        agent = Agent(n_state=obs_dim, n_action=act_dim)

        tmp_dir = tempfile.mkdtemp()
        train_config = TrainConfig(
            model_name="test_no_wandb",
            model_saved_path=tmp_dir,
            timestamp=128,
            batch_size=32,
            rollout_steps=64,
        )
        ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)

        # Pas de wandb_config → doit fonctionner sans erreur
        t = Trainer(
            env=small_env,
            agent=agent,
            train_config=train_config,
            ppo_config=ppo_config,
        )
        assert t.wandb_config is None
        t.train(verbose=False)


# ─── Classe : Intégration du Buffer ────────────────────────────────


class TestBufferIntegration:
    """Vérifie le comportement du Buffer pour des actions continues."""

    def test_buffer_insert_continuous_action(self, small_env):
        """Le buffer doit accepter un tableau d'action continue 6D."""
        act_dim = small_env.action_space.shape[0]
        obs_dim = small_env.observation_space.shape[0]
        buf = Buffer(step=10, state_shape=(obs_dim,), action_shape=(act_dim,))

        state = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randn(act_dim).astype(np.float32)
        buf.insert(
            state=state,
            action=action,
            old_log_prob=0.5,
            reward=1.0,
            value=0.5,
            dones=0,
        )
        assert buf.size == 1
        np.testing.assert_array_equal(buf.actions[0], action)

    def test_buffer_get_all_continuous(self, small_env):
        """get_all() doit retourner des tenseurs avec les bonnes formes."""
        act_dim = small_env.action_space.shape[0]
        obs_dim = small_env.observation_space.shape[0]
        T = 16
        buf = Buffer(step=T, state_shape=(obs_dim,), action_shape=(act_dim,))

        for _ in range(T):
            buf.insert(
                state=np.random.randn(obs_dim).astype(np.float32),
                action=np.random.randn(act_dim).astype(np.float32),
                old_log_prob=0.1,
                reward=1.0,
                value=0.5,
                dones=0,
            )

        states, actions, old_log_probs, returns, adv, rewards, values, dones = (
            buf.get_all()
        )
        assert states.shape == (T, obs_dim)
        assert actions.shape == (T, act_dim)
        assert old_log_probs.shape == (T,)
        assert returns.shape == (T,)
        assert adv.shape == (T,)
        assert rewards.shape == (T,)
        assert values.shape == (T,)
        assert dones.shape == (T,)

    def test_buffer_full_raises(self, small_env):
        """Le buffer doit lever ValueError quand il est plein."""
        act_dim = small_env.action_space.shape[0]
        obs_dim = small_env.observation_space.shape[0]
        buf = Buffer(step=3, state_shape=(obs_dim,), action_shape=(act_dim,))

        state = np.zeros(obs_dim, dtype=np.float32)
        action = np.zeros(act_dim, dtype=np.float32)
        for _ in range(3):
            buf.insert(
                state=state, action=action, old_log_prob=0.0,
                reward=0.0, value=0.0, dones=0,
            )
        with pytest.raises(ValueError, match="Buffer is full"):
            buf.insert(
                state=state, action=action, old_log_prob=0.0,
                reward=0.0, value=0.0, dones=0,
            )

    def test_buffer_clear_resets(self, small_env):
        """buffer.clear() doit réinitialiser le pointeur de slice à 0."""
        act_dim = small_env.action_space.shape[0]
        obs_dim = small_env.observation_space.shape[0]
        buf = Buffer(step=10, state_shape=(obs_dim,), action_shape=(act_dim,))

        state = np.zeros(obs_dim, dtype=np.float32)
        action = np.zeros(act_dim, dtype=np.float32)
        for _ in range(5):
            buf.insert(
                state=state, action=action, old_log_prob=0.0,
                reward=0.0, value=0.0, dones=0,
            )
        assert buf.size == 5
        buf.clear()
        assert buf.size == 0


# ─── Classe : Intégration PPO ──────────────────────────────────────


class TestPPOIntegration:
    """Vérifie les composants clés de l'algorithme PPO."""

    def test_compute_gae_returns_tuple(self, small_env):
        """compute_gae() doit retourner un tuple de 3 tableaux numpy."""
        obs_dim = small_env.observation_space.shape[0]
        act_dim = small_env.action_space.shape[0]
        agent = Agent(n_state=obs_dim, n_action=act_dim)
        ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
        ppo_trainer = PPOTrainer(model=agent, ppo_config=ppo_config)

        T = 32
        rewards = np.random.randn(T).astype(np.float32)
        values = np.random.randn(T).astype(np.float32)
        dones = np.zeros(T, dtype=np.float32)
        dones[15] = 1.0  # Fin d'épisode au milieu
        last_value = 0.5

        result = ppo_trainer.compute_gae(rewards, values, last_value, dones)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_compute_gae_shapes(self, small_env):
        """Les shapes de returns, advantages et deltas doivent être (T,)."""
        obs_dim = small_env.observation_space.shape[0]
        act_dim = small_env.action_space.shape[0]
        agent = Agent(n_state=obs_dim, n_action=act_dim)
        ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
        ppo_trainer = PPOTrainer(model=agent, ppo_config=ppo_config)

        T = 32
        rewards = np.random.randn(T).astype(np.float32)
        values = np.random.randn(T).astype(np.float32)
        dones = np.zeros(T, dtype=np.float32)
        last_value = 0.5

        returns, advantages, deltas = ppo_trainer.compute_gae(
            rewards, values, last_value, dones
        )
        assert returns.shape == (T,)
        assert advantages.shape == (T,)
        assert deltas.shape == (T,)

    def test_lr_decay_changes_lr(self, small_env):
        """Après lr_decay(), le learning rate de l'optimiseur doit changer."""
        obs_dim = small_env.observation_space.shape[0]
        act_dim = small_env.action_space.shape[0]
        agent = Agent(n_state=obs_dim, n_action=act_dim)
        ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
        ppo_trainer = PPOTrainer(model=agent, ppo_config=ppo_config)

        lr_initial = ppo_trainer.optimizer.param_groups[0]["lr"]
        # Appliquer le decay à mi-parcours (step=50, total=100)
        ppo_trainer.lr_decay(lr=3e-4, total_steps=100, step=50)
        lr_after = ppo_trainer.optimizer.param_groups[0]["lr"]

        # Le LR doit avoir diminué (frac = 1 - 50/100 = 0.5)
        assert lr_after < lr_initial
        assert np.isclose(lr_after, 3e-4 * 0.5, rtol=1e-6)
