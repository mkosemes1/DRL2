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
import shutil

# ─── Configuration des chemins d'import ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agent"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Sauter tous les tests si pybullet n'est pas disponible
pybullet = pytest.importorskip("pybullet")
pytest.importorskip("pybullet_data")

from agri_drone_env import AgriDroneEnv
from model import Agent
from train import Trainer
from rl_template.train import BaseTrain, Buffer, PPOTrainer, EmptyBufferError
from rl_template.config import PPOConfig, TrainConfig


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def small_env_config():
    """Configuration minimale pour un environnement rapide (50 pas max)."""
    return {
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


@pytest.fixture
def small_env(small_env_config):
    """Environnement AgriDroneEnv minimal pour les tests."""
    env = AgriDroneEnv(small_env_config)
    yield env
    env.close()


@pytest.fixture
def obs_dim(small_env):
    """Dimension d'observation calculée depuis l'environnement."""
    return small_env.observation_space.shape[0]


@pytest.fixture
def agent(obs_dim):
    """Agent PPO avec les dimensions correspondant à small_env."""
    return Agent(n_state=obs_dim, n_action=6)


@pytest.fixture
def trainer(small_env, agent):
    """Trainer complet initialisé avec un环境, un agent, et des configs minimales.

    Utilise un répertoire temporaire pour la sauvegarde des modèles.
    Le cleanup est automatique via ``tmp_path``.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_config = TrainConfig(
            model_name="test_model",
            model_saved_path=tmp_dir,
            timestamp=128,
            batch_size=32,
            rollout_steps=64,
        )
        ppo_config = PPOConfig(
            lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2
        )
        t = Trainer(
            env=small_env,
            agent=agent,
            train_config=train_config,
            ppo_config=ppo_config,
        )
        yield t


# ─── Tests : Héritage et initialisation ────────────────────────────

def test_trainer_is_subclass_of_base_train():
    """Trainer doit hériter de BaseTrain (interface rl_template)."""
    assert issubclass(Trainer, BaseTrain)


def test_trainer_init(trainer):
    """Le Trainer doit s'initialiser sans erreur avec les configs minimales."""
    assert trainer is not None


def test_trainer_has_buffer(trainer):
    """Le Trainer possède un buffer de type Buffer."""
    assert isinstance(trainer.buffer, Buffer)


def test_trainer_has_ppo_trainer(trainer):
    """Le Trainer possède un PPOTrainer."""
    assert isinstance(trainer.ppo_trainer, PPOTrainer)


def test_trainer_has_agent(trainer):
    """Le Trainer possède un Agent."""
    assert isinstance(trainer.agent, Agent)


def test_trainer_has_env(trainer):
    """Le Trainer possède un AgriDroneEnv."""
    assert isinstance(trainer.env, AgriDroneEnv)


def test_trainer_buffer_shape(trainer, obs_dim):
    """Le buffer doit avoir les bonnes dimensions d'état et d'action.

    - ``state_shape``: ``(obs_dim,)``
    - ``action_shape``: ``(6,)`` (actions continues)
    - ``old_log_probs``: ``(rollout_steps, 6)`` (par dimension d'action)
    """
    assert trainer.buffer.states.shape[1:] == (obs_dim,)
    assert trainer.buffer.actions.shape[1:] == (6,)
    assert trainer.buffer.old_log_probs.shape == (64, 6)


# ─── Tests : Rollout phase ────────────────────────────────────────

def test_trainer_rollout_phase(trainer, small_env):
    """rollout_phase(state) doit remplir le buffer sans erreur."""
    state, _ = small_env.reset()
    trainer.rollout_phase(state)
    # Le buffer doit contenir des données après le rollout
    assert trainer.buffer.size > 0


def test_trainer_rollout_fills_buffer(trainer, small_env):
    """Après un rollout, le buffer doit contenir exactement rollout_steps entrées."""
    state, _ = small_env.reset()
    trainer.rollout_phase(state)
    assert trainer.buffer.size == trainer.train_config.rollout_steps


def test_trainer_rollout_handles_episode_end(trainer, small_env_config, obs_dim):
    """Le rollout doit gérer correctement les fins d'épisode (terminated/truncated).

    Avec max_episode_steps=50 et rollout_steps=64, l'épisode sera
    tronqué au pas 50. Le rollout doit continuer à collecter
    des données après la réinitialisation.
    """
    env = AgriDroneEnv(small_env_config)
    try:
        agent = Agent(n_state=obs_dim, n_action=6)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tc = TrainConfig(
                model_name="test_ep", model_saved_path=tmp_dir,
                timestamp=128, batch_size=32, rollout_steps=64,
            )
            pc = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
            t = Trainer(env=env, agent=agent, train_config=tc, ppo_config=pc)

            state, _ = env.reset()
            t.rollout_phase(state)
            # Le buffer doit être rempli malgré la troncation de l'épisode
            assert t.buffer.size == 64
    finally:
        env.close()


# ─── Tests : Update weights ────────────────────────────────────────

def test_trainer_update_weights(trainer, small_env):
    """update_weights(0) doit retourner un tuple de 4 floats (loss, pi, v, ent)."""
    state, _ = small_env.reset()
    trainer.rollout_phase(state)
    result = trainer.update_weights(0)
    assert len(result) == 4


def test_trainer_update_weights_types(trainer, small_env):
    """Tous les retours de update_weights doivent être des floats."""
    state, _ = small_env.reset()
    trainer.rollout_phase(state)
    loss, pi_loss, v_loss, entropy = trainer.update_weights(0)
    assert isinstance(loss, float)
    assert isinstance(pi_loss, float)
    assert isinstance(v_loss, float)
    assert isinstance(entropy, float)


def test_trainer_update_clears_buffer(trainer, small_env):
    """Après update_weights, le buffer doit être vidé (size == 0)."""
    state, _ = small_env.reset()
    trainer.rollout_phase(state)
    assert trainer.buffer.size == 64  # Avant update
    trainer.update_weights(0)
    assert trainer.buffer.size == 0  # Après update


# ─── Tests : Sauvegarde du modèle ──────────────────────────────────

def test_trainer_save_model(trainer):
    """save_model() doit créer le fichier de poids du modèle."""
    trainer.save_model()
    assert os.path.isfile(trainer.train_config.model_path)


# ─── Tests : Boucle d'entraînement complète ────────────────────────

def test_trainer_train_short(trainer):
    """train(verbose=False) doit s'exécuter sans erreur.

    Avec timestamp=128 et rollout_steps=64, la boucle effectue
    2 mises à jour PPO.
    """
    trainer.train(verbose=False)


# ─── Tests : Buffer ────────────────────────────────────────────────

def test_buffer_insert(obs_dim):
    """L'insertion dans le buffer doit fonctionner correctement.

    Vérifie que les données insérées sont stockées aux bonnes positions.
    """
    buf = Buffer(step=10, state_shape=(obs_dim,), action_shape=(6,))
    state = np.random.randn(obs_dim).astype(np.float32)
    action = np.random.randn(6).astype(np.float32)
    log_prob = 0.5
    buf.insert(state=state, action=action, old_log_prob=log_prob,
               reward=1.0, value=0.5, dones=0)
    assert buf.size == 1
    np.testing.assert_array_equal(buf.states[0], state)
    np.testing.assert_array_equal(buf.actions[0], action)
    assert buf.rewards[0] == 1.0
    assert buf.values[0] == 0.5
    assert buf.dones[0] == 0.0


def test_buffer_full_raises(obs_dim):
    """Le buffer doit lever ValueError quand il est plein."""
    buf = Buffer(step=3, state_shape=(obs_dim,), action_shape=(6,))
    state = np.zeros(obs_dim, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    for _ in range(3):
        buf.insert(state=state, action=action, old_log_prob=0.0,
                   reward=0.0, value=0.0, dones=0)
    with pytest.raises(ValueError, match="Buffer is full"):
        buf.insert(state=state, action=action, old_log_prob=0.0,
                   reward=0.0, value=0.0, dones=0)


def test_buffer_clear(obs_dim):
    """buffer.clear() doit réinitialiser le pointeur de slice à 0."""
    buf = Buffer(step=10, state_shape=(obs_dim,), action_shape=(6,))
    state = np.zeros(obs_dim, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    for _ in range(5):
        buf.insert(state=state, action=action, old_log_prob=0.0,
                   reward=0.0, value=0.0, dones=0)
    assert buf.size == 5
    buf.clear()
    assert buf.size == 0


# ─── Tests : PPOTrainer ────────────────────────────────────────────

def test_ppo_trainer_compute_gae(obs_dim):
    """compute_gae() doit retourner des tableaux de la bonne taille.

    Les shapes de returns, advantages et deltas doivent correspondre
    à la taille de l'entrée (T,).
    """
    act_dim = 6
    agent = Agent(n_state=obs_dim, n_action=act_dim)
    ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
    ppo_trainer = PPOTrainer(model=agent, ppo_config=ppo_config)

    T = 32
    rewards = np.random.randn(T).astype(np.float32)
    values = np.random.randn(T).astype(np.float32)
    dones = np.zeros(T, dtype=np.float32)
    dones[15] = 1.0  # Fin d'épisode au milieu
    last_value = 0.5

    returns, advantages, deltas = ppo_trainer.compute_gae(
        rewards, values, last_value, dones
    )
    assert returns.shape == (T,)
    assert advantages.shape == (T,)
    assert deltas.shape == (T,)


def test_ppo_trainer_update(obs_dim):
    """PPOTrainer.update() doit retourner 4 losses (float) pour des actions discrètes.

    ``PPOTrainer.update()`` de rl_template est conçu pour des actions
    discrètes (log-prob scalaire par pas). Pour les actions continues,
    le ``Trainer.update_weights()`` effectue la somme des log-probs
    sur les dimensions d'action avant de calculer le ratio.

    Ce test vérifie que ``ppo_trainer.update()`` fonctionne avec des
    actions scalaires (discret) et un ``old_log_probs`` 1D.
    """
    act_dim = 1  # Action scalaire (discrète)
    agent = Agent(n_state=obs_dim, n_action=act_dim)
    ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
    ppo_trainer = PPOTrainer(model=agent, ppo_config=ppo_config)

    T = 32
    buf = Buffer(step=T, state_shape=(obs_dim,), action_shape=(act_dim,))
    # old_log_probs reste 1D (T,) pour actions discrètes

    for i in range(T):
        state = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randn(act_dim).astype(np.float32)
        log_prob = float(np.random.randn())  # scalaire
        buf.insert(state=state, action=action, old_log_prob=log_prob,
                   reward=np.random.randn(), value=np.random.randn(), dones=0)

    returns = np.random.randn(T).astype(np.float32)
    adv = np.random.randn(T).astype(np.float32)
    buf.insert_returns(returns, adv)

    result = ppo_trainer.update(buf, T, 0, batch_size=16)
    assert len(result) == 4
    loss, pi_loss, v_loss, entropy = result
    assert isinstance(loss, float)
    assert isinstance(pi_loss, float)
    assert isinstance(v_loss, float)
    assert isinstance(entropy, float)


def test_trainer_update_continuous_actions(trainer, small_env):
    """Trainer.update_weights() doit fonctionner correctement avec des actions continues.

    Le ``Trainer`` implémente sa propre boucle PPO qui somme les
    log-probabilités sur les dimensions d'action (6D) avant de
    calculer le ratio, ce que ``PPOTrainer.update()`` ne fait pas.
    """
    state, _ = small_env.reset()
    trainer.rollout_phase(state)
    result = trainer.update_weights(0)
    assert len(result) == 4
    loss, pi_loss, v_loss, entropy = result
    assert isinstance(loss, float)
    assert isinstance(pi_loss, float)
    assert isinstance(v_loss, float)
    assert isinstance(entropy, float)


def test_trainer_update_empty_buffer_raises(trainer, small_env):
    """Trainer.update_weights() doit lever EmptyBufferError si le buffer est insuffisant.

    Le buffer doit contenir au moins ``require_buffer_size`` (10) entrées
    avant de pouvoir effectuer une mise à jour PPO.
    """
    state, _ = small_env.reset()
    # Ne pas faire de rollout — le buffer est vide
    with pytest.raises(EmptyBufferError):
        trainer.update_weights(0)
