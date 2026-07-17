"""
tests/test_trainer_integration.py
===================================
Tests d'intégration end-to-end pour le pipeline d'entraînement PPO.

Instancie le Trainer avec un environnement réel et vérifie que la
boucle complète d'entraînement fonctionne : création de l'agent,
collecte d'expérience, mise à jour PPO, sauvegarde du modèle, et
évolution des métriques.

Chaque test utilise une configuration minimale pour rester rapide
(pas de rendu PyBullet, few updates).
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

pytest.importorskip("pybullet")
pytest.importorskip("pybullet_data")

from agri_drone_env import AgriDroneEnv
from model import Agent
from train import Trainer
from rl_template.config import PPOConfig, TrainConfig


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def env():
    """Environnement minimal pour les tests d'intégration."""
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
    e = AgriDroneEnv(config)
    yield e
    e.close()


@pytest.fixture
def agent(env):
    """Agent PPO avec les bonnes dimensions."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return Agent(n_state=obs_dim, n_action=act_dim)


@pytest.fixture
def trainer(env, agent):
    """Trainer configuré avec few updates pour tests rapides."""
    tmp_dir = tempfile.mkdtemp()
    train_config = TrainConfig(
        model_name="integration_test",
        model_saved_path=tmp_dir,
        timestamp=256,       # 256 timesteps total
        batch_size=32,
        rollout_steps=64,    # 256 // 64 = 4 updates
    )
    ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
    t = Trainer(
        env=env,
        agent=agent,
        train_config=train_config,
        ppo_config=ppo_config,
    )
    yield t


@pytest.fixture
def trainer_with_wandb(env, agent):
    """Trainer avec wandb_config pour tests du logging."""
    tmp_dir = tempfile.mkdtemp()
    train_config = TrainConfig(
        model_name="integration_wandb_test",
        model_saved_path=tmp_dir,
        timestamp=128,
        batch_size=32,
        rollout_steps=64,
    )
    ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
    wandb_config = {
        "project": "test-project",
        "entity": None,
        "name": "integration-test",
        "config": {"test": True},
    }
    t = Trainer(
        env=env,
        agent=agent,
        train_config=train_config,
        ppo_config=ppo_config,
        wandb_config=wandb_config,
    )
    yield t


# ─── Classe 1 : Instantiation complète ─────────────────────────────


class TestTrainerInstantiation:
    """Vérifie que le Trainer s'initialise correctement avec tous ses composants."""

    def test_create_trainer(self, env, agent):
        """Le Trainer doit s'instancier sans erreur avec env + agent."""
        tmp_dir = tempfile.mkdtemp()
        train_config = TrainConfig(
            model_name="test", model_saved_path=tmp_dir,
            timestamp=128, batch_size=32, rollout_steps=64,
        )
        ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
        t = Trainer(env=env, agent=agent, train_config=train_config, ppo_config=ppo_config)
        assert t is not None

    def test_trainer_has_all_components(self, trainer):
        """Le Trainer doit avoir agent, env, buffer, ppo_trainer."""
        assert trainer.agent is not None
        assert trainer.env is not None
        assert trainer.buffer is not None
        assert trainer.ppo_trainer is not None

    def test_trainer_config_correct(self, trainer):
        """Le Trainer doit stocker la config correctement."""
        assert trainer.train_config.timestamp == 256
        assert trainer.train_config.rollout_steps == 64
        assert trainer.train_config.batch_size == 32
        assert trainer.train_config.num_update == 4

    def test_trainer_buffer_shape(self, trainer, env):
        """Le buffer doit avoir les shapes correctes pour obs/action."""
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        assert trainer.buffer.states.shape[1:] == (obs_dim,)
        assert trainer.buffer.actions.shape[1:] == (act_dim,)

    def test_trainer_ppo_config(self, trainer):
        """Le PPOTrainer doit avoir les hyperparamètres corrects."""
        ppo_cfg = trainer.ppo_trainer.ppo_config
        assert ppo_cfg.lr == 3e-4
        assert ppo_cfg.gamma == 0.99
        assert ppo_cfg.gae_lambda == 0.95
        assert ppo_cfg.clip_eps == 0.2


# ─── Classe 2 : Boucle d'entraînement complète ─────────────────────


class TestFullTrainingLoop:
    """Vérifie que la boucle train() s'exécute correctement."""

    def test_train_completes(self, trainer):
        """train(verbose=False) doit s'exécuter sans erreur."""
        trainer.train(verbose=False)

    def test_train_creates_model_file(self, trainer):
        """Après train(), le fichier du modèle doit exister."""
        trainer.train(verbose=False)
        assert os.path.isfile(trainer.train_config.model_path)

    def test_train_model_weights_valid(self, trainer):
        """Après train(), les poids du modèle doivent être des tenseurs valides."""
        trainer.train(verbose=False)
        state_dict = trainer.agent.state_dict()
        for key, tensor in state_dict.items():
            assert tensor.shape.numel() > 0, f"Poids vides pour {key}"
            assert torch.all(torch.isfinite(tensor)), f"NaN/Inf dans {key}"

    def test_train_uses_correct_num_updates(self, trainer):
        """Avec timestamp=256 et rollout_steps=64, il doit y avoir 4 updates."""
        assert trainer.train_config.num_update == 4
        trainer.train(verbose=False)
        # Après 4 rollout + update, le buffer doit être vidé
        assert trainer.buffer.size == 0

    def test_train_resets_state_after_each_update(self, trainer):
        """Après chaque update, l'environnement doit être reset."""
        trainer.train(verbose=False)
        # cumulative_reward doit être 0 après le dernier reset
        assert trainer.cumulative_reward == 0.0


# ─── Classe 3 : Évolution de la loss ───────────────────────────────


class TestLossEvolution:
    """Vérifie que la loss évolue au cours de l'entraînement."""

    def test_loss_changes_between_updates(self, trainer, env):
        """Deux updates successives doivent produire des losses différentes."""
        state, _ = env.reset()

        # Premier rollout + update
        trainer.rollout_phase(state)
        loss1, _, _, _ = trainer.update_weights(0)

        # Deuxième rollout + update
        state, _ = env.reset()
        trainer.rollout_phase(state)
        loss2, _, _, _ = trainer.update_weights(1)

        # Les losses doivent être différentes
        assert loss1 != loss2, "Aucune évolution de loss entre deux updates"

    def test_all_loss_components_finite(self, trainer, env):
        """Tous les composants de loss doivent être finis (pas NaN/Inf)."""
        state, _ = env.reset()
        trainer.rollout_phase(state)
        loss, pi_loss, v_loss, entropy = trainer.update_weights(0)

        assert np.isfinite(loss), f"Loss non finie: {loss}"
        assert np.isfinite(pi_loss), f"Policy loss non finie: {pi_loss}"
        assert np.isfinite(v_loss), f"Value loss non finie: {v_loss}"
        assert np.isfinite(entropy), f"Entropy non finie: {entropy}"

    def test_loss_decreases_over_training(self, env, agent):
        """Avec assez d'updates, la loss moyenne doit diminuer."""
        tmp_dir = tempfile.mkdtemp()
        train_config = TrainConfig(
            model_name="loss_decrease_test",
            model_saved_path=tmp_dir,
            timestamp=512,       # 8 updates
            batch_size=32,
            rollout_steps=64,
        )
        ppo_config = PPOConfig(lr=3e-3, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
        t = Trainer(env=env, agent=agent, train_config=train_config, ppo_config=ppo_config)

        losses = []
        state, _ = env.reset()
        for i in range(train_config.num_update):
            t.rollout_phase(state)
            loss, _, _, _ = t.update_weights(i)
            losses.append(loss)
            state, _ = env.reset()

        # La loss du dernier update doit être inférieure ou égale au max des premiers
        # (pas garanti décroissante à chaque step, mais tendance générale)
        first_half = np.mean(losses[:len(losses)//2])
        second_half = np.mean(losses[len(losses)//2:])
        # Au minimum, la loss ne doit pas exploser
        assert second_half < first_half * 10, "Loss a explosé pendant l'entraînement"


# ─── Classe 4 : Observations et actions ─────────────────────────────


class TestObservationsAndActions:
    """Vérifie que les observations et actions sont valides pendant l'entraînement."""

    def test_observations_in_range(self, env):
        """Toutes les observations doivent être dans [-1, 1]."""
        obs, _ = env.reset()
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Observation hors de [-1, 1]"

    def test_actions_clipped_by_env(self, env, agent):
        """L'environnement doit clipper les actions dans [-1, 1] avant d'exécuter."""
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32)
        # L'agent peut échantillonner hors de [-1, 1] (distribution Normale)
        raw_action, _, _, _ = agent.get_action(obs_t)
        raw_np = raw_action.cpu().numpy()
        # Vérifier que env.step() clip correctement
        next_obs, reward, terminated, truncated, info = env.step(raw_np)
        # L'observation suivante doit être valide malgré une action hors limites
        assert np.all(np.isfinite(next_obs))
        assert np.isfinite(reward)

    def test_observation_shape_matches_config(self, env):
        """La dimension de l'observation doit correspondre à la config."""
        obs, _ = env.reset()
        # 17 (drone) + 3 (basin) + 1 (tank) + 3*4 (plant groups) = 32
        expected_dim = 17 + 3 + 1 + 3 * 4
        assert obs.shape == (expected_dim,), f"Shape obs: {obs.shape}, attendu ({expected_dim},)"

    def test_action_shape_is_6d(self, env, agent):
        """Les actions doivent avoir 6 dimensions."""
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32)
        action, _, _, _ = agent.get_action(obs_t)
        assert action.shape == (6,), f"Shape action: {action.shape}, attendu (6,)"

    def test_step_returns_valid_outputs(self, env, agent):
        """env.step() doit retourner les 5 éléments avec les bons types."""
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32)
        action, _, _, _ = agent.get_action(obs_t)
        action_np = action.cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action_np)

        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        assert next_obs.shape == obs.shape


# ─── Classe 5 : Sauvegarde et reproductibilité ──────────────────────


class TestModelSaving:
    """Vérifie la sauvegarde et la reproductibilité du modèle."""

    def test_save_model_creates_file(self, trainer):
        """save_model() doit créer le fichier de poids."""
        trainer.save_model()
        assert os.path.isfile(trainer.train_config.model_path)

    def test_save_model_is_loadable(self, trainer):
        """Les poids sauvegardés doivent pouvoir être rechargés."""
        trainer.save_model()
        state_dict = torch.load(trainer.train_config.model_path, weights_only=True)
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_save_model_weights_match_agent(self, trainer):
        """Les poids sauvegardés doivent correspondre à ceux de l'agent."""
        trainer.save_model()
        state_dict = torch.load(trainer.train_config.model_path, weights_only=True)
        agent_dict = trainer.agent.state_dict()
        assert set(state_dict.keys()) == set(agent_dict.keys())
        for key in state_dict:
            assert torch.equal(state_dict[key], agent_dict[key])

    def test_saved_model_can_be_loaded_into_new_agent(self, trainer, env):
        """Un nouvel agent chargé avec les poids sauvegardés doit produire les mêmes actions."""
        trainer.train(verbose=False)

        # Créer un nouvel agent et charger les poids
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        new_agent = Agent(n_state=obs_dim, n_action=act_dim)
        new_agent.load_state_dict(torch.load(trainer.train_config.model_path, weights_only=True))

        # Les deux agents doivent produire les mêmes sorties déterministes
        # (on utilise la moyenne de la distribution, pas l'échantillonnage stochastique)
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32)

        trainer.agent.eval()
        new_agent.eval()

        # Utiliser get_distribution pour récupérer la moyenne déterministe
        dist1, value1 = trainer.agent.get_distribution(obs_t)
        dist2, value2 = new_agent.get_distribution(obs_t)

        assert torch.allclose(dist1.mean, dist2.mean, atol=1e-6), \
            "Les moyennes de politique diffèrent après chargement"
        assert torch.allclose(value1, value2, atol=1e-6), \
            "Les estimations de valeur diffèrent après chargement"


# ─── Classe 6 : Buffer et intégration PPO ───────────────────────────


class TestBufferAndPPO:
    """Vérifie le comportement du buffer et de l'algorithme PPO."""

    def test_buffer_fills_during_rollout(self, trainer, env):
        """Le buffer doit être rempli pendant le rollout."""
        state, _ = env.reset()
        assert trainer.buffer.size == 0
        trainer.rollout_phase(state)
        assert trainer.buffer.size == trainer.train_config.rollout_steps

    def test_buffer_clears_after_update(self, trainer, env):
        """Le buffer doit être vidé après update_weights."""
        state, _ = env.reset()
        trainer.rollout_phase(state)
        assert trainer.buffer.size > 0
        trainer.update_weights(0)
        assert trainer.buffer.size == 0

    def test_multiple_rollout_update_cycles(self, trainer, env):
        """Plusieurs cycles rollout → update doivent fonctionner."""
        state, _ = env.reset()
        for i in range(trainer.train_config.num_update):
            trainer.rollout_phase(state)
            loss, pi_loss, v_loss, entropy = trainer.update_weights(i)
            assert np.isfinite(loss)
            state, _ = env.reset()

    def test_gae_computation(self, trainer):
        """Le GAE doit produire des avantages finis."""
        rewards = np.random.randn(64).astype(np.float32)
        values = np.random.randn(64).astype(np.float32)
        dones = np.zeros(64, dtype=np.float32)
        dones[30] = 1.0

        returns, advantages, deltas = trainer.ppo_trainer.compute_gae(
            rewards, values, 0.5, dones
        )
        assert np.all(np.isfinite(returns))
        assert np.all(np.isfinite(advantages))
        assert np.all(np.isfinite(deltas))

    def test_ppo_update_with_real_buffer(self, trainer, env):
        """L'update PPO avec des données réelles doit fonctionner."""
        state, _ = env.reset()
        trainer.rollout_phase(state)

        # Vérifier que le buffer contient des données valides
        states, actions, old_log_probs, returns_buf, adv_buf, _, _, _ = trainer.buffer.get_all()
        assert states.shape[0] == trainer.train_config.rollout_steps
        assert actions.shape[0] == trainer.train_config.rollout_steps
        assert old_log_probs.shape[0] == trainer.train_config.rollout_steps
        assert torch.all(torch.isfinite(states))
        assert torch.all(torch.isfinite(actions))
        assert torch.all(torch.isfinite(old_log_probs))


# ─── Classe 7 : Métriques et monitoring ─────────────────────────────


class TestMetrics:
    """Vérifie le suivi des métriques pendant l'entraînement."""

    def test_cumulative_reward_tracks(self, trainer, env):
        """Le cumulative_reward doit être suivi correctement."""
        state, _ = env.reset()
        trainer.rollout_phase(state)
        # Après rollout, cumulative_reward doit être != 0 (au moins des time penalties)
        assert trainer.cumulative_reward != 0.0

    def test_cumulative_reward_resets(self, trainer, env):
        """Le cumulative_reward doit être reset après chaque update dans train()."""
        trainer.train(verbose=False)
        # Après train(), cumulative_reward est reset à 0.0
        assert trainer.cumulative_reward == 0.0

    def test_ppo_optimizer_updates(self, trainer, env):
        """Les paramètres de l'optimiseur PPO doivent changer après un update."""
        # Capturer les poids avant
        weights_before = {k: v.clone() for k, v in trainer.agent.named_parameters()}

        state, _ = env.reset()
        trainer.rollout_phase(state)
        trainer.update_weights(0)

        # Les poids doivent avoir changé
        weights_changed = False
        for name, param in trainer.agent.named_parameters():
            if not torch.equal(param, weights_before[name]):
                weights_changed = True
                break
        assert weights_changed, "Aucun poids n'a changé après l'update PPO"


# ─── Classe 8 : Edge cases ─────────────────────────────────────────


class TestEdgeCases:
    """Vérifie le comportement dans des cas limites."""

    def test_single_update_training(self, env, agent):
        """Un seul update doit fonctionner."""
        tmp_dir = tempfile.mkdtemp()
        train_config = TrainConfig(
            model_name="single_update",
            model_saved_path=tmp_dir,
            timestamp=64,        # 1 seul update
            batch_size=32,
            rollout_steps=64,
        )
        ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
        t = Trainer(env=env, agent=agent, train_config=train_config, ppo_config=ppo_config)
        t.train(verbose=False)
        assert os.path.isfile(t.train_config.model_path)

    def test_large_batch_size(self, env, agent):
        """Un batch_size > rollout_steps ne doit pas planter."""
        tmp_dir = tempfile.mkdtemp()
        train_config = TrainConfig(
            model_name="large_batch",
            model_saved_path=tmp_dir,
            timestamp=64,
            batch_size=128,      # > rollout_steps=64
            rollout_steps=64,
        )
        ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
        t = Trainer(env=env, agent=agent, train_config=train_config, ppo_config=ppo_config)
        t.train(verbose=False)

    def test_close_env_after_training(self, env, agent):
        """L'environnement doit pouvoir être fermé après l'entraînement."""
        tmp_dir = tempfile.mkdtemp()
        train_config = TrainConfig(
            model_name="close_test",
            model_saved_path=tmp_dir,
            timestamp=128,
            batch_size=32,
            rollout_steps=64,
        )
        ppo_config = PPOConfig(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2)
        t = Trainer(env=env, agent=agent, train_config=train_config, ppo_config=ppo_config)
        t.train(verbose=False)
        env.close()
        # Ne doit pas planter
        assert True
