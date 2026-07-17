"""
train.py
========
Pipeline d'entraînement PPO pour l'agent drone agricole.

Hérite de ``BaseTrain`` (rl_template) et surcharge uniquement les
méthodes qui posent problème pour les actions continues (6D) :
  - ``rollout_phase`` : corrige le ``.item()`` qui plante pour un tableau 6D.
  - ``update_weights`` : corrige le mismatch de shape ``log_prob (batch,6)``
    vs ``advantages (batch,)`` en sommant les log-probs par dimension d'action.

Le reste (``save_model``, ``__init__``, ``require_buffer_size``) est
réutilisé directement depuis ``BaseTrain``.

Ajoute tqdm (barre de progression) et wandb (logging des métriques).
"""

import os
import sys
import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from rl_template.train import BaseTrain
from rl_template.common import Buffer
from rl_template.algorithms.ppo.ppo import PPOTrainer
from rl_template.config import PPOConfig, TrainConfig

# wandb est optionnel — import silencieux si non installé
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer(BaseTrain):
    """Pipeline d'entraînement PPO pour drone agricole.

    Hérite de ``BaseTrain`` et surcharge uniquement ce qui est nécessaire
    pour gérer les actions continues (espace d'action 6D) :

    - ``rollout_phase`` : convertit les actions tenseurs en tableaux numpy
      et stocke les log-probs comme scalaires (somme sur les dimensions
      d'action) pour rester compatible avec ``Buffer.insert()``.
    - ``update_weights`` : implémente la boucle PPO directement avec
      sommation des log-probs avant le calcul du ratio, car
      ``PPOTrainer.update()`` de rl_template est incompatible avec les
      actions continues (mismatch de shape).
    - ``save_model`` : réutilisé tel quel depuis ``BaseTrain``.
    """

    def __init__(self, env, agent, train_config, ppo_config, wandb_config=None):
        """Initialise le pipeline d'entraînement.

        Crée le buffer avec ``action_shape=(n_action,)`` pour stocker
        les actions continues, et instancie le ``PPOTrainer``.

        Args:
            env: Environnement Gymnasium (doit exposer ``observation_space``
                et ``action_space``).
            agent: Agent RL (doit hériter de ``BaseAgent``).
            train_config: Configuration d'entraînement (``TrainConfig``).
            ppo_config: Hyperparamètres PPO (``PPOConfig``).
            wandb_config: Configuration wandb optionnelle (dict avec
                ``project``, ``entity``, ``name``, ``config``).
        """
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        buffer = Buffer(
            step=train_config.rollout_steps,
            state_shape=(obs_dim,),
            action_shape=(act_dim,),
        )

        ppo_trainer = PPOTrainer(model=agent, ppo_config=ppo_config)

        super().__init__(
            agent=agent,
            env=env,
            buffer=buffer,
            train_config=train_config,
            ppo_trainer=ppo_trainer,
        )

        self.wandb_config = wandb_config
        self._wandb_initialized = False

    # ------------------------------------------------------------------
    # Surcharge : rollout_phase — corrige .item() pour actions continues
    # ------------------------------------------------------------------
    def rollout_phase(self, state):
        """Collecte d'expérience avec gestion des actions continues.

        Surcharge de ``BaseTrain.rollout_phase`` pour corriger :
          - ``action_t.item()`` → ``action_t.cpu().numpy()`` (6D array)
          - ``log_prob`` scalaire → ``log_prob.sum().item()`` (somme sur
            les 6 dimensions d'action pour compatibilité Buffer)
          - ``value`` tensor → ``value.item()`` (scalaire)

        Args:
            state: Observation initiale (numpy array, forme ``(obs_dim,)``).
        """
        for _ in range(self.train_config.rollout_steps):
            with torch.inference_mode():
                state_t = torch.tensor(state, dtype=torch.float32)
                action_t, log_prob, _, value = self.agent.get_action(state_t)

            # Convertir en numpy pour l'environnement
            action_np = action_t.cpu().numpy()
            # Sommer les log-probs sur les dimensions d'action → scalaire
            log_prob_scalar = log_prob.sum().item()

            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.insert(
                state=state,
                action=action_np,
                old_log_prob=log_prob_scalar,
                reward=reward,
                value=value.item(),
                dones=int(done),
            )
            self.cumulative_reward += reward

            if done:
                state, _ = self.env.reset()
            else:
                state = next_state

        # Valeur de bootstrap pour le GAE
        with torch.inference_mode():
            state_t = torch.tensor(state, dtype=torch.float32)
            _, _, _, next_value = self.agent.get_action(state_t)
        self.last_value = next_value.item()

    # ------------------------------------------------------------------
    # Surcharge : update_weights — PPO avec log-probs sommés
    # ------------------------------------------------------------------
    def update_weights(self, step):
        """Calcule le GAE et met à jour les poids via PPO.

        Surcharge de ``BaseTrain.update_weights`` car
        ``PPOTrainer.update()`` de rl_template est incompatible avec les
        actions continues :
          - ``new_log_probs`` a la forme ``(batch, n_action)``
          - ``old_log_probs`` a la forme ``(batch,)``
          - Le soustraction ``new - old`` provoque un mismatch de shape

        Cette méthode implémente la boucle PPO directement en sommant
        les log-probs sur les dimensions d'action avant le calcul du
        ratio.

        Args:
            step: Étape d'entraînement courante (pour le LR decay).

        Returns:
            Tuple ``(loss, pi_loss, v_loss, entropy)`` (floats).
        """
        if self.buffer.size < self.require_buffer_size:
            from rl_template.errors import EmptyBufferError

            raise EmptyBufferError(self.buffer.size, self.require_buffer_size)

        # --- GAE (réutilise PPOTrainer.compute_gae) ---
        with torch.inference_mode():
            returns, adv, _ = self.ppo_trainer.compute_gae(
                self.buffer.rewards,
                self.buffer.values,
                self.last_value,
                self.buffer.dones,
            )
            self.buffer.insert_returns(returns, adv)

        # --- LR decay ---
        self.ppo_trainer.lr_decay(
            self.ppo_trainer.ppo_config.lr,
            self.train_config.timestamp,
            step,
        )

        # --- Extraction des données du buffer ---
        states, actions, old_log_probs, returns_buf, adv_buf, _, _, _ = (
            self.buffer.get_all()
        )

        # Normalisation
        advantages = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
        returns_norm = (returns_buf - returns_buf.mean()) / (returns_buf.std() + 1e-8)

        # --- Boucle PPO ---
        dataset_size = states.size(0)
        batch_size = self.train_config.batch_size
        num_batch = dataset_size // batch_size
        epochs = 10

        size_total = int((dataset_size / batch_size) * epochs)
        epoch_losses = torch.zeros(size_total, dtype=torch.float32)
        epoch_pi_losses = torch.zeros(size_total)
        epoch_v_losses = torch.zeros(size_total)
        epoch_entropies = torch.zeros(size_total)
        index_loss = 0

        batch_rollout = torch.arange(0, dataset_size, batch_size)
        mse_loss = nn.MSELoss()
        ppo_cfg = self.ppo_trainer.ppo_config

        for _ in range(epochs):
            shuffle_index = batch_rollout[torch.randperm(num_batch)]

            for start in shuffle_index:
                end = start + batch_size
                idx = torch.arange(start, end)
                if idx.numel() == 0:
                    continue

                _, new_log_probs, dist_entropy, new_values = self.agent.get_action(
                    states[idx], actions[idx]
                )

                # Sommer les log-probs sur les dimensions d'action
                new_log_probs_sum = new_log_probs.sum(dim=-1)
                old_log_probs_sum = old_log_probs[idx]

                logratio = new_log_probs_sum - old_log_probs_sum
                ratio = torch.exp(logratio)

                idx_adv = advantages[idx].flatten()
                surr1 = ratio * idx_adv
                surr2 = torch.clamp(ratio, 1.0 - ppo_cfg.clip_eps, 1.0 + ppo_cfg.clip_eps) * idx_adv

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = mse_loss(new_values.flatten(), returns_norm[idx].flatten())
                entropy_loss = dist_entropy.sum(dim=-1).mean()

                loss = policy_loss + ppo_cfg.value_coef * value_loss + ppo_cfg.ent_coef * entropy_loss

                self.ppo_trainer.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                self.ppo_trainer.optimizer.step()

                epoch_losses[index_loss] = loss.detach()
                epoch_pi_losses[index_loss] = policy_loss.detach()
                epoch_v_losses[index_loss] = value_loss.detach()
                epoch_entropies[index_loss] = entropy_loss.detach()
                index_loss += 1

        self.buffer.clear()

        return (
            epoch_losses[:index_loss].mean().item(),
            epoch_pi_losses[:index_loss].mean().item(),
            epoch_v_losses[:index_loss].mean().item(),
            epoch_entropies[:index_loss].mean().item(),
        )

    # ------------------------------------------------------------------
    # save_model : réutilise BaseTrain.save_model() — pas de surcharge
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Initialisation wandb
    # ------------------------------------------------------------------
    def _init_wandb(self):
        """Initialise wandb si la configuration est fournie."""
        if self.wandb_config is None or not WANDB_AVAILABLE:
            return
        if self._wandb_initialized:
            return

        wandb.init(
            project=self.wandb_config.get("project", "agri-drone-rl"),
            entity=self.wandb_config.get("entity", None),
            name=self.wandb_config.get("name", "ppo-training"),
            config={
                **self.wandb_config.get("config", {}),
                "timestamp": self.train_config.timestamp,
                "rollout_steps": self.train_config.rollout_steps,
                "batch_size": self.train_config.batch_size,
                "lr": self.ppo_trainer.ppo_config.lr,
                "gamma": self.ppo_trainer.ppo_config.gamma,
                "gae_lambda": self.ppo_trainer.ppo_config.gae_lambda,
                "clip_eps": self.ppo_trainer.ppo_config.clip_eps,
            },
        )
        self._wandb_initialized = True

    def _log_wandb(self, metrics: dict, step: int):
        """Log les métriques dans wandb."""
        if not self._wandb_initialized:
            return
        wandb.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Boucle d'entraînement principale
    # ------------------------------------------------------------------
    def train(self, verbose=True):
        """Exécute la boucle complète d'entraînement PPO.

        Utilise tqdm pour la barre de progression et wandb pour le
        logging des métriques (si configuré).

        Args:
            verbose: Si ``True``, affiche les métriques à chaque update.
        """
        self._init_wandb()
        state, _ = self.env.reset()
        total_updates = self.train_config.num_update

        pbar = tqdm(range(total_updates), desc="Entraînement PPO", disable=not verbose)

        for update_step in pbar:
            # Phase de rollout
            self.rollout_phase(state)

            # Mise à jour des poids
            loss, pi_loss, v_loss, entropy = self.update_weights(update_step)

            # Métriques
            current_lr = self.ppo_trainer.optimizer.param_groups[0]["lr"]
            metrics = {
                "train/loss": loss,
                "train/policy_loss": pi_loss,
                "train/value_loss": v_loss,
                "train/entropy": entropy,
                "train/cumulative_reward": self.cumulative_reward,
                "train/learning_rate": current_lr,
                "train/buffer_size": self.buffer.size,
            }

            # Log wandb
            self._log_wandb(metrics, step=update_step)

            # Affichage dans tqdm
            if verbose:
                pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "π": f"{pi_loss:.4f}",
                    "V": f"{v_loss:.4f}",
                    "H": f"{entropy:.4f}",
                    "R": f"{self.cumulative_reward:.1f}",
                })

            # Réinitialiser pour le prochain rollout
            state, _ = self.env.reset()
            self.cumulative_reward = 0.0

        # Sauvegarde finale
        self.save_model()

        if self._wandb_initialized:
            wandb.finish()

        if verbose:
            print(f"\n✅ Entraînement terminé — modèle sauvegardé: {self.train_config.model_path}")
