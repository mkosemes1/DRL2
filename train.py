"""
train.py
========
Pipeline d'entraînement PPO pour l'agent drone agricole.

Classe Trainer qui hérite de BaseTrain et orchestre la boucle
rollout → GAE → update PPO → sauvegarde. Gère correctement les
actions continues (espace d'action 6D) et la conversion
numpy ↔ torch entre l'environnement et l'agent.
"""

import os
import sys
import torch
import numpy as np
from torch import nn
import tqdm

from rl_template.train import BaseTrain, Buffer, PPOTrainer, EmptyBufferError
from rl_template.config import PPOConfig, TrainConfig


class Trainer(BaseTrain):
    """Pipeline d'entraînement PPO pour drone agricole.

    Hérite de ``BaseTrain`` et implémente les méthodes abstraites :
      - ``rollout_phase`` : collecte d'expérience avec actions continues.
      - ``update_weights`` : calcul GAE + mise à jour PPO.
      - ``save_model`` : sauvegarde des poids de l'agent.

    Cette classe corrige les problèmes de ``BaseTrain`` pour les
    espaces d'action continus :
      - Conversion numpy → torch avant de passer l'état à l'agent.
      - Stockage des actions continues (tableau 6D) dans le buffer.
      - Stockage des log-probabilités par dimension d'action (shape
        ``(T, n_action)``) au lieu d'un scalaire, compatible avec
        ``PPOTrainer.update``.

    Attributes:
        agent: Réseau de neurones acteur-critique (``Agent``).
        env: Environnement Gymnasium (``AgriDroneEnv``).
        buffer: Buffer de rollout pré-alloué (``Buffer``).
        ppo_trainer: Entraîneur PPO (``PPOTrainer``).
        train_config: Configuration d'entraînement (``TrainConfig``).
    """

    def __init__(self, env, agent, train_config, ppo_config):
        """Initialise le pipeline d'entraînement.

        Crée le buffer avec les dimensions correctes pour les actions
        continues, redimensionne ``old_log_probs`` pour stocker les
        log-probabilités par dimension d'action, et instancie le
        ``PPOTrainer``.

        Args:
            env: Environnement Gymnasium (doit exposer ``observation_space``
                et ``action_space``).
            agent: Agent RL (doit hériter de ``BaseAgent``).
            train_config: Configuration d'entraînement (``TrainConfig``).
            ppo_config: Configuration PPO (``PPOConfig``).
        """
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        buffer = Buffer(
            step=train_config.rollout_steps,
            state_shape=(obs_dim,),
            action_shape=(act_dim,),
        )
        # Redimensionner old_log_probs pour stocker les log-probs
        # par dimension d'action (compatible avec PPOTrainer.update)
        buffer.old_log_probs = np.zeros(
            (train_config.rollout_steps, act_dim), dtype=np.float32
        )

        ppo_trainer = PPOTrainer(model=agent, ppo_config=ppo_config)

        super().__init__(
            agent=agent,
            env=env,
            buffer=buffer,
            train_config=train_config,
            ppo_trainer=ppo_trainer,
        )

    def rollout_phase(self, state):
        """Collecte d'expérience avec gestion correcte des actions continues.

        Contrairement à ``BaseTrain.rollout_phase``, cette méthode :
          - Convertit l'état numpy en tenseur torch avant de le passer
            à l'agent (``nn.Linear`` nécessite des tenseurs).
          - Stocke les actions continues (6D) dans le buffer.
          - Stocke les log-probabilités par dimension d'action.
          - Marque ``dones=1`` quand l'épisode se termine naturellement
            OU par troncation (timeout).

        Args:
            state: Observation initiale (numpy array, forme ``(obs_dim,)``).
        """
        for _ in range(self.train_config.rollout_steps):
            with torch.inference_mode():
                state_t = torch.tensor(state, dtype=torch.float32)
                action_t, log_prob, _, value = self.agent.get_action(state_t)

            action_np = action_t.cpu().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.insert(
                state=state,
                action=action_np,
                old_log_prob=log_prob.cpu().numpy(),
                reward=reward,
                value=value.item(),
                dones=int(done),
            )
            self.cumulative_reward += reward

            if done:
                state, _ = self.env.reset()
            else:
                state = next_state

        with torch.inference_mode():
            state_t = torch.tensor(state, dtype=torch.float32)
            _, _, _, next_value = self.agent.get_action(state_t)
        self.last_value = next_value.item()

    def update_weights(self, step):
        """Calcule le GAE et met à jour les poids via PPO.

        Pour les actions continues (espace d'action 6D), les
        log-probabilités sont sommées sur les dimensions d'action
        avant de calculer le ratio PPO. Le ``ppo_trainer.update()``
        de rl_template ne gère pas correctement les actions
        continues (incompatibilité de shape entre ``log_prob``
        ``(batch, n_action)`` et ``advantages`` ``(batch,)``).
        Cette méthode implémente la boucle PPO directement.

        Args:
            step: Étape d'entraînement courante (pour le decay du LR).

        Returns:
            Tuple ``(loss, pi_loss, v_loss, entropy)`` où chaque
            élément est un ``float``.

        Raises:
            EmptyBufferError: Si le buffer contient moins de
                ``require_buffer_size`` entrées.
        """
        if self.buffer.size < self.require_buffer_size:
            raise EmptyBufferError(self.buffer.size, self.require_buffer_size)

        rewards_list = self.buffer.rewards
        values_list = self.buffer.values
        dones_list = self.buffer.dones

        with torch.inference_mode():
            returns, adv, _ = self.ppo_trainer.compute_gae(
                rewards_list, values_list, self.last_value, dones_list
            )
            self.buffer.insert_returns(returns, adv)

        # --- Mise à jour PPO adaptée aux actions continues ---
        # Utilise le même optimiseur que ppo_trainer
        self.ppo_trainer.lr_decay(
            self.ppo_trainer.ppo_config.lr,
            self.train_config.timestamp,
            step,
        )

        states, actions, old_log_probs, returns_buf, adv_buf, _, _, _ = (
            self.buffer.get_all()
        )

        # Normaliser les avantages et les returns
        advantages = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
        returns_norm = (returns_buf - returns_buf.mean()) / (returns_buf.std() + 1e-8)

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
                # pour obtenir un scalaire par pas de temps
                new_log_probs_sum = new_log_probs.sum(dim=-1)
                old_log_probs_sum = old_log_probs[idx].sum(dim=-1) if old_log_probs[idx].dim() > 1 else old_log_probs[idx]

                logratio = new_log_probs_sum - old_log_probs_sum
                ratio = torch.exp(logratio)

                idx_adv = advantages[idx].flatten()
                surr1 = ratio * idx_adv
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.ppo_trainer.ppo_config.clip_eps,
                        1.0 + self.ppo_trainer.ppo_config.clip_eps,
                    )
                    * idx_adv
                )

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = mse_loss(
                    new_values.flatten(), returns_norm[idx].flatten()
                )
                entropy_loss = dist_entropy.sum(dim=-1).mean()

                loss = (
                    policy_loss
                    + self.ppo_trainer.ppo_config.value_coef * value_loss
                    + self.ppo_trainer.ppo_config.ent_coef * entropy_loss
                )

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

    def save_model(self):
        """Sauvegarde les poids de l'agent dans ``train_config.model_path``.

        Crée le répertoire de destination s'il n'existe pas.
        """
        os.makedirs(os.path.dirname(self.train_config.model_path), exist_ok=True)
        torch.save(self.agent.state_dict(), self.train_config.model_path)

    def train(self, verbose=False):
        """Exécute la boucle complète d'entraînement PPO.

        La boucle effectue ``num_update`` itérations de :
          1. Rollout (collecte de ``rollout_steps`` transitions).
          2. Mise à jour des poids PPO (GAE + clipping).
          3. Réinitialisation de l'environnement.

        Args:
            verbose: Si ``True``, affiche les métriques à chaque update.
        """
        state, _ = self.env.reset()

        for update_step in range(self.train_config.num_update):
            self.rollout_phase(state)

            loss, pi_loss, v_loss, entropy = self.update_weights(update_step)

            # Réinitialiser l'environnement pour le prochain rollout
            state, _ = self.env.reset()

            if verbose:
                print(
                    f"Update {update_step + 1}/{self.train_config.num_update} "
                    f"| Loss: {loss:.4f} | Pi: {pi_loss:.4f} "
                    f"| V: {v_loss:.4f} | Ent: {entropy:.4f}"
                )

        self.save_model()
