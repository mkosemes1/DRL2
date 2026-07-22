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
import wandb
from agent.model import Agent
from environment.env import AgriDroneEnv
from tqdm import tqdm
from rl_template.train import BaseTrain
from rl_template.common import Buffer
from rl_template.algorithms.ppo.ppo import PPOTrainer
from rl_template.config import PPOConfig, TrainConfig


#Configuration
config = {
    "world": {
        "size_x": 60.0,
        "size_y": 60.0,
        "ground_z": 0.0,
        "size_z": 50.0,
        "field_cells_x": 20,
        "field_cells_y": 20,
    },
    "drone": {
        "dry_mass": 10.0,
        "payload_mass_full": 5.0,
        "gravity": 9.81,
        "max_thrust_total": 350.0,
        "drag_coefficient": 0.08,
        "max_tilt_angle_rad": 0.5236,
        "max_angular_rate": 3.0,
        "attitude_time_constant": 0.08,
        "urdf_path": os.path.join("environment", "agri_hexacopter_pro.urdf"),
    },
    "simulation": {"dt": 0.02, "max_episode_steps": 5000},
    "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
    "water_task": {
        "basin_position": [15.0, 15.0, 0.5],
        "basin_refill_radius": 3.0,
        "water_consumption": 2.0,
        "watering_proximity": 2.0,
        "num_plant_groups": 5,
    },
}


train_config = TrainConfig(device="cuda:0", model_name="agriDrone", model_saved_path="./checkpoints", timestamp=10_000_000)
ppo_config = PPOConfig(clip_eps=0.2, ent_coef=0.001)
env = AgriDroneEnv(config)
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape
buffer = Buffer(step=train_config.rollout_steps, state_shape=obs_dim, action_shape=act_dim)
agent = Agent(obs_dim[0], act_dim[0])
ppo_trainer = PPOTrainer(agent, ppo_config)
trainer = BaseTrain(agent, env, buffer, train_config, ppo_trainer)

wandb.login()

log_config = {
        'epochs': train_config.num_update,
        'lr': ppo_config.lr,
        'gamma': ppo_config.gamma,
        'gae_lambda': ppo_config.gae_lambda,
        'clip_eps': ppo_config.clip_eps,
        'ent_coef': ppo_config.ent_coef,
        'value_coef': ppo_config.value_coef,
        }

with wandb.init(project="drone", config=log_config) as run:
    for step in tqdm(range(train_config.num_update)):
        state, _ = env.reset()
        trainer.rollout_phase(state)
        loss, policy_loss, value_loss, entropy_loss = trainer.update_weights(step)
        run.log({'Loss': loss,
                 'policy loss': policy_loss,
                 'value loss': value_loss,
                 'entropy loss': entropy_loss,
                 'reward': trainer.cumulative_reward})
    env.close()
    trainer.save_model()
