#!/usr/bin/env python3
"""
run_train.py
============
Script pour lancer l'entraînement PPO du drone agricole.

Utilisation :
    uv run python run_train.py                     # Entraînement standard
    uv run python run_train.py --episodes 100      # Plus d'updates
    uv run python run_train.py --wandb             # Avec logging wandb
    uv run python run_train.py --no-tqdm           # Sans barre de progression
"""

import sys
import os
import argparse

# Configuration des chemins
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environment"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

from agri_drone_env import AgriDroneEnv
from model import Agent
from train import Trainer
from rl_template.config import PPOConfig, TrainConfig


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement PPO du drone agricole")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Nombre d'updates PPO (défaut: 100)")
    parser.add_argument("--rollout-steps", type=int, default=128,
                        help="Steps par rollout (défaut: 128)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Taille des mini-batches (défaut: 64)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (défaut: 3e-4)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (défaut: 0.99)")
    parser.add_argument("--wandb", action="store_true",
                        help="Activer le logging wandb")
    parser.add_argument("--wandb-project", type=str, default="agri-drone-rl",
                        help="Nom du projet wandb (défaut: agri-drone-rl)")
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Désactiver la barre de progression")
    parser.add_argument("--saved-dir", type=str, default="saved_models",
                        help="Répertoire de sauvegarde (défaut: saved_models)")
    return parser.parse_args()


def main():
    """Boucle principale d'entraînement."""
    args = parse_args()

    print("🚁 Entraînement PPO — Drone Agricole")
    print("=" * 50)

    # ── Configuration de l'environnement ──
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
            "urdf_path": os.path.join(os.path.dirname(__file__),
                                       "environment", "agri_hexacopter_pro.urdf"),
        },
        "simulation": {"dt": 0.02, "max_episode_steps": 1000},
        "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
        "water_task": {
            "basin_position": [15.0, 15.0, 0.5],
            "basin_refill_radius": 3.0,
            "water_consumption": 2.0,
            "watering_proximity": 2.0,
            "num_plant_groups": 5,
        },
    }

    # ── Création de l'environnement ──
    print("📦 Création de l'environnement...")
    env = AgriDroneEnv(config=config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"   Obs dim: {obs_dim} | Act dim: {act_dim}")

    # ── Création de l'agent ──
    print("🤖 Création de l'agent...")
    agent = Agent(n_state=obs_dim, n_action=act_dim)
    num_params = sum(p.numel() for p in agent.parameters())
    print(f"   Paramètres: {num_params:,}")

    # ── Configuration d'entraînement ──
    timestamp = args.episodes * args.rollout_steps
    train_config = TrainConfig(
        model_name="agri_drone_ppo",
        model_saved_path=args.saved_dir,
        timestamp=timestamp,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
    )

    ppo_config = PPOConfig(
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        value_coef=0.5,
    )

    # ── Configuration wandb (optionnel) ──
    wandb_config = None
    if args.wandb:
        wandb_config = {
            "project": args.wandb_project,
            "entity": None,
            "name": "ppo-agri-drone",
            "config": {
                "lr": args.lr,
                "gamma": args.gamma,
                "rollout_steps": args.rollout_steps,
                "batch_size": args.batch_size,
                "num_plant_groups": 5,
            },
        }
        print(f"📊 wandb activé — projet: {args.wandb_project}")

    # ── Création du Trainer ──
    print("🚀 Création du Trainer...")
    trainer = Trainer(
        env=env,
        agent=agent,
        train_config=train_config,
        ppo_config=ppo_config,
        wandb_config=wandb_config,
    )

    print(f"   Updates: {train_config.num_update} | Rollout: {args.rollout_steps} steps")
    print(f"   Batch: {args.batch_size} | LR: {args.lr} | Gamma: {args.gamma}")
    print(f"   Sauvegarde: {train_config.model_path}")
    print("=" * 50)

    # ── Lancement de l'entraînement ──
    trainer.train(verbose=not args.no_tqdm)

    print("\n" + "=" * 50)
    print("✅ Entraînement terminé !")
    print(f"   Modèle sauvegardé: {train_config.model_path}")
    env.close()


if __name__ == "__main__":
    main()
