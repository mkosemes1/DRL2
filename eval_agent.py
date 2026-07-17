#!/usr/bin/env python3
"""
eval_agent.py
=============
Évalue un agent sauvegardé dans l'environnement et génère un GIF.

Utilisation :
    uv run python eval_agent.py                              # Utilise le dernier modèle
    uv run python eval_agent.py --model saved_models/agri_drone_ppo.pt
    uv run python eval_agent.py --episodes 3 --frameskip 2
    uv run python eval_agent.py --output demo.gif --fps 15
"""

import sys
import os
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environment"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

from agri_drone_env import AgriDroneEnv
from model import Agent
from rl_template.config import TrainConfig


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation d'un agent drone agricole")
    parser.add_argument("--model", type=str, default=None,
                        help="Chemin vers le fichier .pt du modèle (défaut: dernier modèle)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Nombre d'épisodes à enregistrer (défaut: 1)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Nombre max de pas par épisode (défaut: 500)")
    parser.add_argument("--frameskip", type=int, default=2,
                        help="Capturer une image tous les N pas (défaut: 2)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Images par seconde du GIF (défaut: 15)")
    parser.add_argument("--output", type=str, default="eval_output.gif",
                        help="Nom du fichier de sortie (défaut: eval_output.gif)")
    parser.add_argument("--width", type=int, default=640,
                        help="Largeur de l'image en pixels (défaut: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Hauteur de l'image en pixels (défaut: 480)")
    return parser.parse_args()


def find_latest_model(saved_dir="saved_models"):
    """Trouve le dernier modèle sauvegardé dans le répertoire.

    Parcourt le répertoire ``saved_dir`` à la recherche de fichiers
    ``.pt`` et retourne le chemin du plus récent (par date de
    modification).

    Args:
        saved_dir: Répertoire contenant les modèles sauvegardés.

    Returns:
        Chemin du modèle le plus récent, ou ``None`` si aucun
        modèle n'est trouvé.
    """
    if not os.path.exists(saved_dir):
        return None
    pt_files = [f for f in os.listdir(saved_dir) if f.endswith(".pt")]
    if not pt_files:
        return None
    # Trier par date de modification (le plus récent en premier)
    pt_files.sort(key=lambda f: os.path.getmtime(os.path.join(saved_dir, f)), reverse=True)
    return os.path.join(saved_dir, pt_files[0])


def create_gif(frames, output_path, fps=15):
    """Crée un GIF à partir d'une liste d'images RGB.

    Convertit les tableaux numpy en images PIL puis les assemble
    en un fichier GIF animé.

    Args:
        frames: Liste de tableaux numpy ``(H, W, 3)`` dtype ``uint8``.
        output_path: Chemin du fichier GIF de sortie.
        fps: Images par seconde (défaut: 15).
    """
    if not frames:
        print("⚠️ Aucune image à enregistrer.")
        return

    # Convertir les tableaux numpy en images PIL
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Durée entre les images en millisecondes
    duration = int(1000 / fps)

    # Sauvegarder en GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,  # boucle infinie
    )
    print(f"✅ GIF sauvegardé: {output_path} ({len(frames)} images, {fps} fps)")


def main():
    """Évalue l'agent et génère un GIF de la trajectoire."""
    args = parse_args()

    print("🔍 Évaluation de l'agent drone agricole")
    print("=" * 50)

    # ── Trouver le modèle ──
    model_path = args.model
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("❌ Aucun modèle trouvé dans saved_models/. Entraînez d'abord avec:")
            print("   uv run python run_train.py")
            sys.exit(1)
    print(f"📦 Modèle: {model_path}")

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
        "simulation": {"dt": 0.02, "max_episode_steps": args.max_steps},
        "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
        "water_task": {
            "basin_position": [15.0, 15.0, 0.5],
            "basin_refill_radius": 3.0,
            "water_consumption": 2.0,
            "watering_proximity": 2.0,
            "num_plant_groups": 5,
        },
    }

    # ── Créer l'environnement en mode rgb_array ──
    print("🌍 Création de l'environnement (mode rgb_array)...")
    env = AgriDroneEnv(config=config, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # ── Charger l'agent ──
    print("🤖 Chargement de l'agent...")
    agent = Agent(n_state=obs_dim, n_action=act_dim)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    agent.load_state_dict(state_dict)
    agent.eval()
    print(f"   Paramètres chargés: {sum(p.numel() for p in agent.parameters()):,}")

    # ── Boucle d'évaluation ──
    all_frames = []
    for ep in range(args.episodes):
        print(f"\n🎮 Épisode {ep + 1}/{args.episodes}")
        obs, info = env.reset()
        total_reward = 0.0
        frames = []

        for step in range(args.max_steps):
            # Capturer l'image (tous les N pas)
            if step % args.frameskip == 0:
                frame = env.render()
                if frame is not None:
                    # Ajouter des informations sur l'image
                    frames.append(frame)

            # Sélectionner l'action (déterministe : moyenne de la distribution)
            with torch.inference_mode():
                obs_t = torch.tensor(obs, dtype=torch.float32)
                dist, _ = agent.get_distribution(obs_t)
                action = dist.mean  # action déterministe

            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward

            # Afficher les informations
            if step % 50 == 0:
                tank = info.get("water_tank_level", 0)
                unwatered = info.get("num_unwatered", 0)
                print(f"   Step {step:03d} | reward={total_reward:.2f} | "
                      f"tank={tank:.0f} | unwatered={unwatered}")

            if terminated or truncated:
                break

            obs = next_obs

        # Ajouter les frames de cet épisode
        all_frames.extend(frames)
        status = "✅ Mission accomplie" if terminated else "⏱️ Timeout"
        print(f"   {status} | Récompense totale: {total_reward:.2f} | Frames: {len(frames)}")

    # ── Créer le GIF ──
    print(f"\n🎬 Création du GIF ({len(all_frames)} frames)...")
    create_gif(all_frames, args.output, fps=args.fps)

    env.close()
    print("\n" + "=" * 50)
    print("✅ Évaluation terminée !")


if __name__ == "__main__":
    main()
