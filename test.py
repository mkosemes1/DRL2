"""
test.py
Script d'évaluation visuelle pour l'agent drone agricole.
Charge les poids sauvegardés et lance l'environnement en mode 'human'.
"""

import os
import time
import torch
import numpy as np

from environment.env import AgriDroneEnv
from agent.model import Agent

# Reprendre exactement la même configuration que pour l'entraînement
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

def test_agent(checkpoint_path="./checkpoints/agriDrone.pt", num_episodes=5):
    # 1. Initialiser l'environnement avec le rendu visuel
    env = AgriDroneEnv(config, render_mode="human")
    
    # 2. Initialiser l'agent avec les bonnes dimensions
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    agent = Agent(obs_dim[0], act_dim[0])
    
    # 3. Charger les poids de l'entraînement
    if os.path.exists(checkpoint_path):
        print(f"✅ Chargement des poids depuis {checkpoint_path}...")
        # On charge sur CPU pour l'inférence visuelle
        agent.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        agent.eval()  # Passer le modèle en mode évaluation
    else:
        print(f"⚠️ Aucun checkpoint trouvé à {checkpoint_path}. L'agent agira avec des poids aléatoires.")

    # 4. Boucle d'évaluation
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while True:
            # Convertir l'état en tenseur pour PyTorch
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Récupérer l'action de l'agent sans calculer les gradients
            with torch.no_grad():
                # On utilise la fonction get_action de ton modèle
                action_t, _, _, _ = agent.get_action(state_tensor)
            
            # Reconvertir l'action en numpy array
            action = action_t.squeeze(0).numpy()
            
            # Faire un pas dans l'environnement
            next_state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            state = next_state
            step += 1
            #print(total_reward)
            
            #done = terminated or truncated
            
            # Imprimer quelques infos utiles de temps en temps
            if step % 50 == 0:
                dist = info.get('distance_to_goal', 0)
                tank = info.get('water_tank_level', 0)
                print(f"Pas {step:03d} | Dist: {dist:.2f}m | Réservoir: {tank:.1f}% | Récompense: {reward:.2f}")
            
            if info.get('just_watered'):
                print(f"💧 GROUPE ARROSÉ AU PAS {step} ! (+5.0 reward)")
                
            if info.get('just_refilled'):
                print(f"🔄 RÉSERVOIR REMPLI AU PAS {step} ! (+1.0 reward)")

            # Ralentir un tout petit peu la boucle pour que l'œil humain puisse suivre
            time.sleep(0.02)

            if terminated or truncated:
                print(f"Fin d'épisode ! Crash: {info.get('crashed')}. Reset de l'environnement.")
                next_state, info = env.reset()
            
        print(f"Épisode {ep+1} terminé en {step} pas. Récompense totale: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test_agent()
