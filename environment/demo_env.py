#!/usr/bin/env python3
"""
demo_env.py
===========
Démonstration avec AgriDroneEnv (champs affichés).
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.agri_drone_env import AgriDroneEnv

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
        "urdf_path": os.path.join(os.path.dirname(__file__), "agri_hexacopter_pro.urdf")
    },
    "simulation": {"dt": 0.02, "max_episode_steps": 1000},
    "normalization": {"max_velocity": 50.0, "max_distance": 100.0},
}

env = AgriDroneEnv(config=config, render_mode="human")
obs, info = env.reset()

print("=== VOL AVEC CHAMPS AFFICHÉS ===")
action = np.array([0.8, 0.0, 0.5, 0.0, -1.0, -1.0], dtype=np.float32)

for step in range(400):
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if step % 10 == 0:
        state = env.dynamics.state
        speed_h = np.sqrt(state.vx**2 + state.vy**2)
        print(f"Step {step:03d} | z={state.z:6.2f} m | v_h={speed_h:6.2f} m/s | x={state.x:6.2f} m")
    if terminated or truncated:
        break
    time.sleep(0.02)

env.close()
print("\n✅ Fin de la démonstration.")