#!/usr/bin/env python3
"""
test_dynamics.py
=================
Test unitaire du modèle physique DroneDynamics.
Applique une poussée constante et affiche l'altitude.
"""

import sys
import os
import numpy as np
import time

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.drone_dynamics import DroneDynamics, DroneParams

# Paramètres minimaux
params = DroneParams(
    dry_mass=10.0,
    payload_mass=5.0,
    gravity=9.81,
    max_thrust_total=350.0,
    drag_coefficient=0.08,
    max_tilt_angle_rad=0.5236,
    max_angular_rate=3.0,
    attitude_time_constant=0.08,
    max_velocity=30.0
)

world_bounds = {
    "x": (-20, 20),
    "y": (-20, 20),
    "z": (0, 30)
}

dt = 0.02
dynamics = DroneDynamics(params, world_bounds, dt)
dynamics.reset(np.array([0, 0, 1.0]))  # départ à 1 m d'altitude

print("=== Test de poussée verticale ===")
print("Throttle = 0.8 (90% de poussée), pitch=0, roll=0\n")

action = np.array([0.8, 0.0, 0.0, 0.0])  # [throttle, roll, pitch, yaw]

for step in range(200):  # 4 secondes à 50 Hz
    state = dynamics.step(action)
    if step % 10 == 0:
        print(f"Step {step:03d} | z={state.z:6.2f} m | vz={state.vz:6.2f} m/s")
    time.sleep(0.02)  # pour simuler le pas de temps réel

print("\n=== Test avec inclinaison pour avancer ===")
# On réinitialise
dynamics.reset(np.array([0, 0, 1.0]))
action = np.array([0.8, 0.0, 0.5, 0.0])  # pitch = 0.5 (≈15°)

for step in range(200):
    state = dynamics.step(action)
    if step % 10 == 0:
        v_h = np.sqrt(state.vx**2 + state.vy**2)
        print(f"Step {step:03d} | z={state.z:6.2f} m | v_h={v_h:6.2f} m/s | x={state.x:6.2f} m")
    time.sleep(0.02)
