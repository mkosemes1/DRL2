#!/usr/bin/env python3
"""
minimal_fly_env.py
Environnement Gymnasium ultra-minimal qui utilise le modèle physique
de DroneDynamics et le rendu PyBullet.
AUCUNE logique agricole, AUCUNE limitation (batterie, collisions, etc.).
Le drone vole exactement comme dans force_fly.py.
"""

import os
import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.drone_dynamics import DroneDynamics, DroneParams


class MinimalFlyEnv(gym.Env):
    """Environnement minimal pour voler rapidement."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode

        # Paramètres physiques (identiques à force_fly)
        params = DroneParams(
            dry_mass=10.0,
            payload_mass=5.0,
            gravity=9.81,
            max_thrust_total=350.0,
            drag_coefficient=0.08,
            max_tilt_angle_rad=0.5236,
            max_angular_rate=3.0,
            attitude_time_constant=0.08,
            max_velocity=50.0
        )
        world_bounds = {"x": (-100, 100), "y": (-100, 100), "z": (0, 100)}
        dt = 0.02
        self.dynamics = DroneDynamics(params, world_bounds, dt)
        self.dynamics.reset(np.array([0.0, 0.0, 1.0]))

        # Espaces Gymnasium
        # Observation : [x, y, z, vx, vy, vz, roll, pitch, yaw] (9 dims)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        # Action : [throttle, roll, pitch, yaw] (4 dims)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Rendu PyBullet (initialisé au premier render)
        self._client = None
        self._drone_id = None
        self._prop_joints = []
        self._prop_angle = 0.0
        self._urdf_path = os.path.join(os.path.dirname(__file__), "agri_hexacopter_pro.urdf")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.dynamics.reset(np.array([0.0, 0.0, 1.0]))
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        state = self.dynamics.step(action)  # [throttle, roll, pitch, yaw]
        obs = self._get_obs()
        reward = 0.0   # pas de récompense pour l'instant
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        s = self.dynamics.state
        return np.array([s.x, s.y, s.z, s.vx, s.vy, s.vz, s.roll, s.pitch, s.yaw], dtype=np.float32)

    def render(self):
        if self.render_mode != "human":
            return

        # Initialiser PyBullet au premier appel
        if self._client is None:
            self._client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=45, cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 2])
            p.loadURDF("plane.urdf")
            self._drone_id = p.loadURDF(self._urdf_path, basePosition=[0, 0, 1])
            # Récupérer les joints des hélices
            self._prop_joints = []
            for i in range(p.getNumJoints(self._drone_id)):
                info = p.getJointInfo(self._drone_id, i)
                if info[1].decode("utf-8").startswith("j_p"):
                    self._prop_joints.append(i)

        # Mettre à jour la position du drone
        state = self.dynamics.state
        pos = [state.x, state.y, state.z]
        ori = p.getQuaternionFromEuler([state.roll, state.pitch, state.yaw])
        p.resetBasePositionAndOrientation(self._drone_id, pos, ori)

        # Animer les hélices (utiliser le throttle de la dernière action ? On le passe en attribut)
        # On garde un throttle par défaut : on peut stocker le dernier action[0] dans step
        # Pour simplifier, on utilise une vitesse constante
        self._prop_angle += 20.0
        for j in self._prop_joints:
            p.resetJointState(self._drone_id, j, self._prop_angle)

        p.stepSimulation()

    def close(self):
        if self._client is not None and p.isConnected(self._client):
            p.disconnect(self._client)
            self._client = None


# ---------- Point d'entrée pour la démonstration ----------
if __name__ == "__main__":
    env = MinimalFlyEnv(render_mode="human")
    obs, info = env.reset()

    print("=== VOL AVEC MinimalFlyEnv ===")
    # Action : throttle=0.8, roll=0.0, pitch=0.5, yaw=0.0 (comme force_fly)
    action = np.array([0.8, 0.0, 0.5, 0.0], dtype=np.float32)

    for step in range(200):
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if step % 10 == 0:
            state = env.dynamics.state
            speed_h = np.sqrt(state.vx**2 + state.vy**2)
            print(f"Step {step:03d} | z={state.z:6.2f} m | v_h={speed_h:6.2f} m/s | x={state.x:6.2f} m")
        time.sleep(0.02)

    env.close()
    print("\n✅ Fin de la démonstration.")
