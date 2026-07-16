"""
environment/agri_drone_env.py
=============================
Environnement Gymnasium pour drone agricole.
Version avec vol fonctionnel et affichage de la grille de champs.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

from physics.drone_dynamics import DroneDynamics, DroneParams
from physics.wind_model import WindModel
from obstacles import ObstacleManager
from utils.normalization import normalize


class FieldCell:
    """Représente une cellule du champ."""
    def __init__(self):
        self.healthy = True
        self.wet = True
        self.sprayed = False
        self.watered = False
        self.visited = False


class AgriDroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(self, config: dict, render_mode: str | None = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        # Paramètres du monde
        world_cfg = config["world"]
        self.world_bounds = {
            "x": (-world_cfg["size_x"]/2, world_cfg["size_x"]/2),
            "y": (-world_cfg["size_y"]/2, world_cfg["size_y"]/2),
            "z": (world_cfg["ground_z"], world_cfg["size_z"]),
        }
        self.field_size = (world_cfg.get("field_cells_x", 20), world_cfg.get("field_cells_y", 20))

        # Grille du champ (initialisée plus tard)
        self.field_grid = None
        self.total_cells = self.field_size[0] * self.field_size[1]

        # Paramètres du drone
        drone_cfg = config["drone"]
        params = DroneParams(
            dry_mass=drone_cfg.get("dry_mass", 10.0),
            payload_mass=drone_cfg.get("payload_mass_full", 5.0),
            gravity=drone_cfg.get("gravity", 9.81),
            max_thrust_total=drone_cfg.get("max_thrust_total", 350.0),
            drag_coefficient=drone_cfg.get("drag_coefficient", 0.08),
            max_tilt_angle_rad=drone_cfg.get("max_tilt_angle_rad", 0.5236),
            max_angular_rate=drone_cfg.get("max_angular_rate", 3.0),
            attitude_time_constant=drone_cfg.get("attitude_time_constant", 0.08),
            max_velocity=config["normalization"].get("max_velocity", 50.0),
        )
        self.dt = config["simulation"].get("dt", 0.02)
        self.max_steps = config["simulation"].get("max_episode_steps", 1000)
        self.dynamics = DroneDynamics(params, self.world_bounds, self.dt)

        # Désactiver les fonctionnalités agricoles (sauf affichage)
        self.wind = WindModel(enabled=False)
        self.obstacle_manager = ObstacleManager(self.world_bounds, min_radius=1.0, max_radius=3.0)
        self.obstacle_manager.obstacles = []

        # Espaces Gymnasium
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # État interne
        self.goal_position = np.array([20.0, 20.0, 2.0])
        self.base_position = np.array([0.0, 0.0, 0.05])
        self.battery_level = 1e6
        self.pesticide_level = 5.0
        self.water_level = 5.0
        self.maladies_traitees = 0
        self.sec_arrosees = 0
        self.returning_home = False
        self.step_count = 0

        # Rendu PyBullet (lazy)
        self._client = None
        self._drone_id = None
        self._prop_joints = []
        self._prop_angle = 0.0
        self._urdf_path = drone_cfg.get("urdf_path", os.path.join(os.path.dirname(__file__), "agri_hexacopter_pro.urdf"))
        self._field_created = False
        self._cell_ids = []  # pour stocker les IDs des cubes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.battery_level = 1e6
        self.pesticide_level = 5.0
        self.water_level = 5.0
        self.maladies_traitees = 0
        self.sec_arrosees = 0
        self.returning_home = False
        self.dynamics.reset(np.array([0.0, 0.0, 1.0]))
        self.goal_position = np.array([20.0, 20.0, 2.0])

        # Initialiser la grille du champ avec des cellules aléatoires (pour l'affichage)
        if self.field_grid is None:
            self.field_grid = [[FieldCell() for _ in range(self.field_size[1])] for _ in range(self.field_size[0])]
            # Quelques cellules malades et sèches pour l'exemple
            rng = np.random.default_rng()
            for i in range(self.field_size[0]):
                for j in range(self.field_size[1]):
                    if rng.random() < 0.15:
                        self.field_grid[i][j].healthy = False
                    if rng.random() < 0.15:
                        self.field_grid[i][j].wet = False

        # Réinitialiser les flags de visite/traitement
        for row in self.field_grid:
            for cell in row:
                cell.visited = False
                cell.sprayed = False
                cell.watered = False

        obs = self._get_obs()
        info = {"goal_position": self.goal_position.tolist()}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        flight_action = action[:4]
        state = self.dynamics.step(flight_action)

        # Mettre à jour la cellule sous le drone (pour le visuel)
        cell = self._get_cell_under_drone()
        if cell is not None:
            cell.visited = True
            # Simuler pulvérisation et irrigation (pour les couleurs)
            if action[4] > 0.0 and not cell.healthy and not cell.sprayed:
                cell.healthy = True
                cell.sprayed = True
            if action[5] > 0.0 and not cell.wet and not cell.watered:
                cell.wet = True
                cell.watered = True

        self.battery_level -= 0.001
        self.step_count += 1

        terminated = False
        truncated = (self.step_count >= self.max_steps)
        reward = 0.0

        obs = self._get_obs()
        info = {
            "distance_to_goal": np.linalg.norm(state.position() - self.goal_position),
            "battery_level": self.battery_level,
            "pesticide_level": self.pesticide_level,
            "water_level": self.water_level,
            "maladies_traitees": 0,
            "sec_arrosees": 0,
            "visited_percentage": 0.0,
            "returning_home": self.returning_home,
        }
        return obs, reward, terminated, truncated, info

    def _get_cell_under_drone(self):
        pos = self.dynamics.state.position()
        x, y = pos[0], pos[1]
        x_min, x_max = self.world_bounds["x"]
        y_min, y_max = self.world_bounds["y"]
        nx, ny = self.field_size
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return None
        i = int((x - x_min) / (x_max - x_min) * nx)
        j = int((y - y_min) / (y_max - y_min) * ny)
        i = np.clip(i, 0, nx-1)
        j = np.clip(j, 0, ny-1)
        return self.field_grid[i][j]

    def _get_obs(self):
        s = self.dynamics.state
        max_v = self.config["normalization"].get("max_velocity", 50.0)
        max_d = self.config["normalization"].get("max_distance", 100.0)
        xb, yb, zb = self.world_bounds["x"], self.world_bounds["y"], self.world_bounds["z"]

        def safe_norm(x, min_val, max_val):
            if max_val - min_val == 0:
                return 0.0
            return 2.0 * (x - min_val) / (max_val - min_val) - 1.0

        obs = np.array([
            safe_norm(s.x, xb[0], xb[1]),
            safe_norm(s.y, yb[0], yb[1]),
            safe_norm(s.z, zb[0], zb[1]),
            safe_norm(s.vx, -max_v, max_v),
            safe_norm(s.vy, -max_v, max_v),
            safe_norm(s.vz, -max_v, max_v),
            safe_norm(s.roll, -np.pi, np.pi),
            safe_norm(s.pitch, -np.pi, np.pi),
            safe_norm(s.yaw, -np.pi, np.pi),
            safe_norm(s.roll_rate, -3.0, 3.0),
            safe_norm(s.pitch_rate, -3.0, 3.0),
            safe_norm(s.yaw_rate, -3.0, 3.0),
            safe_norm(np.linalg.norm(s.position() - self.goal_position), 0, max_d),
            safe_norm(0.0, 0, np.pi),
            safe_norm(self.battery_level, 0, 1e6),
            safe_norm(0.0, 0, max_d),
            safe_norm(0.0, 0, 1.0),
        ], dtype=np.float32)
        return obs

    def render(self):
        if self.render_mode != "human":
            return

        # Initialisation PyBullet
        if self._client is None:
            self._client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(cameraDistance=12, cameraYaw=45, cameraPitch=-35,
                                         cameraTargetPosition=[0, 0, 2])
            p.loadURDF("plane.urdf")
            try:
                self._drone_id = p.loadURDF(self._urdf_path, basePosition=[0, 0, 1])
            except:
                self._drone_id = p.loadURDF("sphere_small.urdf", basePosition=[0, 0, 1])
                print("⚠️ URDF drone non trouvé, sphère utilisée.")
            # Récupérer les joints des hélices
            self._prop_joints = []
            for i in range(p.getNumJoints(self._drone_id)):
                info = p.getJointInfo(self._drone_id, i)
                if info[1].decode("utf-8").startswith("j_p"):
                    self._prop_joints.append(i)

        # Créer la grille de champs si pas encore faite
        if not self._field_created and self.field_grid is not None:
            self._create_field_grid()
            self._field_created = True

        # Mettre à jour les couleurs du champ
        if self._field_created and self.field_grid is not None:
            self._update_field_colors()

        # Mettre à jour la position du drone
        state = self.dynamics.state
        pos = [state.x, state.y, state.z]
        ori = p.getQuaternionFromEuler([state.roll, state.pitch, state.yaw])
        p.resetBasePositionAndOrientation(self._drone_id, pos, ori)

        # Hélices
        self._prop_angle += 20.0
        for j in self._prop_joints:
            p.resetJointState(self._drone_id, j, self._prop_angle)

        p.stepSimulation()

    def _create_field_grid(self):
        """Crée les cubes représentant les cellules du champ."""
        nx, ny = self.field_size
        x_min, x_max = self.world_bounds["x"]
        y_min, y_max = self.world_bounds["y"]
        cell_size_x = (x_max - x_min) / nx
        cell_size_y = (y_max - y_min) / ny
        half_x = cell_size_x * 0.45
        half_y = cell_size_y * 0.45

        for i in range(nx):
            row = []
            for j in range(ny):
                x = x_min + (i + 0.5) * cell_size_x
                y = y_min + (j + 0.5) * cell_size_y
                z = 0.025
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[half_x, half_y, 0.025],
                    rgbaColor=[0.2, 0.8, 0.2, 0.8]  # vert par défaut
                )
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual,
                    basePosition=[x, y, z]
                )
                row.append(body_id)
            self._cell_ids.append(row)

    def _update_field_colors(self):
        """Met à jour les couleurs des cellules selon leur état."""
        for i, row in enumerate(self.field_grid):
            for j, cell in enumerate(row):
                if i >= len(self._cell_ids) or j >= len(self._cell_ids[i]):
                    continue
                body_id = self._cell_ids[i][j]
                if not cell.healthy and not cell.wet:
                    color = [0.8, 0.2, 0.1, 0.8]  # rouge foncé
                elif not cell.healthy:
                    color = [0.9, 0.1, 0.1, 0.8]   # rouge
                elif not cell.wet:
                    color = [0.8, 0.6, 0.2, 0.8]   # jaune
                elif cell.sprayed or cell.watered:
                    color = [0.1, 0.5, 0.9, 0.8]   # bleu
                else:
                    color = [0.2, 0.8, 0.2, 0.8]   # vert
                p.changeVisualShape(body_id, -1, rgbaColor=color)

    def close(self):
        if self._client is not None and p.isConnected(self._client):
            p.disconnect(self._client)
            self._client = None