"""
environment/agri_drone_env.py
Environnement Gymnasium pour drone agricole.
Version avec vol fonctionnel et affichage de la grille de champs.
"""

import os
import numpy as np
import pybullet as p
import pybullet_data

from gymnasium import spaces
from rl_template.env import BaseEnv
from .physics.drone_dynamics import DroneWrapper
from .physics.wind_model import WindModel
from .obstacles import ObstacleManager
from .reward.reward_function import RewardCalculator, RewardConfig


class FieldCell:
    """Représente une cellule individuelle de la grille du champ agricole.

    Chaque cellule possède des attributs booléens décrivant son état
    sanitaire et son historique de traitement par le drone.

    Attributes:
        healthy: True si la plante est saine, False si elle est malade.
        wet: True si le sol est humide, False s'il est sec.
        sprayed: True si la cellule a déjà été pulvérisée.
        watered: True si la cellule a déjà été arrosée.
        visited: True si le drone a survolé cette cellule au moins une fois.
    """

    def __init__(self):
        self.healthy = True
        self.wet = True
        self.sprayed = False
        self.watered = False
        self.visited = False


class AgriDroneEnv(BaseEnv):
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(self, config: dict, render_mode: str | None = None):
        super().__init__()
        self.config = config
        this_render_mode = render_mode
        self.render_mode = this_render_mode
        
        # --- Connexion PyBullet ---
        if self.render_mode == "human":
            self._client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self._client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        drone_cfg = config["drone"]
        urdf_path = drone_cfg.get("urdf_path", os.path.join(os.path.dirname(__file__), "agri_hexacopter_pro.urdf"))

        # --- Paramètres du monde ---
        world_cfg = config["world"]
        self.world_bounds = {
            "x": (-world_cfg["size_x"]/2, world_cfg["size_x"]/2),
            "y": (-world_cfg["size_y"]/2, world_cfg["size_y"]/2),
            "z": (world_cfg["ground_z"], world_cfg["size_z"]),
        }
        self.field_size = (world_cfg.get("field_cells_x", 20), world_cfg.get("field_cells_y", 20))
        self.field_grid = None
        self._field_created = False
        self._cell_ids = []

        self.dt = config["simulation"].get("dt", 0.02)
        self.max_steps = config["simulation"].get("max_episode_steps", 1000)
        
        self.dynamics = DroneWrapper(urdf_path, client_id=self._client)
        self._drone_id = self.dynamics.drone_id

        # Modèle de vent (activable)
        self.wind = WindModel(enabled=True)
        self.obstacle_manager = ObstacleManager(self.world_bounds, min_radius=1.0, max_radius=3.0)
        self.obstacle_manager.obstacles = []

        # Tâche d'irrigation & Bassine
        water_cfg = config.get("water_task", {})
        self.water_basin_position = np.array(
            water_cfg.get("basin_position", [15.0, 15.0, 0.5]), dtype=np.float32
        )
        self.basin_refill_radius = water_cfg.get("basin_refill_radius", 3.0)
        self.water_consumption = water_cfg.get("water_consumption", 2.0)
        self.watering_proximity = water_cfg.get("watering_proximity", 2.0)
        self.num_plant_groups = water_cfg.get("num_plant_groups", 5)

        # --- Espaces Gymnasium ---
        # 18 dims (drone) + 3 (basin pos) + 1 (tank level) + 2 (wind vec) + N*4 (plant groups)
        obs_dim = 19 + 4 + self.num_plant_groups * 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # États internes
        self.goal_position = np.array([20.0, 20.0, 2.0])
        self.battery_level = 1e6
        self.water_tank_level = 100.0
        self.plant_groups = np.zeros((self.num_plant_groups, 4), dtype=np.float32)
        self.step_count = 0

        self.reward_calc = RewardCalculator(RewardConfig())
        self._prop_joints = []
        self._prop_angle = 0.0

        # Création de la bassine visuelle dans le simulateur
        self._create_water_basin_visual()

    def _create_water_basin_visual(self):
        """Crée un cylindre bleu translucide matérialisant la bassine d'eau."""
        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.basin_refill_radius,
            length=0.05,
            rgbaColor=[0.1, 0.4, 0.9, 0.7],
            physicsClientId=self._client
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=[self.water_basin_position[0], self.water_basin_position[1], 0.02],
            physicsClientId=self._client
        )

    def reset(self, seed=None, options=None):
        # On passe le seed et les options au parent si géré par BaseEnv / Gymnasium
        super().reset(seed=seed)
        
        self.step_count = 0
        self.battery_level = 1e6
        self.water_tank_level = 100.0
        self.wind.reset()

        start_pos = [0.0, 0.0, 1.0]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.dynamics.reset(base_position=start_pos)

        # Reste de ta logique de reset...
        if self.field_grid is None:
            self.field_grid = [[FieldCell() for _ in range(self.field_size[1])] for _ in range(self.field_size[0])]
        
        rng = np.random.default_rng()
        for row in self.field_grid:
            for cell in row:
                cell.healthy = rng.random() >= 0.15
                cell.wet = rng.random() >= 0.15
                cell.visited = False
                cell.sprayed = False
                cell.watered = False

        x_min, x_max = self.world_bounds["x"]
        y_min, y_max = self.world_bounds["y"]
        z_min, z_max = self.world_bounds["z"]
        for k in range(self.num_plant_groups):
            self.plant_groups[k] = [
                rng.uniform(x_min * 0.8, x_max * 0.8),
                rng.uniform(y_min * 0.8, y_max * 0.8),
                rng.uniform(z_min + 0.5, z_max * 0.5),
                0.0,
            ]

        obs = self._get_obs()
        info = {"goal_position": self.goal_position.tolist()}
        
        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        flight_action = action[:4]

        prev_dist = self._distance_to_nearest_unwatered()

        # 1. Action physique du drone
        self.dynamics.apply_action(flight_action)

        # 2. Application du vent physique (désactivé par défaut pour l'apprentissage)
        current_wind = self.wind.step()
        if self.wind.enabled:
            wind_force = [current_wind[0], current_wind[1], 0.0]
            p.applyExternalForce(
                self._drone_id, -1, forceObj=wind_force, posObj=[0, 0, 0], 
                flags=p.WORLD_FRAME, physicsClientId=self._client
            )

        p.stepSimulation(physicsClientId=self._client)

        # Récupération de l'état physique actuel
        pos, lin_vel, ang_vel, (roll, pitch, yaw) = self.dynamics.get_raw_state()
        drone_pos = np.array(pos, dtype=np.float32)

        # --- ANTI-TRICHE : Détection du retournement (drone sur le dos) ---
        is_flipped = abs(roll) > 1.4 or abs(pitch) > 1.4

        # Maintien et états du sol
        cell = self._get_cell_under_drone()
        if cell is not None:
            cell.visited = True
            if action[4] > 0.0 and not cell.healthy and not cell.sprayed:
                cell.healthy = True
                cell.sprayed = True
            if action[5] > 0.0 and not cell.wet and not cell.watered:
                cell.wet = True
                cell.watered = True

        # Arrosage des groupes de plantes
        just_watered = False
        if action[5] > 0.0 and self.water_tank_level >= self.water_consumption:
            min_idx, min_dist = self._nearest_unwatered_group_index()
            if min_idx >= 0 and min_dist < self.watering_proximity:
                self.plant_groups[min_idx, 3] = 1.0
                self.water_tank_level -= self.water_consumption
                just_watered = True

        # Remplissage à la bassine
        just_refilled = False
        dist_to_basin = np.linalg.norm(drone_pos - self.water_basin_position)
        if dist_to_basin < self.basin_refill_radius:
            if self.water_tank_level < 98.0:
                just_refilled = True
            self.water_tank_level = 100.0

        self.battery_level -= 0.001
        self.step_count += 1

        curr_dist = self._distance_to_nearest_unwatered()
        all_watered = bool(np.all(self.plant_groups[:, 3] >= 0.5))
        
        # Le drone crash s'il touche le sol OU s'il est retourné sur le dos
        crashed = bool(drone_pos[2] < 0.15) or is_flipped

        # Calcul de la récompense
        reward, reward_terms = self.reward_calc.compute_water_task(
            tank_level=self.water_tank_level,
            prev_dist=prev_dist,
            curr_dist=curr_dist,
            drone_pos=drone_pos,
            roll=roll,
            pitch=pitch,
            lin_vel=lin_vel,
            just_watered=just_watered,
            just_refilled=just_refilled,
            all_watered=all_watered,
            num_unwatered=int(np.sum(self.plant_groups[:, 3] < 0.5)),
            crashed=crashed,
        )

        terminated = all_watered or crashed
        truncated = (self.step_count >= self.max_steps)

        obs = self._get_obs()
        info = {
            "distance_to_goal": np.linalg.norm(drone_pos - self.goal_position),
            "water_tank_level": self.water_tank_level,
            "just_watered": just_watered,
            "just_refilled": just_refilled,
            "all_watered": all_watered,
            "crashed": crashed,
            "reward_terms": reward_terms,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_drone_pos(self):
        pos, _, _, _ = self.dynamics.get_raw_state()
        return pos

    def _get_cell_under_drone(self):
        pos = self._get_drone_pos()
        x, y = pos[0], pos[1]
        x_min, x_max = self.world_bounds["x"]
        y_min, y_max = self.world_bounds["y"]
        nx, ny = self.field_size
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return None
        i = int((x - x_min) / (x_max - x_min) * nx)
        j = int((y - y_min) / (y_max - y_min) * ny)
        return self.field_grid[np.clip(i, 0, nx-1)][np.clip(j, 0, ny-1)]

    def _distance_to_nearest_unwatered(self):
        drone_pos = self._get_drone_pos()
        min_dist = float("inf")
        for k in range(self.num_plant_groups):
            if self.plant_groups[k, 3] < 0.5:
                dist = np.linalg.norm(drone_pos - self.plant_groups[k, :3])
                if dist < min_dist:
                    min_dist = dist
        return min_dist if min_dist < float("inf") else 0.0

    def _nearest_unwatered_group_index(self):
        drone_pos = self._get_drone_pos()
        min_dist = float("inf")
        min_idx = -1
        for k in range(self.num_plant_groups):
            if self.plant_groups[k, 3] < 0.5:
                dist = np.linalg.norm(drone_pos - self.plant_groups[k, :3])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = k
        return min_idx, min_dist

    def _get_obs(self):
        pos, lin_vel, ang_vel, (roll, pitch, yaw) = self.dynamics.get_raw_state()
        drone_position = np.array(pos, dtype=np.float32)

        max_v = self.config["normalization"].get("max_velocity", 50.0)
        max_d = self.config["normalization"].get("max_distance", 100.0)
        xb, yb, zb = self.world_bounds["x"], self.world_bounds["y"], self.world_bounds["z"]

        def safe_norm(x, min_val, max_val):
            if max_val - min_val == 0:
                return 0.0
            return 2.0 * (x - min_val) / (max_val - min_val) - 1.0

        obs_list = [
            safe_norm(pos[0], xb[0], xb[1]),
            safe_norm(pos[1], yb[0], yb[1]),
            safe_norm(pos[2], zb[0], zb[1]),
            safe_norm(lin_vel[0], -max_v, max_v),
            safe_norm(lin_vel[1], -max_v, max_v),
            safe_norm(lin_vel[2], -max_v, max_v),
            safe_norm(roll, -np.pi, np.pi),
            safe_norm(pitch, -np.pi, np.pi),
            np.cos(yaw),
            np.sin(yaw),
            safe_norm(ang_vel[0], -3.0, 3.0),
            safe_norm(ang_vel[1], -3.0, 3.0),
            safe_norm(ang_vel[2], -3.0, 3.0),
            safe_norm(np.linalg.norm(drone_position - self.goal_position), 0, max_d),
            safe_norm(0.0, 0, np.pi),
            safe_norm(self.battery_level, 0, 1e6),
            safe_norm(0.0, 0, max_d),
            # Vent directionnel normalisé (wx, wy)
            safe_norm(self.wind.current_wind[0], -self.wind.max_speed, self.wind.max_speed),
            safe_norm(self.wind.current_wind[1], -self.wind.max_speed, self.wind.max_speed),
        ]

        # Bassine et réservoir
        obs_list.append(safe_norm(self.water_basin_position[0], xb[0], xb[1]))
        obs_list.append(safe_norm(self.water_basin_position[1], yb[0], yb[1]))
        obs_list.append(safe_norm(self.water_basin_position[2], zb[0], zb[1]))
        obs_list.append(safe_norm(self.water_tank_level, 0.0, 100.0))

        # Groupes de plantes
        for k in range(self.num_plant_groups):
            obs_list.append(safe_norm(self.plant_groups[k, 0], xb[0], xb[1]))
            obs_list.append(safe_norm(self.plant_groups[k, 1], yb[0], yb[1]))
            obs_list.append(safe_norm(self.plant_groups[k, 2], zb[0], zb[1]))
            obs_list.append(2.0 * self.plant_groups[k, 3] - 1.0)

        return np.array(obs_list, dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.render_mode == "human":
            return self._render_human()
        return None

    def _render_human(self):
        if not self._field_created and self.field_grid is not None:
            self._create_field_grid()
            self._field_created = True

        if self._field_created and self.field_grid is not None:
            self._update_field_colors()

        p.stepSimulation(physicsClientId=self._client)
        return None

    def _create_field_grid(self):
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
                    rgbaColor=[0.2, 0.8, 0.2, 0.8],
                    physicsClientId=self._client
                )
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual,
                    basePosition=[x, y, z],
                    physicsClientId=self._client
                )
                row.append(body_id)
            self._cell_ids.append(row)

    def _update_field_colors(self):
        if not hasattr(self, "_last_colors"):
            self._last_colors = {}

        for i, row in enumerate(self.field_grid):
            for j, cell in enumerate(row):
                if i >= len(self._cell_ids) or j >= len(self._cell_ids[i]):
                    continue
                body_id = self._cell_ids[i][j]
                
                if not cell.healthy and not cell.wet:
                    color = (0.8, 0.2, 0.1, 0.8)
                elif not cell.healthy:
                    color = (0.9, 0.1, 0.1, 0.8)
                elif not cell.wet:
                    color = (0.8, 0.6, 0.2, 0.8)
                elif cell.sprayed or cell.watered:
                    color = (0.1, 0.5, 0.9, 0.8)
                else:
                    color = (0.2, 0.8, 0.2, 0.8)

                if self._last_colors.get(body_id) != color:
                    p.changeVisualShape(body_id, -1, rgbaColor=list(color), physicsClientId=self._client)
                    self._last_colors[body_id] = color

    def close(self):
        if self._client is not None and p.isConnected(self._client):
            p.disconnect(self._client)
            self._client = None
