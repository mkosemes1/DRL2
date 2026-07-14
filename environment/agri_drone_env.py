"""
environment/agri_drone_env.py
================================
Environnement Gymnasium principal du drone agricole.

Respecte l'API Gymnasium standard :
  - reset(seed, options) -> (obs, info)
  - step(action) -> (obs, reward, terminated, truncated, info)
  - render()
  - close()

Architecture pensée pour l'extensibilité future (caméra RGB,
multispectrale, pulvérisation...) : toutes ces fonctionnalités
pourront être ajoutées en étendant observation_space et en
branchant de nouveaux capteurs dans _get_observation(), sans
toucher au coeur de la dynamique/reward.

Rendu 3D :
  Le rendu PyBullet (environment/pybullet_renderer.py) est
  totalement découplé de la dynamique d'entraînement. Il n'est
  initialisé que si render_mode="human", et ne fait qu'afficher
  l'état calculé par physics/drone_dynamics.py — il ne recalcule
  aucune physique. Cela permet un entraînement rapide (dynamique
  custom légère) tout en offrant une visualisation 3D fidèle du
  vrai modèle URDF lors de l'évaluation.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from physics.drone_dynamics import DroneDynamics, DroneParams
from physics.wind_model import WindModel
from obstacles import ObstacleManager
from reward.reward_function import RewardCalculator, RewardConfig
from utils.normalization import normalize


class AgriDroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(self, config: dict, render_mode: str | None = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        world_cfg = config["world"]
        self.world_bounds = {
            "x": (-world_cfg["size_x"] / 2, world_cfg["size_x"] / 2),
            "y": (-world_cfg["size_y"] / 2, world_cfg["size_y"] / 2),
            "z": (world_cfg["ground_z"], world_cfg["size_z"]),
        }

        drone_cfg = config["drone"]
        self.drone_params = DroneParams(
            dry_mass=drone_cfg["dry_mass"],
            payload_mass=drone_cfg["payload_mass_full"],
            gravity=drone_cfg["gravity"],
            max_thrust_total=drone_cfg["max_thrust_total"],
            drag_coefficient=drone_cfg["drag_coefficient"],
            max_tilt_angle_rad=drone_cfg["max_tilt_angle_rad"],
            max_angular_rate=drone_cfg["max_angular_rate"],
            attitude_time_constant=drone_cfg["attitude_time_constant"],
            max_velocity=config["normalization"]["max_velocity"],
        )

        self.dt = config["simulation"]["dt"]
        self.max_steps = config["simulation"]["max_episode_steps"]
        self.max_velocity = config["normalization"]["max_velocity"]
        self.max_distance = config["normalization"]["max_distance"]
        self.battery_capacity = config["battery"]["capacity_wh"]
        self.battery_alpha = config["battery"]["consumption_coefficient"]
        self.success_radius = config["goal"]["success_radius"]

        self.dynamics = DroneDynamics(self.drone_params, self.world_bounds, self.dt)
        self.wind = WindModel(
            max_speed=config["wind"]["max_speed"],
            gust_probability=config["wind"]["gust_probability"],
            enabled=config["wind"]["enabled"],
        )
        self.obstacle_manager = ObstacleManager(
            self.world_bounds,
            min_radius=config["obstacles"]["min_radius"],
            max_radius=config["obstacles"]["max_radius"],
        )
        self.reward_calculator = RewardCalculator(RewardConfig())

        # --- Espaces Gymnasium ---
        # Observation (17 dimensions, toutes normalisées dans [-1, 1]) :
        # [x,y,z, vx,vy,vz, roll,pitch,yaw, roll_r,pitch_r,yaw_r,
        #  distance_goal, direction_goal, battery, nearest_obstacle, wind_speed]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)

        # Action : [throttle, roll_cmd, pitch_cmd, yaw_cmd] dans [-1, 1]
        # PPO préfère un espace d'action continu car :
        #   - la politique est une gaussienne paramétrée (mean, std),
        #     naturellement adaptée à des sorties continues ;
        #   - un multirotor a un contrôle intrinsèquement continu
        #     (pas d'ensemble discret d'actions physiquement pertinent) ;
        #   - discrétiser l'espace ferait exploser le nombre d'actions
        #     possibles (4 dimensions continues -> combinatoire énorme)
        #     et perdrait la notion de proximité entre actions voisines.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # État interne
        self.current_step = 0
        self.goal_position = np.zeros(3)
        self.battery_level = self.battery_capacity
        self.curriculum_stage: dict = {
            "obstacle_count": config["obstacles"]["count_stage_default"],
            "wind_enabled": config["wind"]["enabled"],
            "randomize_mass": False,
        }

        # Dernière action de throttle normalisée [0,1], utilisée uniquement
        # pour animer visuellement la vitesse de rotation des hélices
        # dans le rendu PyBullet (aucun impact sur la physique/apprentissage).
        self._last_throttle_normalized = 0.0

        # Renderer PyBullet créé de façon paresseuse (lazy), uniquement
        # si render_mode == "human" et lors du premier appel à render().
        self._pybullet_renderer = None

        self._np_random = np.random.default_rng()

    # ------------------------------------------------------------------
    def set_curriculum_stage(self, stage: dict) -> None:
        """Appelé par le CurriculumManager pour changer la difficulté."""
        self.curriculum_stage = stage
        self.wind.enabled = stage.get("wind_enabled", False)

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.current_step = 0
        self.battery_level = self.battery_capacity
        self._last_throttle_normalized = 0.0

        # Domain randomization de la masse (cuve pleine/vide aléatoire)
        if self.curriculum_stage.get("randomize_mass", False):
            fill_ratio = self._np_random.uniform(0.0, 1.0)
            self.dynamics.params.payload_mass = fill_ratio * self.config["drone"]["payload_mass_full"]
        else:
            self.dynamics.params.payload_mass = self.config["drone"]["payload_mass_full"]

        # Position de départ aléatoire (domain randomization)
        start_pos = np.array([
            self._np_random.uniform(-5, 5),
            self._np_random.uniform(-5, 5),
            1.0,
        ])
        self.dynamics.reset(start_pos)

        # Objectif GPS aléatoire, suffisamment loin du départ
        xb, yb = self.world_bounds["x"], self.world_bounds["y"]
        while True:
            self.goal_position = np.array([
                self._np_random.uniform(xb[0] * 0.8, xb[1] * 0.8),
                self._np_random.uniform(yb[0] * 0.8, yb[1] * 0.8),
                self._np_random.uniform(2.0, 10.0),
            ])
            if np.linalg.norm(self.goal_position[:2] - start_pos[:2]) > 10.0:
                break

        # Obstacles selon le stage courant
        n_obstacles = self.curriculum_stage.get("obstacle_count", 0)
        self.obstacle_manager.generate(
            n_obstacles,
            exclude_zone=[(start_pos, 2.0), (self.goal_position, 2.0)],
        )

        self.wind.reset()
        self.reward_calculator.reset()

        # Si un renderer 3D est déjà actif, on resynchronise la scène
        # (nouveaux marqueurs d'objectif et d'obstacles) sans le recréer.
        if self._pybullet_renderer is not None:
            self._pybullet_renderer.reset_scene(
                self.goal_position, self.obstacle_manager.obstacles
            )

        obs = self._get_observation()
        info = {"goal_position": self.goal_position.tolist()}
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        distance_old = self._distance_to_goal()

        # Injection du vent comme perturbation de vitesse
        wind_vec = self.wind.step()
        state = self.dynamics.step(action)
        state.vx += wind_vec[0] * self.dt
        state.vy += wind_vec[1] * self.dt

        distance_new = self._distance_to_goal()
        heading_error = self._heading_error()

        # Consommation batterie (Wh) proportionnelle à throttle²
        throttle_normalized = (action[0] + 1.0) / 2.0
        self._last_throttle_normalized = float(throttle_normalized)  # pour le rendu 3D

        power_watts = self.battery_alpha * (throttle_normalized ** 2) * 10.0  # facteur d'échelle
        self.battery_level -= power_watts * (self.dt / 3600.0)
        self.battery_level = max(0.0, self.battery_level)

        collided = self.obstacle_manager.check_collision(state.position())
        out_of_bounds = self.dynamics.is_out_of_bounds()
        flipped = self.dynamics.is_flipped()
        reached_goal = distance_new < self.success_radius

        reward, reward_terms = self.reward_calculator.compute(
            distance_old=distance_old,
            distance_new=distance_new,
            heading_error=heading_error,
            action=action,
            angular_rates=np.array([state.roll_rate, state.pitch_rate, state.yaw_rate]),
            collided=collided,
            out_of_bounds=out_of_bounds,
            flipped=flipped,
            reached_goal=reached_goal,
        )

        self.current_step += 1
        terminated = bool(collided or out_of_bounds or flipped or reached_goal
                           or self.battery_level <= 0.0)
        truncated = bool(self.current_step >= self.max_steps)

        obs = self._get_observation()
        info = {
            "reward_terms": reward_terms,
            "distance_to_goal": distance_new,
            "collided": collided,
            "out_of_bounds": out_of_bounds,
            "flipped": flipped,
            "reached_goal": reached_goal,
            "battery_level": self.battery_level,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.dynamics.state.position() - self.goal_position))

    def _heading_error(self) -> float:
        """Angle absolu (0 à pi) entre le cap (yaw) du drone et la direction vers la cible."""
        s = self.dynamics.state
        direction = self.goal_position[:2] - np.array([s.x, s.y])
        target_yaw = np.arctan2(direction[1], direction[0])
        error = target_yaw - s.yaw
        error = (error + np.pi) % (2 * np.pi) - np.pi
        return abs(error)

    def _get_observation(self) -> np.ndarray:
        s = self.dynamics.state
        max_v = self.max_velocity
        max_d = self.max_distance
        xb, yb, zb = self.world_bounds["x"], self.world_bounds["y"], self.world_bounds["z"]

        distance_goal = self._distance_to_goal()
        heading_err = self._heading_error()
        nearest_obs = self.obstacle_manager.nearest_distance(s.position())
        nearest_obs_clamped = min(nearest_obs, max_d)

        obs = np.array([
            normalize(s.x, xb[0], xb[1]),
            normalize(s.y, yb[0], yb[1]),
            normalize(s.z, zb[0], zb[1]),
            normalize(s.vx, -max_v, max_v),
            normalize(s.vy, -max_v, max_v),
            normalize(s.vz, -max_v, max_v),
            normalize(s.roll, -np.pi, np.pi),
            normalize(s.pitch, -np.pi, np.pi),
            normalize(s.yaw, -np.pi, np.pi),
            normalize(s.roll_rate, -self.drone_params.max_angular_rate, self.drone_params.max_angular_rate),
            normalize(s.pitch_rate, -self.drone_params.max_angular_rate, self.drone_params.max_angular_rate),
            normalize(s.yaw_rate, -self.drone_params.max_angular_rate, self.drone_params.max_angular_rate),
            normalize(distance_goal, 0, max_d),
            normalize(heading_err, 0, np.pi),
            normalize(self.battery_level, 0, self.battery_capacity),
            normalize(nearest_obs_clamped, 0, max_d),
            normalize(self.wind.magnitude(), 0, self.config["wind"]["max_speed"]),
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    def render(self):
        """
        Affichage 3D temps réel via PyBullet (render_mode="human" uniquement).

        Le renderer est initialisé au premier appel (lazy init) pour ne pas
        pénaliser l'entraînement lorsque render() n'est jamais appelé.
        Il ne fait qu'afficher l'état déjà calculé par self.dynamics —
        aucune physique n'est recalculée par PyBullet ici.
        """
        if self.render_mode != "human":
            return

        if self._pybullet_renderer is None:
            from environment.pybullet_renderer import PyBulletRenderer
            self._pybullet_renderer = PyBulletRenderer(
                urdf_path=self.config["drone"]["urdf_path"]
            )
            self._pybullet_renderer.reset_scene(
                self.goal_position, self.obstacle_manager.obstacles
            )

        self._pybullet_renderer.update(self.dynamics.state, self._last_throttle_normalized)

    def close(self):
        """Ferme proprement la connexion PyBullet si elle a été ouverte."""
        if self._pybullet_renderer is not None:
            self._pybullet_renderer.close()
            self._pybullet_renderer = None
