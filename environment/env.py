"""
environment/agri_drone_env.py
=============================
Environnement Gymnasium pour drone agricole.
Version avec vol fonctionnel et affichage de la grille de champs.
"""

import os
import sys
import numpy as np
import pybullet as p
import pybullet_data

from gymnasium import spaces
from rl_template.env import BaseEnv
from .physics.drone_dynamics import DroneDynamics, DroneParams
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
        """Initialise l'environnement agricole pour drone hexacoptère.

        Configure le monde physique, les paramètres du drone, la grille
        du champ agricole, les espaces Gymnasium et la tâche d'irrigation
        (water task) avec groupes de plantes et réservoir d'eau.

        L'espace d'observation est composé de :
            17 dims (état du drone) + 3 dims (bassine) + 1 dim (réservoir)
            + N*4 dims (groupes de plantes) = 17 + 3 + 1 + N*4 dimensions.

        L'espace d'action est de 6 dimensions continues :
            4 commandes de vol + 1 pulvérisation + 1 irrigation.

        Args:
            config: Dictionnaire de configuration contenant les clés
                ``world``, ``drone``, ``simulation``, ``normalization``
                et ``water_task``.
            render_mode: Mode de rendu PyBullet (``"human"``, ``"rgb_array"``)
                ou ``None`` pour désactiver le rendu.

        Raises:
            KeyError: Si les clés obligatoires de ``config`` sont absentes.
        """
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

        # --- Configuration de la tâche d'irrigation (water task) ---
        water_cfg = config.get("water_task", {})
        self.water_basin_position = np.array(
            water_cfg.get("basin_position", [15.0, 15.0, 0.5]), dtype=np.float32
        )
        self.basin_refill_radius = water_cfg.get("basin_refill_radius", 3.0)
        self.water_consumption = water_cfg.get("water_consumption", 2.0)
        self.watering_proximity = water_cfg.get("watering_proximity", 2.0)
        self.num_plant_groups = water_cfg.get("num_plant_groups", 5)

        # --- Espaces Gymnasium ---
        # 17 dims d'origine + 3 (basin) + 1 (tank) + N*4 (plant groups)
        obs_dim = 17 + 3 + 1 + self.num_plant_groups * 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # --- État interne ---
        self.goal_position = np.array([20.0, 20.0, 2.0])
        self.base_position = np.array([0.0, 0.0, 0.05])
        self.battery_level = 1e6
        self.pesticide_level = 5.0
        self.water_level = 5.0
        self.maladies_traitees = 0
        self.sec_arrosees = 0
        self.returning_home = False
        self.step_count = 0

        # --- État de la tâche d'irrigation ---
        self.water_tank_level = 100.0
        self.plant_groups = np.zeros((self.num_plant_groups, 4), dtype=np.float32)

        # --- Calculateur de récompense ---
        self.reward_calc = RewardCalculator(RewardConfig())

        # Rendu PyBullet (lazy)
        self._client = None
        self._drone_id = None
        self._prop_joints = []
        self._prop_angle = 0.0
        self._urdf_path = drone_cfg.get("urdf_path", os.path.join(os.path.dirname(__file__), "agri_hexacopter_pro.urdf"))
        self._field_created = False
        self._headless_initialized = False  # client PyBullet DIRECT pour rendu headless
        self._cell_ids = []  # pour stocker les IDs des cubes

    def reset(self, seed=None):
        """Réinitialise l'environnement pour un nouvel épisode.

        Effectue les opérations suivantes :
            - Remet à zéro le compteur de pas, la batterie, les niveaux
              de pesticide et d'eau, et l'état de retour à la base.
            - Réinitialise la position et la vitesse du drone à l'origine.
            - Génère aléatoirement l'état des cellules du champ (maladies,
              zones sèches) lors du premier appel.
            - Réinitialise les flags de visite, pulvérisation et arrosage
              de toutes les cellules.
            - Positionne aléatoirement les ``num_plant_groups`` groupes
              de plantes dans la carte et remplit le réservoir d'eau à 100.

        Args:
            seed: Graine aléatoire pour la reproductibilité.

        Returns:
            Tuple ``(observation, info)`` où ``observation`` est le vecteur
            d'observation normalisé et ``info`` contient les métadonnées
            de l'épisode (position de l'objectif).
        """
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
        else:
            # Ré-aléatiser les états healthy/wet à chaque reset pour varier l'environnement
            rng = np.random.default_rng()
            for i in range(self.field_size[0]):
                for j in range(self.field_size[1]):
                    self.field_grid[i][j].healthy = rng.random() >= 0.15
                    self.field_grid[i][j].wet = rng.random() >= 0.15

        # Réinitialiser les flags de visite/traitement
        for row in self.field_grid:
            for cell in row:
                cell.visited = False
                cell.sprayed = False
                cell.watered = False

        # --- Réinitialiser la tâche d'irrigation ---
        rng = np.random.default_rng()
        x_min, x_max = self.world_bounds["x"]
        y_min, y_max = self.world_bounds["y"]
        z_min, z_max = self.world_bounds["z"]
        for k in range(self.num_plant_groups):
            self.plant_groups[k] = [
                rng.uniform(x_min * 0.8, x_max * 0.8),
                rng.uniform(y_min * 0.8, y_max * 0.8),
                rng.uniform(z_min + 0.5, z_max * 0.5),
                0.0,  # is_watered = False
            ]
        self.water_tank_level = 100.0

        obs = self._get_obs()
        info = {"goal_position": self.goal_position.tolist()}
        return obs, info

    def step(self, action):
        """Exécute un pas de simulation avec l'action donnée.

        Le pas inclut :
            - La mise à jour de la dynamique physique du drone (4 premières
              dimensions de l'action).
            - La mise à jour visuelle de la cellule sous le drone
              (pulvérisation si ``action[4] > 0``, irrigation si ``action[5] > 0``).
            - La consommation d'eau si un groupe de plantes non arrosé
              est à moins de 2 m et que le réservoir contient assez d'eau.
            - Le remplissage du réservoir si le drone est dans le rayon
              de la bassine.
            - Le calcul de la récompense composite (watering, refill,
              distance shaping, mission complete).
            - La détection de fin d'épisode (mission accomplie ou timeout).

        Args:
            action: Tableau numpy de forme ``(6,)`` et valeurs dans
                ``[-1, 1]`` : ``[throttle, roll, pitch, yaw, spray, irrigate]``.

        Returns:
            Tuple ``(obs, reward, terminated, truncated, info)`` selon
            l'interface Gymnasium. ``info`` contient les métriques
            détaillées (niveau réservoir, groupes restants, etc.).
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        flight_action = action[:4]

        # --- Tracer la distance avant le pas de physique ---
        prev_dist = self._distance_to_nearest_unwatered()

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

        # --- Gestion de l'arrosage des groupes de plantes ---
        just_watered = False
        if action[5] > 0.0 and self.water_tank_level >= self.water_consumption:
            min_idx, min_dist = self._nearest_unwatered_group_index()
            if min_idx >= 0 and min_dist < self.watering_proximity:  # à moins de Xm d'un groupe
                self.plant_groups[min_idx, 3] = 1.0  # marquer comme arrosé
                self.water_tank_level -= self.water_consumption
                just_watered = True

        # --- Gestion du remplissage du réservoir à la bassine ---
        just_refilled = False
        drone_pos = self.dynamics.state.position()
        dist_to_basin = np.linalg.norm(drone_pos - self.water_basin_position)
        if dist_to_basin < self.basin_refill_radius:
            if self.water_tank_level < 98.0:
                just_refilled = True
            self.water_tank_level = 100.0

        self.battery_level -= 0.001
        self.step_count += 1

        # --- Calcul de la distance courante ---
        curr_dist = self._distance_to_nearest_unwatered()

        # --- Vérifier si la mission est accomplie ---
        all_watered = bool(np.all(self.plant_groups[:, 3] >= 0.5))

        # --- Calcul de la récompense ---
        reward, reward_terms = self.reward_calc.compute_water_task(
            tank_level=self.water_tank_level,
            prev_dist=prev_dist,
            curr_dist=curr_dist,
            just_watered=just_watered,
            just_refilled=just_refilled,
            all_watered=all_watered,
            num_unwatered=int(np.sum(self.plant_groups[:, 3] < 0.5)),
        )

        terminated = all_watered
        truncated = (self.step_count >= self.max_steps)

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
            "water_tank_level": self.water_tank_level,
            "num_unwatered": int(np.sum(self.plant_groups[:, 3] < 0.5)),
            "just_watered": just_watered,
            "just_refilled": just_refilled,
            "all_watered": all_watered,
            "reward_terms": reward_terms,
        }
        return obs, reward, terminated, truncated, info

    def _get_cell_under_drone(self):
        """Retourne la cellule du champ située sous la position projetée du drone.

        Projette la position actuelle du drone sur la grille 2D du champ
        et retourne l'objet ``FieldCell`` correspondant. Si le drone est
        en dehors des limites du champ, retourne ``None``.

        Returns:
            L'instance ``FieldCell`` sous le drone, ou ``None`` si hors limites.
        """
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

    def _distance_to_nearest_unwatered(self):
        """Calcule la distance du drone au groupe de plantes non arrosé le plus proche.

        Parcourt tous les groupes de plantes et retourne la distance
        euclidienne minimale vers ceux dont le statut d'arrosage est
        inférieur à 0.5.

        Returns:
            Distance minimale en mètres (float). Retourne 0.0 si tous
            les groupes sont déjà arrosés.
        """
        drone_pos = self.dynamics.state.position()
        min_dist = float("inf")
        for k in range(self.num_plant_groups):
            if self.plant_groups[k, 3] < 0.5:  # non arrosé
                group_pos = self.plant_groups[k, :3]
                dist = np.linalg.norm(drone_pos - group_pos)
                if dist < min_dist:
                    min_dist = dist
        return min_dist if min_dist < float("inf") else 0.0

    def _nearest_unwatered_group_index(self):
        """Trouve l'indice et la distance du groupe non arrosé le plus proche.

        Parcourt tous les groupes de plantes et identifie celui dont le
        statut d'arrosage est inférieur à 0.5 et qui est le plus proche
        du drone en distance euclidienne.

        Returns:
            Tuple ``(index, distance)`` où ``index`` est l'indice du groupe
            le plus proche (entier >= 0), ou -1 si tous les groupes sont
            déjà arrosés. ``distance`` est la distance en mètres.
        """
        drone_pos = self.dynamics.state.position()
        min_dist = float("inf")
        min_idx = -1
        for k in range(self.num_plant_groups):
            if self.plant_groups[k, 3] < 0.5:  # non arrosé
                group_pos = self.plant_groups[k, :3]
                dist = np.linalg.norm(drone_pos - group_pos)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = k
        return min_idx, min_dist

    def _get_obs(self):
        """Construit le vecteur d'observation normalisé.

        La disposition complète du vecteur est la suivante :

        - **Dimensions 0–16** : État du drone (17 dims)
            0–2   : Position (x, y, z)
            3–5   : Vitesse linéaire (vx, vy, vz)
            6–8   : Attitude (roll, pitch, yaw)
            9–11  : Vitesses angulaires (roll_rate, pitch_rate, yaw_rate)
            12    : Distance à l'objectif
            13    : Erreur de cap (heading error)
            14    : Niveau de batterie
            15    : Distance à l'obstacle le plus proche (toujours 0 ici)
            16    : Intensité du vent (toujours 0 ici)

        - **Dimensions 17–19** : Coordonnées de la bassine d'eau (x, y, z)
        - **Dimension 20** : Niveau du réservoir d'eau (0–100)
        - **Dimensions 21 à 20+N*4** : Matrice des groupes de plantes
            aplatie, chaque groupe contenant (x, y, z, is_watered).

        Toutes les valeurs continues sont normalisées dans ``[-1, 1]``.
        Le booléen ``is_watered`` est transformé de 0.0/1.0 vers -1.0/+1.0.

        Returns:
            Tableau numpy de forme ``(obs_dim,)`` et dtype ``float32``.
        """
        s = self.dynamics.state
        max_v = self.config["normalization"].get("max_velocity", 50.0)
        max_d = self.config["normalization"].get("max_distance", 100.0)
        xb, yb, zb = self.world_bounds["x"], self.world_bounds["y"], self.world_bounds["z"]

        def safe_norm(x, min_val, max_val):
            """Normalise x dans [-1, 1] en évitant la division par zéro."""
            if max_val - min_val == 0:
                return 0.0
            return 2.0 * (x - min_val) / (max_val - min_val) - 1.0

        obs_list = [
            # --- 17 dims d'origine (état du drone) ---
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
        ]

        # --- Coordonnées de la bassine d'eau (3 dims) ---
        obs_list.append(safe_norm(self.water_basin_position[0], xb[0], xb[1]))
        obs_list.append(safe_norm(self.water_basin_position[1], yb[0], yb[1]))
        obs_list.append(safe_norm(self.water_basin_position[2], zb[0], zb[1]))

        # --- Niveau du réservoir d'eau (1 dim) ---
        obs_list.append(safe_norm(self.water_tank_level, 0.0, 100.0))

        # --- Matrice des groupes de plantes aplatie (N*4 dims) ---
        for k in range(self.num_plant_groups):
            obs_list.append(safe_norm(self.plant_groups[k, 0], xb[0], xb[1]))
            obs_list.append(safe_norm(self.plant_groups[k, 1], yb[0], yb[1]))
            obs_list.append(safe_norm(self.plant_groups[k, 2], zb[0], zb[1]))
            obs_list.append(2.0 * self.plant_groups[k, 3] - 1.0)  # normaliser 0/1 vers -1/+1

        obs = np.array(obs_list, dtype=np.float32)
        return obs

    def render(self):
        """Rendu de l'environnement selon le mode configuré.

        Modes supportés :
          - ``"human"`` : affiche la fenêtre PyBullet GUI en temps réel.
          - ``"rgb_array"`` : capture une image RGB sans fenêtre (headless)
            et la retourne comme tableau numpy de forme ``(H, W, 3)``.
          - ``None`` : ne fait rien.

        Returns:
            Tableau numpy ``(H, W, 3)`` dtype ``uint8`` si
            ``render_mode == "rgb_array"``, sinon ``None``.
        """
        if self.render_mode is None:
            return None

        if self.render_mode == "rgb_array":
            return self._render_rgb_array()

        if self.render_mode == "human":
            return self._render_human()

        return None

    def _init_headless(self):
        """Initialise PyBullet en mode DIRECT (sans GUI) pour le rendu headless.

        Crée une connexion PyBullet sans fenêtre, charge le modèle URDF
        du drone (ou une sphère de secours), le sol, et configure la
        caméra pour la capture d'images.
        """
        if self._headless_initialized:
            return

        self._client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.resetDebugVisualizerCamera(
            cameraDistance=15, cameraYaw=45, cameraPitch=-30,
            cameraTargetPosition=[0, 0, 2], physicsClientId=self._client,
        )

        # Sol
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        # Drone
        try:
            self._drone_id = p.loadURDF(
                self._urdf_path, basePosition=[0, 0, 1],
                physicsClientId=self._client,
            )
        except Exception:
            self._drone_id = p.loadURDF(
                "sphere_small.urdf", basePosition=[0, 0, 1],
                physicsClientId=self._client,
            )

        # Récupérer les joints des hélices
        self._prop_joints = []
        for i in range(p.getNumJoints(self._drone_id, physicsClientId=self._client)):
            info = p.getJointInfo(self._drone_id, i, physicsClientId=self._client)
            if info[1].decode("utf-8").startswith("j_p"):
                self._prop_joints.append(i)

        self._headless_initialized = True

    def _render_rgb_array(self):
        """Capture une image RGB de la scène en mode headless.

        Initialise PyBullet en mode DIRECT si nécessaire, met à jour
        la position du drone et les couleurs du champ, puis capture
        une image via ``getCameraImage()``.

        Returns:
            Tableau numpy ``(H, W, 3)`` dtype ``uint8`` représentant
            l'image RGB de la scène.
        """
        self._init_headless()

        # Créer la grille si nécessaire
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
        p.resetBasePositionAndOrientation(
            self._drone_id, pos, ori, physicsClientId=self._client,
        )

        # Hélices
        self._prop_angle += 20.0
        for j in self._prop_joints:
            p.resetJointState(
                self._drone_id, j, self._prop_angle,
                physicsClientId=self._client,
            )

        p.stepSimulation(physicsClientId=self._client)

        # Capturer l'image
        width, height = 640, 480
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[12, 12, 8],
            cameraTargetPosition=[0, 0, 2],
            cameraUpVector=[0, 0, 1],
            physicsClientId=self._client,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width / height, nearVal=0.1, farVal=100.0,
            physicsClientId=self._client,
        )
        _, _, rgb_img, _, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            physicsClientId=self._client,
        )

        # Convertir en tableau numpy (H, W, 3)
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        return rgb_array

    def _render_human(self):
        """Rendu en mode GUI PyBullet (fenêtre temps réel).

        Identique à l'ancienne implémentation : initialise PyBullet
        en mode GUI, charge le drone, anime les hélices et affiche
        la grille du champ.
        """
        # Initialisation PyBullet
        if self._client is None:
            self._client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=12, cameraYaw=45, cameraPitch=-35,
                cameraTargetPosition=[0, 0, 2],
            )
            p.loadURDF("plane.urdf")
            try:
                self._drone_id = p.loadURDF(self._urdf_path, basePosition=[0, 0, 1])
            except Exception:
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
        return None

    def _create_field_grid(self):
        """Crée les cubes PyBullet représentant visuellement les cellules du champ.

        Pour chaque cellule de la grille ``field_grid``, un cube de forme
        visuelle est créé à la position correspondante dans le simulateur
        PyBullet. Les IDs des cubes sont stockés dans ``_cell_ids`` pour
        une mise à jour ultérieure des couleurs.
        """
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
        """Met à jour les couleurs des cubes du champ selon l'état des cellules.

        La couleur de chaque cellule reflète son état :
            - Rouge foncé : malade et sec
            - Rouge : malade
            - Jaune : sec
            - Bleu : pulvérisée ou arrosée
            - Vert : saine et humide (par défaut)
        """
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
        """Ferme les connexions PyBullet et libère les ressources.

        Déconnecte les clients GUI et DIRECT s'ils sont actifs.
        """
        if self._client is not None and p.isConnected(self._client):
            p.disconnect(self._client)
            self._client = None
        self._headless_initialized = False
