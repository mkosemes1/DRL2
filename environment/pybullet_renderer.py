"""
environment/pybullet_renderer.py
===================================
Rendu 3D temps réel via PyBullet, découplé de la physique
d'entraînement. Ce module charge l'URDF réel du drone et
synchronise sa pose visuelle avec l'état calculé par
physics/drone_dynamics.py (qui reste le seul responsable de la
dynamique, y compris pendant l'évaluation).

Utilisation : uniquement en mode render_mode="human", jamais
pendant l'entraînement massif (trop lent en boucle serrée).

VERSION ÉTENDUE : affiche également une grille de cellules
agricoles avec des couleurs représentant l'état du champ.
"""

from __future__ import annotations
import numpy as np
import pybullet as p
import pybullet_data


class PyBulletRenderer:
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=15, cameraYaw=45, cameraPitch=-30,
            cameraTargetPosition=[0, 0, 2]
        )

        self.plane_id = p.loadURDF("plane.urdf")
        self.drone_id = p.loadURDF(self.urdf_path, basePosition=[0, 0, 1])

        # Récupère les indices de joints des hélices pour les animer
        self.prop_joint_indices = []
        for i in range(p.getNumJoints(self.drone_id)):
            info = p.getJointInfo(self.drone_id, i)
            joint_name = info[1].decode("utf-8")
            if joint_name.startswith("j_p"):
                self.prop_joint_indices.append(i)

        self._prop_angle = 0.0
        self.goal_marker_id = None
        self.obstacle_ids: list[int] = []

        # --- Nouveaux attributs pour la grille de champ ---
        self.cell_ids: list[list[int]] = []  # 2D liste d'IDs des cubes
        self.cell_size = 1.0                 # taille d'une cellule en mètres
        self.grid_offset = (0, 0)            # décalage pour centrer la grille
        self.field_created = False

    # ------------------------------------------------------------------
    def reset_scene(self, goal_position: np.ndarray, obstacles: list[dict]) -> None:
        """Recrée les marqueurs visuels (objectif + obstacles) à chaque reset()."""
        # Supprime les anciens marqueurs
        if self.goal_marker_id is not None:
            p.removeBody(self.goal_marker_id)
        for oid in self.obstacle_ids:
            p.removeBody(oid)
        self.obstacle_ids = []

        # Marqueur de l'objectif (sphère verte translucide)
        goal_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 0.5]
        )
        self.goal_marker_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=goal_visual,
            basePosition=goal_position.tolist()
        )

        # Obstacles (sphères rouges opaques)
        for obs in obstacles:
            visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=obs["radius"], rgbaColor=[0.8, 0.1, 0.1, 0.9]
            )
            oid = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=visual,
                basePosition=obs["pos"].tolist()
            )
            self.obstacle_ids.append(oid)

    # ------------------------------------------------------------------
    def create_field_grid(self, field_size: tuple, world_bounds: dict) -> None:
        """
        Crée la grille de cellules agricoles.
        field_size : (nx, ny) nombre de cellules en x et y
        world_bounds : {"x": (xmin,xmax), "y": (ymin,ymax)}
        """
        if self.field_created:
            return
        nx, ny = field_size
        x_min, x_max = world_bounds["x"]
        y_min, y_max = world_bounds["y"]

        self.cell_size = min((x_max - x_min) / nx, (y_max - y_min) / ny)
        self.grid_offset = ((x_min + x_max) / 2, (y_min + y_max) / 2)

        # Supprimer les anciennes cellules (au cas où)
        for row in self.cell_ids:
            for cid in row:
                p.removeBody(cid)
        self.cell_ids = []

        # Créer les cellules (cubes plats, hauteur 0.05)
        for i in range(nx):
            row_ids = []
            for j in range(ny):
                # Calcul de la position du centre de la cellule
                x = x_min + (i + 0.5) * (x_max - x_min) / nx
                y = y_min + (j + 0.5) * (y_max - y_min) / ny
                z = 0.025  # juste au-dessus du sol

                # Forme de base (cube plat)
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[self.cell_size * 0.45, self.cell_size * 0.45, 0.025],
                    rgbaColor=[0.2, 0.6, 0.2, 0.8]  # vert par défaut
                )
                collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[self.cell_size * 0.45, self.cell_size * 0.45, 0.025]
                )
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual,
                    baseCollisionShapeIndex=collision,
                    basePosition=[x, y, z]
                )
                row_ids.append(body_id)
            self.cell_ids.append(row_ids)
        self.field_created = True

    # ------------------------------------------------------------------
    def update_field(self, field_grid: list[list]) -> None:
        """
        Met à jour les couleurs des cellules en fonction de leur état.
        field_grid : une liste de listes de FieldCell (ou tout objet avec attributs healthy, wet, sprayed, watered)
        """
        if not self.field_created:
            return

        # Définition des couleurs selon l'état
        colors = {
            "healthy_wet": [0.2, 0.8, 0.2, 0.8],      # vert sain
            "healthy_dry": [0.8, 0.6, 0.2, 0.8],      # jaune (sec)
            "diseased_wet": [0.9, 0.1, 0.1, 0.8],     # rouge (malade)
            "diseased_dry": [0.7, 0.2, 0.1, 0.8],     # orange foncé
            "sprayed": [0.1, 0.3, 0.8, 0.8],          # bleu (pulvérisé)
            "watered": [0.1, 0.6, 0.9, 0.8],          # bleu clair (arrosé)
            "default": [0.5, 0.5, 0.5, 0.5]           # gris
        }

        for i, row in enumerate(field_grid):
            for j, cell in enumerate(row):
                if i >= len(self.cell_ids) or j >= len(self.cell_ids[i]):
                    continue
                body_id = self.cell_ids[i][j]

                # Déterminer la couleur
                if not cell.healthy and not cell.wet:
                    color = colors["diseased_dry"]
                elif not cell.healthy:
                    color = colors["diseased_wet"]
                elif not cell.wet:
                    color = colors["healthy_dry"]
                elif cell.sprayed or cell.watered:
                    # Si traité ou arrosé, on le met en bleu pour indiquer l'intervention
                    color = colors["watered"] if cell.watered else colors["sprayed"]
                else:
                    color = colors["healthy_wet"]

                # Mettre à jour la couleur
                p.changeVisualShape(body_id, -1, rgbaColor=color)

    # ------------------------------------------------------------------
    def update(self, state, throttle_normalized: float) -> None:
        """
        Synchronise la pose du drone dans PyBullet avec l'état calculé
        par drone_dynamics.py, et anime la rotation des hélices en
        fonction du throttle (purement visuel).
        """
        position = [state.x, state.y, state.z]
        orientation = p.getQuaternionFromEuler([state.roll, state.pitch, state.yaw])
        p.resetBasePositionAndOrientation(self.drone_id, position, orientation)

        # Animation des hélices (vitesse de rotation proportionnelle au throttle)
        self._prop_angle += (5.0 + throttle_normalized * 40.0)
        for idx in self.prop_joint_indices:
            p.resetJointState(self.drone_id, idx, self._prop_angle)

        p.stepSimulation()

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)


if __name__ == "__main__":
    import time
    import os
    print("=== Test autonome du Renderer PyBullet ===")
    # Création d'un URDF temporaire pour le test
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_test = os.path.join(current_dir, "drone_test.urdf")
    if not os.path.exists(urdf_test):
        with open(urdf_test, "w") as f:
            f.write('''<robot name="test_drone">
                <link name="base_link"><visual><geometry><box size="0.5 0.5 0.1"/></geometry></visual></link>
                <link name="prop1"><visual><geometry><cylinder radius="0.2" length="0.01"/></geometry></visual></link>
                <joint name="j_p1" type="continuous"><parent link="base_link"/><child link="prop1"/><origin xyz="0.3 0.3 0.05"/></joint>
            </robot>''')

    renderer = PyBulletRenderer(urdf_path=urdf_test)
    faux_objectif = np.array([5.0, 5.0, 4.0])
    faux_obstacles = [{"pos": np.array([2.0, 2.0, 1.0]), "radius": 1.0}]
    renderer.reset_scene(faux_objectif, faux_obstacles)

    # Créer une grille fictive pour tester
    class FakeCell:
        def __init__(self, h, w):
            self.healthy = h
            self.wet = w
            self.sprayed = False
            self.watered = False

    grid = [[FakeCell(True, True) for _ in range(10)] for _ in range(10)]
    # Mettre quelques cellules malades et sèches
    grid[2][3].healthy = False
    grid[2][4].wet = False
    grid[5][5].healthy = False
    grid[5][5].wet = False
    grid[7][8].sprayed = True

    renderer.create_field_grid((10, 10), {"x": (-10,10), "y": (-10,10)})
    renderer.update_field(grid)

    class FakeState:
        x, y, z = 0.0, 0.0, 1.0
        roll, pitch, yaw = 0.0, 0.0, 0.0

    state = FakeState()
    print("🟢 Fenêtre PyBullet ouverte. Fermeture automatique dans 10 secondes...")
    for _ in range(500):
        state.z += 0.01
        state.yaw += 0.02
        renderer.update(state, throttle_normalized=0.6)
        time.sleep(0.02)

    renderer.close()
    print("🟢 Test terminé.")