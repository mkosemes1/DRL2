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
    print("=== Test autonome du Renderer PyBullet ===")
    
    # Configuration d'un chemin temporaire pour tester (ou utilisez votre vrai URDF)
    # Note: Si vous n'avez pas de fichier URDF sous la main, créez un fichier "test.urdf" minimal
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_test = os.path.join(current_dir, "drone_test.urdf")
    if not os.path.exists(urdf_test):
        # Création d'un URDF de secours ultra-simple (une boîte avec 4 hélices)
        with open(urdf_test, "w") as f:
            f.write('''<robot name="test_drone">
                <link name="base_link"><visual><geometry><box size="0.5 0.5 0.1"/></geometry></visual></link>
                <link name="prop1"><visual><geometry><cylinder radius="0.2" length="0.01"/></geometry></visual></link>
                <joint name="j_p1" type="continuous"><parent link="base_link"/><child link="prop1"/><origin xyz="0.3 0.3 0.05"/></joint>
            </robot>''')

    # Initialisation du renderer
    renderer = PyBulletRenderer(urdf_path=urdf_test)
    
    # Données fictives pour l'objectif et les obstacles
    faux_objectif = np.array([5.0, 5.0, 4.0])
    faux_obstacles = [
        {"pos": np.array([2.0, 2.0, 1.0]), "radius": 1.0},
        {"pos": np.array([-3.0, 4.0, 2.0]), "radius": 1.5}
    ]
    
    renderer.reset_scene(faux_objectif, faux_obstacles)
    
    # Classe factice pour simuler l'objet 'state' attendu par update()
    class FakeState:
        x, y, z = 0.0, 0.0, 1.0
        roll, pitch, yaw = 0.0, 0.0, 0.0
    
    state = FakeState()
    print("🟢 Fenêtre PyBullet ouverte. Fermeture automatique dans 10 secondes...")
    
    # Boucle d'animation (simulation de mouvement)
    for t in range(500):
        state.z += 0.01          # Le drone monte
        state.yaw += 0.02         # Le drone tourne sur lui-même
        state.pitch = 0.2 * np.sin(t * 0.05) # Oscillation
        
        renderer.update(state, throttle_normalized=0.6)
        time.sleep(0.02) # Calage à ~50 FPS
        
    renderer.close()
    print("🟢 Test terminé.")
