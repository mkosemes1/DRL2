import pybullet as p
import numpy as np

class DroneWrapper:
    def __init__(self, urdf_path, client_id, start_pos=[0, 0, 1], start_ori=[0, 0, 0, 1]):
        self.client_id = client_id  # <--- INDISPENSABLE
        self.start_pos = start_pos
        self.start_ori = start_ori
        
        # On charge le drone DANS LE BON CLIENT
        try:
            self.drone_id = p.loadURDF(urdf_path, self.start_pos, self.start_ori, physicsClientId=self.client_id)
        except Exception:
            self.drone_id = p.loadURDF("sphere_small.urdf", self.start_pos, self.start_ori, physicsClientId=self.client_id)
            print("⚠️ URDF drone non trouvé, sphère utilisée.")

    def reset(self, base_position=None):
        pos = base_position if base_position is not None else self.start_pos
        p.resetBasePositionAndOrientation(self.drone_id, pos, self.start_ori, physicsClientId=self.client_id)
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client_id)

    def apply_action(self, flight_action):
        """
        flight_action: [throttle, roll, pitch, yaw] entre -1 et 1
        """
        # Exemple de dynamique simple pour l'action spatiale :
        normalized_throttle = (flight_action[0] + 1.0) / 2.0  # Devient entre 0 et 1
        thrust = normalized_throttle * 350.0                  # Devient entre 0 et 350 N
        roll_cmd = flight_action[1] * 10.0
        pitch_cmd = flight_action[2] * 10.0
        yaw_cmd = flight_action[3] * 5.0

        # Poussée vers le haut
        p.applyExternalForce(self.drone_id, -1, forceObj=[0, 0, thrust], posObj=[0, 0, 0], 
                             flags=p.LINK_FRAME, physicsClientId=self.client_id)
        
        # Couples de rotation
        p.applyExternalTorque(self.drone_id, -1, torqueObj=[roll_cmd, pitch_cmd, yaw_cmd], 
                              flags=p.LINK_FRAME, physicsClientId=self.client_id)

    def get_raw_state(self):
        """
        Retourne les valeurs brutes pour que l'environnement puisse les normaliser.
        """
        pos, ori_quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.client_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(ori_quat)
        
        return np.array(pos), lin_vel, ang_vel, (roll, pitch, yaw)
