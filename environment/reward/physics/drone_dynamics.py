"""
physics/drone_dynamics.py
==========================
Modèle physique simplifié avec débogage optionnel.
"""

from dataclasses import dataclass
import numpy as np

DEBUG = True   # Mettre à False pour désactiver les logs

@dataclass
class DroneState:
    x: float = 0.0
    y: float = 0.0
    z: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])


@dataclass
class DroneParams:
    dry_mass: float = 10.0
    payload_mass: float = 5.0
    gravity: float = 9.81
    max_thrust_total: float = 350.0
    drag_coefficient: float = 0.15
    max_tilt_angle_rad: float = 0.5236
    max_angular_rate: float = 3.0
    attitude_time_constant: float = 0.15
    max_velocity: float = 15.0

    @property
    def total_mass(self) -> float:
        return self.dry_mass + self.payload_mass


class DroneDynamics:
    def __init__(self, params: DroneParams, world_bounds: dict, dt: float = 0.05):
        self.params = params
        self.world_bounds = world_bounds
        self.dt = dt
        self.state = DroneState()
        self._step_counter = 0   # pour limiter les logs

    def reset(self, position: np.ndarray) -> None:
        self.state = DroneState(x=position[0], y=position[1], z=position[2])
        self._step_counter = 0

    def step(self, action: np.ndarray) -> DroneState:
        self._step_counter += 1
        throttle_cmd = float(np.clip(action[0], -1.0, 1.0))
        roll_cmd = float(np.clip(action[1], -1.0, 1.0))
        pitch_cmd = float(np.clip(action[2], -1.0, 1.0))
        yaw_cmd = float(np.clip(action[3], -1.0, 1.0))

        p = self.params
        s = self.state

        # Poussée
        throttle_normalized = (throttle_cmd + 1.0) / 2.0
        thrust = throttle_normalized * p.max_thrust_total

        # Consignes d'attitude
        desired_roll = roll_cmd * p.max_tilt_angle_rad
        desired_pitch = pitch_cmd * p.max_tilt_angle_rad
        desired_yaw_rate = yaw_cmd * p.max_angular_rate

        # Réponse du 1er ordre
        tau = p.attitude_time_constant
        roll_rate = (desired_roll - s.roll) / tau
        pitch_rate = (desired_pitch - s.pitch) / tau
        roll_rate = np.clip(roll_rate, -p.max_angular_rate, p.max_angular_rate)
        pitch_rate = np.clip(pitch_rate, -p.max_angular_rate, p.max_angular_rate)

        s.roll += roll_rate * self.dt
        s.pitch += pitch_rate * self.dt
        s.yaw += desired_yaw_rate * self.dt
        s.yaw = (s.yaw + np.pi) % (2*np.pi) - np.pi

        s.roll_rate = roll_rate
        s.pitch_rate = pitch_rate
        s.yaw_rate = desired_yaw_rate

        # Accélérations
        m = p.total_mass
        # Projection de la poussée
        ax = (thrust / m) * (np.cos(s.roll)*np.sin(s.pitch)*np.cos(s.yaw) + np.sin(s.roll)*np.sin(s.yaw))
        ay = (thrust / m) * (np.cos(s.roll)*np.sin(s.pitch)*np.sin(s.yaw) - np.sin(s.roll)*np.cos(s.yaw))
        az = (thrust / m) * np.cos(s.roll)*np.cos(s.pitch) - p.gravity

        # Traînée
        ax -= (p.drag_coefficient / m) * s.vx
        ay -= (p.drag_coefficient / m) * s.vy
        az -= (p.drag_coefficient / m) * s.vz

        # Intégration de la vitesse
        s.vx += ax * self.dt
        s.vy += ay * self.dt
        s.vz += az * self.dt

        # Saturation de vitesse
        speed = np.linalg.norm([s.vx, s.vy, s.vz])
        if speed > p.max_velocity:
            scale = p.max_velocity / speed
            s.vx *= scale
            s.vy *= scale
            s.vz *= scale

        # Intégration de la position
        s.x += s.vx * self.dt
        s.y += s.vy * self.dt
        s.z += s.vz * self.dt

        # Sol
        if s.z < 0.05:
            s.z = 0.05
            s.vz = 0.0

        # Logs de débogage (toutes les 50 étapes)
        if DEBUG and self._step_counter % 50 == 0:
            print(f"[DEBUG] Step {self._step_counter}: "
                  f"thrust={thrust:.1f} N, pitch={s.pitch:.3f} rad, "
                  f"ax={ax:.2f} m/s², vx={s.vx:.2f} m/s, "
                  f"alt={s.z:.2f} m")

        return s

    def is_out_of_bounds(self) -> bool:
        s = self.state
        xb, yb, zb = self.world_bounds["x"], self.world_bounds["y"], self.world_bounds["z"]
        return not (xb[0] <= s.x <= xb[1] and yb[0] <= s.y <= yb[1] and zb[0] <= s.z <= zb[1])

    def is_flipped(self) -> bool:
        limit = np.deg2rad(80)
        return abs(self.state.roll) > limit or abs(self.state.pitch) > limit