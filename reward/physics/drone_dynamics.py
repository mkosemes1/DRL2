"""
physics/drone_dynamics.py
==========================
Modèle physique simplifié mais cohérent du drone agricole.

Principe :
  - Le drone est modélisé comme un corps rigide unique.
  - L'action [throttle, roll_cmd, pitch_cmd, yaw_cmd] pilote :
      * la poussée totale (throttle)
      * une commande d'inclinaison désirée (roll, pitch)
      * une commande de vitesse de lacet (yaw)
  - L'attitude (roll, pitch) suit un modèle de premier ordre vers la
    consigne (approxime la réponse du contrôleur bas-niveau réel).
  - La poussée est ensuite projetée dans le repère monde en fonction
    de l'inclinaison (roll, pitch) pour produire l'accélération
    horizontale -> c'est le mécanisme physique réel qui fait avancer
    un multirotor (on "penche" pour se déplacer).
  - Gravité, traînée aérodynamique linéaire, saturation de vitesse
    et d'accélération, limites de la carte sont incluses.

Toutes les équations sont commentées et correspondent à celles
demandées dans le cahier des charges :
    x(t+1) = x(t) + vx * dt
    v(t+1) = v(t) + a * dt
    a = F / m
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DroneState:
    """État complet du drone à un instant t."""
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
    """Paramètres physiques du drone (modifiables pour le Domain Randomization)."""
    dry_mass: float = 10.0
    payload_mass: float = 5.0          # variable : 0 (vide) à payload_mass_full (pleine)
    gravity: float = 9.81
    max_thrust_total: float = 350.0    # Newtons
    drag_coefficient: float = 0.15
    max_tilt_angle_rad: float = 0.5236  # 30°
    max_angular_rate: float = 3.0       # rad/s
    attitude_time_constant: float = 0.15
    max_velocity: float = 15.0

    @property
    def total_mass(self) -> float:
        return self.dry_mass + self.payload_mass


class DroneDynamics:
    """
    Intègre la dynamique du drone pas de temps par pas de temps (dt).
    Utilisé par l'environnement Gymnasium à chaque appel de step().
    """

    def __init__(self, params: DroneParams, world_bounds: dict, dt: float = 0.05):
        self.params = params
        self.world_bounds = world_bounds  # {"x": (min,max), "y": (min,max), "z": (min,max)}
        self.dt = dt
        self.state = DroneState()

    def reset(self, position: np.ndarray) -> None:
        """Réinitialise l'état du drone à une position donnée, vitesses nulles."""
        self.state = DroneState(x=position[0], y=position[1], z=position[2])

    def step(self, action: np.ndarray) -> DroneState:
        """
        Applique une action normalisée [-1,1]^4 = [throttle, roll_cmd, pitch_cmd, yaw_cmd]
        et fait avancer la simulation d'un pas dt.
        """
        throttle_cmd = float(np.clip(action[0], -1.0, 1.0))
        roll_cmd = float(np.clip(action[1], -1.0, 1.0))
        pitch_cmd = float(np.clip(action[2], -1.0, 1.0))
        yaw_cmd = float(np.clip(action[3], -1.0, 1.0))

        p = self.params
        s = self.state

        # --- 1. Poussée totale (throttle in [-1,1] -> [0,1]) ---
        throttle_normalized = (throttle_cmd + 1.0) / 2.0
        thrust = throttle_normalized * p.max_thrust_total  # Newtons

        # --- 2. Consignes d'angle désirées (roll, pitch) ---
        desired_roll = roll_cmd * p.max_tilt_angle_rad
        desired_pitch = pitch_cmd * p.max_tilt_angle_rad
        desired_yaw_rate = yaw_cmd * p.max_angular_rate

        # --- 3. Réponse d'attitude du 1er ordre (modélise le contrôleur bas niveau) ---
        #   d(roll)/dt = (desired_roll - roll) / tau
        tau = p.attitude_time_constant
        roll_rate = (desired_roll - s.roll) / tau
        pitch_rate = (desired_pitch - s.pitch) / tau
        roll_rate = float(np.clip(roll_rate, -p.max_angular_rate, p.max_angular_rate))
        pitch_rate = float(np.clip(pitch_rate, -p.max_angular_rate, p.max_angular_rate))

        s.roll += roll_rate * self.dt
        s.pitch += pitch_rate * self.dt
        s.yaw += desired_yaw_rate * self.dt
        s.yaw = (s.yaw + np.pi) % (2 * np.pi) - np.pi  # wrap [-pi, pi]

        s.roll_rate = roll_rate
        s.pitch_rate = pitch_rate
        s.yaw_rate = desired_yaw_rate

        # --- 4. Projection de la poussée dans le repère monde ---
        # Approximation standard multirotor (petits angles non supposés, formule complète) :
        m = p.total_mass
        ax = (thrust / m) * (np.cos(s.roll) * np.sin(s.pitch) * np.cos(s.yaw)
                              + np.sin(s.roll) * np.sin(s.yaw))
        ay = (thrust / m) * (np.cos(s.roll) * np.sin(s.pitch) * np.sin(s.yaw)
                              - np.sin(s.roll) * np.cos(s.yaw))
        az = (thrust / m) * (np.cos(s.roll) * np.cos(s.pitch)) - p.gravity

        # --- 5. Traînée aérodynamique linéaire : F_drag = -k * v ---
        ax -= (p.drag_coefficient / m) * s.vx
        ay -= (p.drag_coefficient / m) * s.vy
        az -= (p.drag_coefficient / m) * s.vz

        # --- 6. Intégration de la vitesse : v(t+1) = v(t) + a*dt ---
        s.vx += ax * self.dt
        s.vy += ay * self.dt
        s.vz += az * self.dt

        # Saturation de la vitesse (limite physique du drone)
        speed = np.linalg.norm([s.vx, s.vy, s.vz])
        if speed > p.max_velocity:
            scale = p.max_velocity / speed
            s.vx *= scale
            s.vy *= scale
            s.vz *= scale

        # --- 7. Intégration de la position : x(t+1) = x(t) + v*dt ---
        s.x += s.vx * self.dt
        s.y += s.vy * self.dt
        s.z += s.vz * self.dt

        # Sol : ne jamais passer sous z=0
        if s.z < 0.05:
            s.z = 0.05
            s.vz = 0.0

        return s

    def is_out_of_bounds(self) -> bool:
        s = self.state
        xb, yb, zb = self.world_bounds["x"], self.world_bounds["y"], self.world_bounds["z"]
        return not (xb[0] <= s.x <= xb[1] and yb[0] <= s.y <= yb[1] and zb[0] <= s.z <= zb[1])

    def is_flipped(self) -> bool:
        """Considère le drone comme retourné si roll/pitch dépasse 80°."""
        limit = np.deg2rad(80)
        return abs(self.state.roll) > limit or abs(self.state.pitch) > limit