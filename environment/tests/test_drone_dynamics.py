"""
tests/test_drone_dynamics.py
==============================
Tests unitaires pour DroneDynamics, DroneState et DroneParams.

Vérifie la dynamique physique du drone :
  - État par défaut et propriétés
  - Paramètres physiques (masse totale)
  - Reset, step, collision au sol, clamp de vitesse
  - Bornes hors carte, retournement, enveloppe de yaw
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.drone_dynamics import DroneDynamics, DroneState, DroneParams


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def default_params():
    """Paramètres physiques par défaut."""
    return DroneParams()


@pytest.fixture
def world_bounds():
    """Bornes du monde pour les tests."""
    return {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}


@pytest.fixture
def dynamics(default_params, world_bounds):
    """Instance de DroneDynamics initialisée."""
    dyn = DroneDynamics(default_params, world_bounds, dt=0.05)
    dyn.reset(np.array([0.0, 0.0, 1.0]))
    return dyn


# ─── Tests DroneState ─────────────────────────────────────────────

class TestDroneState:
    """Tests pour la classe DroneState."""

    def test_drone_state_defaults(self):
        """DroneState doit avoir des valeurs par défaut : x=0, y=0, z=1, vitesses=0."""
        state = DroneState()
        assert state.x == 0.0
        assert state.y == 0.0
        assert state.z == 1.0
        assert state.vx == 0.0
        assert state.vy == 0.0
        assert state.vz == 0.0
        assert state.roll == 0.0
        assert state.pitch == 0.0
        assert state.yaw == 0.0

    def test_drone_state_position(self):
        """position() doit retourner np.array([x, y, z])."""
        state = DroneState(x=1.0, y=2.0, z=3.0)
        pos = state.position()
        np.testing.assert_array_almost_equal(pos, [1.0, 2.0, 3.0])

    def test_drone_state_velocity(self):
        """velocity() doit retourner np.array([vx, vy, vz])."""
        state = DroneState(vx=4.0, vy=5.0, vz=6.0)
        vel = state.velocity()
        np.testing.assert_array_almost_equal(vel, [4.0, 5.0, 6.0])


# ─── Tests DroneParams ────────────────────────────────────────────

class TestDroneParams:
    """Tests pour la classe DroneParams."""

    def test_drone_params_total_mass(self):
        """total_mass = dry_mass + payload_mass."""
        params = DroneParams(dry_mass=10.0, payload_mass=5.0)
        assert params.total_mass == pytest.approx(15.0)

    def test_drone_params_defaults(self):
        """Valeurs par défaut correspondent aux attentes."""
        params = DroneParams()
        assert params.dry_mass == 10.0
        assert params.payload_mass == 5.0
        assert params.gravity == 9.81
        assert params.max_thrust_total == 350.0
        assert params.max_velocity == 15.0


# ─── Tests DroneDynamics ──────────────────────────────────────────

class TestDroneDynamicsReset:
    """Tests pour la méthode reset()."""

    def test_dynamics_reset(self, dynamics):
        """Reset positionne le drone et annule les vitesses."""
        dynamics.reset(np.array([5.0, 6.0, 2.0]))
        state = dynamics.state
        assert state.x == pytest.approx(5.0)
        assert state.y == pytest.approx(6.0)
        assert state.z == pytest.approx(2.0)
        assert state.vx == 0.0
        assert state.vy == 0.0
        assert state.vz == 0.0


class TestDroneDynamicsStep:
    """Tests pour la méthode step()."""

    def test_dynamics_step_upward(self, dynamics):
        """Throttle élevé, roll=0, pitch=0 → z augmente."""
        action = np.array([0.8, 0.0, 0.0, 0.0])
        z_before = dynamics.state.z
        for _ in range(10):
            dynamics.step(action)
        assert dynamics.state.z > z_before

    def test_dynamics_step_forward(self, dynamics):
        """Throttle élevé, pitch=0.5 → x augmente (le drone avance)."""
        action = np.array([0.8, 0.0, 0.5, 0.0])
        for _ in range(20):
            dynamics.step(action)
        assert dynamics.state.x > 0.0

    def test_dynamics_zero_action(self, dynamics):
        """Action zéro → mouvement minimal (uniquement gravité)."""
        action = np.array([0.0, 0.0, 0.0, 0.0])
        pos_before = dynamics.state.position().copy()
        for _ in range(5):
            dynamics.step(action)
        # Le drone tombe sous l'effet de la gravité avec throttle=0
        # La position change mais reste proche
        dist = np.linalg.norm(dynamics.state.position() - pos_before)
        # L'action est nulle mais la gravité fait bouger le drone
        # Vérifie que le mouvement reste modeste
        assert dist < 5.0


class TestDroneDynamicsGroundCollision:
    """Tests pour la collision au sol."""

    def test_dynamics_ground_collision(self):
        """z ne descend jamais en dessous de 0.05."""
        params = DroneParams()
        bounds = {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}
        dyn = DroneDynamics(params, bounds, dt=0.05)
        dyn.reset(np.array([0.0, 0.0, 0.5]))
        # Action zéro → le drone tombe
        action = np.array([-1.0, 0.0, 0.0, 0.0])
        for _ in range(200):
            dyn.step(action)
        assert dyn.state.z >= 0.05


class TestDroneDynamicsVelocityClamp:
    """Tests pour la limitation de vitesse."""

    def test_dynamics_velocity_clamp(self):
        """La vitesse ne dépasse jamais max_velocity."""
        params = DroneParams(max_velocity=5.0)
        bounds = {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}
        dyn = DroneDynamics(params, bounds, dt=0.05)
        dyn.reset(np.array([0.0, 0.0, 50.0]))
        # Throttle maximal pendant longtemps
        action = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(200):
            dyn.step(action)
            speed = np.linalg.norm([dyn.state.vx, dyn.state.vy, dyn.state.vz])
            assert speed <= params.max_velocity + 1e-6


class TestDroneDynamicsOutOfBounds:
    """Tests pour la détection hors bornes."""

    def test_dynamics_out_of_bounds(self):
        """is_out_of_bounds retourne True quand hors carte."""
        params = DroneParams()
        bounds = {"x": (-10.0, 10.0), "y": (-10.0, 10.0), "z": (0.0, 20.0)}
        dyn = DroneDynamics(params, bounds, dt=0.05)
        dyn.reset(np.array([15.0, 0.0, 1.0]))  # x > x_max
        assert dyn.is_out_of_bounds() is True

    def test_dynamics_not_out_of_bounds(self):
        """is_out_of_bounds retourne False quand dans la carte."""
        params = DroneParams()
        bounds = {"x": (-10.0, 10.0), "y": (-10.0, 10.0), "z": (0.0, 20.0)}
        dyn = DroneDynamics(params, bounds, dt=0.05)
        dyn.reset(np.array([0.0, 0.0, 1.0]))
        assert dyn.is_out_of_bounds() is False


class TestDroneDynamicsFlipped:
    """Tests pour la détection de retournement."""

    def test_dynamics_flipped(self):
        """is_flipped retourne True quand roll > 80°."""
        params = DroneParams()
        bounds = {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}
        dyn = DroneDynamics(params, bounds, dt=0.05)
        dyn.reset(np.array([0.0, 0.0, 50.0]))
        dyn.state.roll = np.deg2rad(85)  # 85° > 80°
        assert dyn.is_flipped() is True

    def test_dynamics_not_flipped(self):
        """is_flipped retourne False pour de petits angles."""
        params = DroneParams()
        bounds = {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}
        dyn = DroneDynamics(params, bounds, dt=0.05)
        dyn.reset(np.array([0.0, 0.0, 50.0]))
        dyn.state.roll = np.deg2rad(10)  # 10° < 80°
        dyn.state.pitch = np.deg2rad(5)
        assert dyn.is_flipped() is False


class TestDroneDynamicsYawWrap:
    """Tests pour l'enveloppe du yaw dans [-pi, pi]."""

    def test_dynamics_yaw_wrap(self):
        """Le yaw reste toujours dans [-pi, pi]."""
        params = DroneParams()
        bounds = {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (0.0, 100.0)}
        dyn = DroneDynamics(params, bounds, dt=0.05)
        dyn.reset(np.array([0.0, 0.0, 50.0]))
        # Tourner le yaw en boucle
        action = np.array([0.5, 0.0, 0.0, 1.0])
        for _ in range(500):
            dyn.step(action)
            assert -np.pi - 1e-9 <= dyn.state.yaw <= np.pi + 1e-9
