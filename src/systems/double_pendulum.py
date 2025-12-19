"""
Double Pendulum System

A double pendulum consists of two pendulums attached end-to-end.
It exhibits rich chaotic behavior for large initial angles and demonstrates
extreme sensitivity to initial conditions.

State variables: [θ1, ω1, θ2, ω2]
- θ1, θ2: Angles of pendulum 1 and 2 (radians)
- ω1, ω2: Angular velocities

System exhibits chaos for larger initial angles (> ~90 degrees).
"""

import numpy as np
from typing import Tuple
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.systems.base import ChaosSystem


class DoublePendulum(ChaosSystem):
    """
    Double Pendulum implementation.

    Uses the Lagrangian formulation to derive equations of motion.
    """

    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        g: float = 9.81
    ):
        """
        Initialize double pendulum.

        Args:
            m1: Mass of first pendulum (kg)
            m2: Mass of second pendulum (kg)
            l1: Length of first pendulum (m)
            l2: Length of second pendulum (m)
            g: Gravitational acceleration (m/s²)
        """
        super().__init__(name="Double Pendulum")
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

    def derivative(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """
        Compute derivatives for the double pendulum.

        Uses the Lagrangian formulation:
        L = T - V (kinetic energy - potential energy)

        Args:
            state: Current state [θ1, ω1, θ2, ω2]
            t: Time (not used)

        Returns:
            Derivatives [dθ1/dt, dω1/dt, dθ2/dt, dω2/dt]
        """
        theta1, omega1, theta2, omega2 = state

        # Differences
        delta_theta = theta2 - theta1

        # Denominators for equations
        denom1 = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * np.cos(delta_theta)**2
        denom2 = (self.l2 / self.l1) * denom1

        # Angular accelerations from Lagrangian mechanics
        alpha1 = (
            self.m2 * self.l1 * omega1**2 * np.sin(delta_theta) * np.cos(delta_theta)
            + self.m2 * self.g * np.sin(theta2) * np.cos(delta_theta)
            + self.m2 * self.l2 * omega2**2 * np.sin(delta_theta)
            - (self.m1 + self.m2) * self.g * np.sin(theta1)
        ) / denom1

        alpha2 = (
            -self.m2 * self.l2 * omega2**2 * np.sin(delta_theta) * np.cos(delta_theta)
            + (self.m1 + self.m2) * self.g * np.sin(theta1) * np.cos(delta_theta)
            - (self.m1 + self.m2) * self.l1 * omega1**2 * np.sin(delta_theta)
            - (self.m1 + self.m2) * self.g * np.sin(theta2)
        ) / denom2

        return np.array([omega1, alpha1, omega2, alpha2])

    def get_cartesian_positions(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert angular state to Cartesian coordinates for visualization.

        Args:
            state: State [θ1, ω1, θ2, ω2] or array of states

        Returns:
            Tuple of ((x1, y1), (x2, y2)) positions
        """
        if state.ndim == 1:
            # Single state
            theta1, _, theta2, _ = state

            x1 = self.l1 * np.sin(theta1)
            y1 = -self.l1 * np.cos(theta1)

            x2 = x1 + self.l2 * np.sin(theta2)
            y2 = y1 - self.l2 * np.cos(theta2)

            return (np.array([x1]), np.array([y1])), (np.array([x2]), np.array([y2]))
        else:
            # Multiple states
            theta1 = state[:, 0]
            theta2 = state[:, 2]

            x1 = self.l1 * np.sin(theta1)
            y1 = -self.l1 * np.cos(theta1)

            x2 = x1 + self.l2 * np.sin(theta2)
            y2 = y1 - self.l2 * np.cos(theta2)

            return (x1, y1), (x2, y2)

    def total_energy(self, state: np.ndarray) -> float:
        """
        Calculate total energy of the system.

        Energy should be conserved (useful for validation).

        Args:
            state: State [θ1, ω1, θ2, ω2]

        Returns:
            Total energy (kinetic + potential)
        """
        theta1, omega1, theta2, omega2 = state

        # Kinetic energy
        T1 = 0.5 * self.m1 * (self.l1 * omega1)**2
        T2 = 0.5 * self.m2 * (
            (self.l1 * omega1)**2 + (self.l2 * omega2)**2 +
            2 * self.l1 * self.l2 * omega1 * omega2 * np.cos(theta1 - theta2)
        )

        # Potential energy (reference at y=0)
        y1 = -self.l1 * np.cos(theta1)
        y2 = y1 - self.l2 * np.cos(theta2)

        V1 = self.m1 * self.g * y1
        V2 = self.m2 * self.g * y2

        return T1 + T2 + V1 + V2

    def get_default_initial_state(self, chaos: bool = True) -> np.ndarray:
        """
        Get a default initial state.

        Args:
            chaos: If True, use large angles for chaotic motion
                  If False, use small angles for regular motion

        Returns:
            Initial state [θ1, ω1, θ2, ω2]
        """
        if chaos:
            # Large angle for chaos (120 degrees)
            return np.array([np.pi * 2/3, 0, np.pi * 2/3, 0])
        else:
            # Small angle for regular motion (10 degrees)
            return np.array([np.pi / 18, 0, np.pi / 18, 0])

    def get_multiple_initial_states(
        self,
        num_states: int = 5,
        base_angle: float = np.pi * 2/3,
        perturbation: float = 0.001
    ) -> list:
        """
        Get multiple nearby initial states to demonstrate sensitivity.

        Args:
            num_states: Number of initial states
            base_angle: Base angle for first pendulum (radians)
            perturbation: Size of perturbation (radians)

        Returns:
            List of initial state vectors
        """
        base_state = np.array([base_angle, 0, base_angle, 0])
        states = [base_state]

        for _ in range(num_states - 1):
            # Perturb only the angles slightly
            pert = perturbation * np.random.randn(2)
            perturbed = base_state.copy()
            perturbed[0] += pert[0]
            perturbed[2] += pert[1]
            states.append(perturbed)

        return states

    def get_trajectory_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get bounds for visualization.

        Returns:
            Tuple of ((x_min, x_max), (y_min, y_max))
        """
        total_length = self.l1 + self.l2
        margin = 0.1
        bound = total_length * (1 + margin)

        return ((-bound, bound), (-bound, bound))

    def is_chaotic_regime(self, state: np.ndarray, threshold: float = np.pi/4) -> bool:
        """
        Simple heuristic to check if in chaotic regime.

        Args:
            state: Current state
            threshold: Angle threshold (radians)

        Returns:
            True if likely in chaotic regime
        """
        theta1, _, theta2, _ = state
        return abs(theta1) > threshold or abs(theta2) > threshold

    def __repr__(self) -> str:
        """String representation."""
        return (f"DoublePendulum(m1={self.m1:.2f}, m2={self.m2:.2f}, "
                f"l1={self.l1:.2f}, l2={self.l2:.2f})")


def create_standard_double_pendulum() -> DoublePendulum:
    """
    Create double pendulum with standard parameters.

    Returns:
        DoublePendulum with equal masses and lengths
    """
    return DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0)


def demonstrate_chaos(
    num_pendulums: int = 5,
    perturbation: float = 0.001,
    t_span: Tuple[float, float] = (0, 20)
) -> list:
    """
    Demonstrate chaotic behavior with slightly different initial conditions.

    Args:
        num_pendulums: Number of pendulums to simulate
        perturbation: Size of perturbation in initial conditions
        t_span: Time span for simulation

    Returns:
        List of trajectories
    """
    pendulum = create_standard_double_pendulum()
    initial_states = pendulum.get_multiple_initial_states(
        num_states=num_pendulums,
        perturbation=perturbation
    )

    trajectories = []
    for state in initial_states:
        traj = pendulum.solve(state, t_span=t_span, dt=0.01)
        trajectories.append(traj)

    return trajectories


if __name__ == "__main__":
    # Quick test
    pendulum = create_standard_double_pendulum()
    print(f"Created {pendulum}")

    # Generate chaotic trajectory
    initial = pendulum.get_default_initial_state(chaos=True)
    print(f"Initial state: θ1={np.degrees(initial[0]):.1f}°, "
          f"θ2={np.degrees(initial[2]):.1f}°")

    trajectory = pendulum.solve(initial, t_span=(0, 20), dt=0.01)
    print(f"Generated trajectory with {len(trajectory)} points")

    # Check energy conservation
    E0 = pendulum.total_energy(initial)
    E_final = pendulum.total_energy(trajectory[-1])
    print(f"Energy: E0={E0:.6f}, E_final={E_final:.6f}, "
          f"drift={(E_final-E0)/E0*100:.4f}%")
