"""
Lorenz Attractor System

The Lorenz system is a set of ordinary differential equations first studied
by Edward Lorenz in 1963. It exhibits chaotic behavior and demonstrates
sensitive dependence on initial conditions (the "butterfly effect").

The system is defined by:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

Where σ, ρ, and β are system parameters.
"""

import numpy as np
from typing import Tuple
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.systems.base import ChaosSystem


class LorenzSystem(ChaosSystem):
    """
    Lorenz Attractor implementation.

    Classic parameters that produce the butterfly attractor:
    - σ (sigma) = 10.0
    - ρ (rho) = 28.0
    - β (beta) = 8/3 ≈ 2.667
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0/3.0
    ):
        """
        Initialize Lorenz system.

        Args:
            sigma: Prandtl number (ratio of momentum to thermal diffusivity)
            rho: Rayleigh number (temperature difference)
            beta: Geometric factor
        """
        super().__init__(name="Lorenz Attractor")
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def derivative(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """
        Compute derivatives for the Lorenz system.

        Args:
            state: Current state [x, y, z]
            t: Time (not used, system is autonomous)

        Returns:
            Derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state

        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        return np.array([dx, dy, dz])

    def get_fixed_points(self) -> list:
        """
        Calculate fixed points of the Lorenz system.

        Returns:
            List of fixed points as numpy arrays
        """
        # Origin is always a fixed point
        origin = np.array([0, 0, 0])

        # Two additional fixed points exist when ρ > 1
        if self.rho > 1:
            c = np.sqrt(self.beta * (self.rho - 1))
            fp1 = np.array([c, c, self.rho - 1])
            fp2 = np.array([-c, -c, self.rho - 1])
            return [origin, fp1, fp2]
        else:
            return [origin]

    def get_attractor_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Get typical bounds of the Lorenz attractor for visualization.

        Returns:
            Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        # These bounds work well for classic parameters
        return ((-25, 25), (-35, 35), (0, 50))

    def get_default_initial_state(self) -> np.ndarray:
        """
        Get a default initial state that produces nice trajectories.

        Returns:
            Initial state [x, y, z]
        """
        return np.array([1.0, 1.0, 1.0])

    def get_multiple_initial_states(self, num_states: int = 5, spread: float = 0.01) -> list:
        """
        Get multiple nearby initial states to demonstrate sensitivity.

        Args:
            num_states: Number of initial states
            spread: Spread of perturbations

        Returns:
            List of initial state vectors
        """
        base_state = self.get_default_initial_state()
        states = [base_state]

        for _ in range(num_states - 1):
            perturbation = spread * np.random.randn(3)
            states.append(base_state + perturbation)

        return states

    def classify_regime(self) -> str:
        """
        Classify the dynamical regime based on parameters.

        Returns:
            String describing the regime
        """
        if self.rho < 1:
            return "stable_fixed_point"
        elif 1 < self.rho < 13.926:
            return "stable_limit_cycle"
        elif 13.926 < self.rho < 24.74:
            return "chaotic_transient"
        elif self.rho > 24.74:
            return "chaotic_attractor"
        else:
            return "unknown"

    def __repr__(self) -> str:
        """String representation."""
        return (f"LorenzSystem(sigma={self.sigma:.2f}, "
                f"rho={self.rho:.2f}, beta={self.beta:.3f})")


def create_classic_lorenz() -> LorenzSystem:
    """
    Create Lorenz system with classic parameters.

    Returns:
        LorenzSystem with σ=10, ρ=28, β=8/3
    """
    return LorenzSystem(sigma=10.0, rho=28.0, beta=8.0/3.0)


def demonstrate_butterfly_effect(
    lorenz: LorenzSystem = None,
    perturbation: float = 1e-8,
    t_span: Tuple[float, float] = (0, 40)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Demonstrate the butterfly effect with two very close initial conditions.

    Args:
        lorenz: LorenzSystem instance (creates default if None)
        perturbation: Size of initial perturbation
        t_span: Time span for simulation

    Returns:
        Tuple of (trajectory1, trajectory2)
    """
    if lorenz is None:
        lorenz = create_classic_lorenz()

    initial1 = np.array([1.0, 1.0, 1.0])
    initial2 = initial1 + perturbation * np.array([1, 0, 0])

    traj1 = lorenz.solve(initial1, t_span=t_span, dt=0.01)
    traj2 = lorenz.solve(initial2, t_span=t_span, dt=0.01)

    return traj1, traj2


if __name__ == "__main__":
    # Quick test
    lorenz = create_classic_lorenz()
    print(f"Created {lorenz}")
    print(f"Regime: {lorenz.classify_regime()}")
    print(f"Fixed points: {lorenz.get_fixed_points()}")

    # Generate trajectory
    trajectory = lorenz.solve(
        initial_state=np.array([1, 1, 1]),
        t_span=(0, 50),
        dt=0.01
    )
    print(f"Generated trajectory with {len(trajectory)} points")
    print(f"Trajectory bounds: x={trajectory[:,0].min():.2f} to {trajectory[:,0].max():.2f}")
