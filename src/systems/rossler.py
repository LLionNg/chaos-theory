"""
Rössler Attractor System

The Rössler attractor is a system of three non-linear ordinary differential
equations discovered by Otto Rössler in 1976. It exhibits chaotic dynamics
with a characteristic scroll-like shape in phase space.

The system is defined by:
    dx/dt = -y - z
    dy/dt = x + ay
    dz/dt = b + z(x - c)

Where a, b, and c are system parameters.
"""

import numpy as np
from typing import Tuple
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.systems.base import ChaosSystem


class RosslerSystem(ChaosSystem):
    """
    Rössler Attractor implementation.

    Classic parameters that produce chaotic behavior:
    - a = 0.2
    - b = 0.2
    - c = 5.7

    The system exhibits different behaviors for different parameter values:
    - c < 2: Convergence to fixed point
    - c ≈ 2-4: Limit cycle
    - c > 4: Chaotic attractor
    """

    def __init__(
        self,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7
    ):
        """
        Initialize Rössler system.

        Args:
            a: First parameter (typically 0.1 to 0.5)
            b: Second parameter (typically 0.1 to 0.5)
            c: Third parameter (typically 2 to 10)
        """
        super().__init__(name="Rössler Attractor")
        self.a = a
        self.b = b
        self.c = c

    def derivative(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """
        Compute derivatives for the Rössler system.

        Args:
            state: Current state [x, y, z]
            t: Time (not used, system is autonomous)

        Returns:
            Derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state

        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)

        return np.array([dx, dy, dz])

    def get_attractor_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Get typical bounds of the Rössler attractor for visualization.

        Returns:
            Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        # These bounds work well for classic parameters
        return ((-15, 15), (-20, 15), (0, 30))

    def get_default_initial_state(self) -> np.ndarray:
        """
        Get a default initial state that produces nice trajectories.

        Returns:
            Initial state [x, y, z]
        """
        return np.array([1.0, 1.0, 1.0])

    def get_multiple_initial_states(
        self,
        num_states: int = 5,
        spread: float = 0.01
    ) -> list:
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
        if self.c < 2:
            return "fixed_point"
        elif self.c < 4:
            return "limit_cycle"
        elif self.c < 6:
            return "chaotic_attractor"
        elif self.c < 8.5:
            return "funnel_attractor"
        else:
            return "complex_dynamics"

    def get_scroll_center(self) -> np.ndarray:
        """
        Get approximate center of the scroll attractor.

        Returns:
            Center point [x, y, z]
        """
        # Approximate for classic parameters
        return np.array([0, 0, self.c - self.a])

    def get_poincare_section(
        self,
        trajectory: np.ndarray,
        plane: str = 'z',
        value: float = None
    ) -> np.ndarray:
        """
        Extract Poincaré section from trajectory.

        A Poincaré section is the intersection of a periodic orbit
        with a lower-dimensional subspace (plane).

        Args:
            trajectory: Full trajectory array
            plane: Which plane to use ('x', 'y', or 'z')
            value: Value of the plane (uses mean if None)

        Returns:
            Array of points where trajectory crosses the plane
        """
        if plane == 'x':
            idx = 0
        elif plane == 'y':
            idx = 1
        else:  # 'z'
            idx = 2

        if value is None:
            value = trajectory[:, idx].mean()

        # Find crossings
        crossings = []
        for i in range(len(trajectory) - 1):
            if (trajectory[i, idx] - value) * (trajectory[i+1, idx] - value) < 0:
                # Linear interpolation for crossing point
                t = (value - trajectory[i, idx]) / (trajectory[i+1, idx] - trajectory[i, idx])
                crossing = trajectory[i] + t * (trajectory[i+1] - trajectory[i])
                crossings.append(crossing)

        return np.array(crossings) if crossings else np.array([])

    def return_map(
        self,
        trajectory: np.ndarray,
        plane: str = 'z',
        value: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate return map from Poincaré section.

        Args:
            trajectory: Full trajectory array
            plane: Which plane to use
            value: Value of the plane

        Returns:
            Tuple of (x_n, x_n+1) arrays for plotting return map
        """
        section = self.get_poincare_section(trajectory, plane, value)

        if len(section) < 2:
            return np.array([]), np.array([])

        # Use first coordinate of section points
        x_n = section[:-1, 0]
        x_n1 = section[1:, 0]

        return x_n, x_n1

    def __repr__(self) -> str:
        """String representation."""
        return (f"RosslerSystem(a={self.a:.2f}, b={self.b:.2f}, c={self.c:.2f}, "
                f"regime={self.classify_regime()})")


def create_classic_rossler() -> RosslerSystem:
    """
    Create Rössler system with classic chaotic parameters.

    Returns:
        RosslerSystem with a=0.2, b=0.2, c=5.7
    """
    return RosslerSystem(a=0.2, b=0.2, c=5.7)


def create_funnel_rossler() -> RosslerSystem:
    """
    Create Rössler system with funnel attractor parameters.

    Returns:
        RosslerSystem with funnel-like dynamics
    """
    return RosslerSystem(a=0.2, b=0.2, c=8.0)


def demonstrate_parameter_variation(
    c_values: list = None,
    t_span: Tuple[float, float] = (0, 100)
) -> dict:
    """
    Demonstrate how changing parameter c affects the system.

    Args:
        c_values: List of c values to try (uses defaults if None)
        t_span: Time span for each simulation

    Returns:
        Dictionary mapping c values to trajectories
    """
    if c_values is None:
        c_values = [2.0, 3.5, 5.0, 5.7, 7.0, 9.0]

    results = {}
    initial_state = np.array([1.0, 1.0, 1.0])

    for c in c_values:
        rossler = RosslerSystem(a=0.2, b=0.2, c=c)
        traj = rossler.solve(initial_state, t_span=t_span, dt=0.01)
        results[c] = {
            'trajectory': traj,
            'regime': rossler.classify_regime()
        }

    return results


if __name__ == "__main__":
    # Quick test
    rossler = create_classic_rossler()
    print(f"Created {rossler}")

    # Generate trajectory
    initial = rossler.get_default_initial_state()
    trajectory = rossler.solve(initial, t_span=(0, 100), dt=0.01)
    print(f"Generated trajectory with {len(trajectory)} points")
    print(f"Trajectory bounds:")
    print(f"  x: {trajectory[:,0].min():.2f} to {trajectory[:,0].max():.2f}")
    print(f"  y: {trajectory[:,1].min():.2f} to {trajectory[:,1].max():.2f}")
    print(f"  z: {trajectory[:,2].min():.2f} to {trajectory[:,2].max():.2f}")

    # Poincaré section
    section = rossler.get_poincare_section(trajectory, plane='z', value=15)
    print(f"Poincaré section: {len(section)} crossings")

    # Test different parameters
    print("\nParameter variation:")
    for c in [2.0, 4.0, 5.7]:
        rs = RosslerSystem(a=0.2, b=0.2, c=c)
        print(f"  c={c}: {rs.classify_regime()}")
