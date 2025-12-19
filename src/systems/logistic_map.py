"""
Logistic Map

The logistic map is a discrete-time dynamical system that demonstrates how
complex, chaotic behavior can arise from simple nonlinear equations.

The map is defined by:
    x[n+1] = r * x[n] * (1 - x[n])

Where:
- x is the population (normalized between 0 and 1)
- r is the growth rate parameter (0 to 4)

Famous for its bifurcation diagram showing the route to chaos through
period-doubling bifurcations.
"""

import numpy as np
from typing import Tuple
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.systems.base import DiscreteSystem


class LogisticMap(DiscreteSystem):
    """
    Logistic Map implementation.

    Demonstrates period-doubling route to chaos:
    - r < 1: Population dies out
    - 1 < r < 3: Single stable fixed point
    - 3 < r < 1 + √6 ≈ 3.45: Period-2 oscillation
    - 3.45 < r < 3.54: Higher period oscillations
    - r > 3.57: Chaotic regime (with periodic windows)
    - r = 4: Full chaos
    """

    def __init__(self, r: float = 3.7):
        """
        Initialize logistic map.

        Args:
            r: Growth rate parameter (0 to 4)
        """
        super().__init__(name="Logistic Map")
        self.r = r

    def iterate(self, x: float, r: float = None) -> float:
        """
        Perform one iteration of the logistic map.

        Args:
            x: Current population value (0 to 1)
            r: Growth rate (uses self.r if None)

        Returns:
            Next population value
        """
        if r is None:
            r = self.r

        return r * x * (1 - x)

    def trajectory(
        self,
        x0: float = 0.1,
        num_iterations: int = 1000,
        transient: int = 100,
        r: float = None
    ) -> np.ndarray:
        """
        Generate trajectory for the logistic map.

        Args:
            x0: Initial value (0 to 1)
            num_iterations: Number of iterations to return
            transient: Number of initial iterations to discard
            r: Growth rate (uses self.r if None)

        Returns:
            Array of population values
        """
        if r is None:
            r = self.r

        total_iterations = num_iterations + transient
        trajectory = np.zeros(total_iterations)
        trajectory[0] = x0

        for i in range(1, total_iterations):
            trajectory[i] = self.iterate(trajectory[i-1], r)

        return trajectory[transient:]

    def bifurcation_diagram(
        self,
        r_min: float = 2.5,
        r_max: float = 4.0,
        num_r: int = 1000,
        num_iterations: int = 1000,
        transient: int = 100,
        x0: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for bifurcation diagram.

        Args:
            r_min: Minimum r value
            r_max: Maximum r value
            num_r: Number of r values to sample
            num_iterations: Iterations per r value
            transient: Initial iterations to discard
            x0: Initial x value for each r

        Returns:
            Tuple of (r_values, x_values) arrays
            - r_values: Array of r parameters
            - x_values: Corresponding x values (attractors)
        """
        r_values = np.linspace(r_min, r_max, num_r)
        all_r = []
        all_x = []

        for r in r_values:
            # Generate trajectory
            traj = self.trajectory(x0, num_iterations, transient, r)

            # Take last portion (attractor values)
            attractor_points = traj[-100:]  # Last 100 points

            # Store r and corresponding x values
            all_r.extend([r] * len(attractor_points))
            all_x.extend(attractor_points)

        return np.array(all_r), np.array(all_x)

    def cobweb_plot_data(
        self,
        x0: float = 0.1,
        num_iterations: int = 50,
        r: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for cobweb plot visualization.

        Cobweb plots show the iteration process graphically by
        bouncing between the function curve and the diagonal.

        Args:
            x0: Initial value
            num_iterations: Number of iterations
            r: Growth rate (uses self.r if None)

        Returns:
            Tuple of (x_coords, y_coords) for plotting cobweb
        """
        if r is None:
            r = self.r

        x_coords = [x0]
        y_coords = [0]

        x = x0
        for _ in range(num_iterations):
            # Vertical line to curve
            y = self.iterate(x, r)
            x_coords.append(x)
            y_coords.append(y)

            # Horizontal line to diagonal
            x_coords.append(y)
            y_coords.append(y)

            x = y

        return np.array(x_coords), np.array(y_coords)

    def find_period(
        self,
        x0: float = 0.1,
        r: float = None,
        max_iterations: int = 10000,
        transient: int = 1000,
        tolerance: float = 1e-6,
        max_period: int = 100
    ) -> int:
        """
        Detect the period of the orbit.

        Args:
            x0: Initial value
            r: Growth rate
            max_iterations: Maximum iterations to check
            transient: Initial iterations to discard
            tolerance: Tolerance for considering values equal
            max_period: Maximum period to detect

        Returns:
            Period of the orbit (0 if chaotic/not periodic)
        """
        if r is None:
            r = self.r

        # Generate trajectory
        traj = self.trajectory(x0, max_iterations, transient, r)

        # Check for periods
        for period in range(1, max_period + 1):
            is_periodic = True
            for i in range(len(traj) - period):
                if abs(traj[i] - traj[i + period]) > tolerance:
                    is_periodic = False
                    break
            if is_periodic:
                return period

        return 0  # Chaotic or period > max_period

    def classify_regime(self, r: float = None) -> str:
        """
        Classify the dynamical regime for given r.

        Args:
            r: Growth rate (uses self.r if None)

        Returns:
            String describing the regime
        """
        if r is None:
            r = self.r

        if r < 0:
            return "invalid"
        elif r < 1:
            return "extinction"
        elif r < 3:
            return "fixed_point"
        elif r < 1 + np.sqrt(6):
            return "period_2"
        elif r < 3.54:
            return "period_doubling"
        elif r < 3.5699:
            return "chaos_onset"
        elif r <= 4:
            return "chaos"
        else:
            return "invalid"

    def feigenbaum_delta(
        self,
        r_values: list = None
    ) -> float:
        """
        Estimate Feigenbaum delta constant.

        The Feigenbaum constant δ ≈ 4.669 describes the rate of
        period-doubling bifurcations.

        Args:
            r_values: List of bifurcation points (auto-detects if None)

        Returns:
            Estimated Feigenbaum delta
        """
        if r_values is None:
            # Approximate bifurcation points
            r_values = [3.0, 3.449, 3.544, 3.5688, 3.5696]

        if len(r_values) < 3:
            return 0.0

        # Calculate delta using successive bifurcations
        deltas = []
        for i in range(len(r_values) - 2):
            r_n = r_values[i]
            r_n1 = r_values[i + 1]
            r_n2 = r_values[i + 2]

            delta = (r_n1 - r_n) / (r_n2 - r_n1)
            deltas.append(delta)

        return np.mean(deltas)

    def __repr__(self) -> str:
        """String representation."""
        return f"LogisticMap(r={self.r:.4f}, regime={self.classify_regime()})"


def create_chaotic_logistic() -> LogisticMap:
    """
    Create logistic map in chaotic regime.

    Returns:
        LogisticMap with r=3.7 (chaotic)
    """
    return LogisticMap(r=3.7)


def create_period_doubling_series() -> list:
    """
    Create series of logistic maps showing period doubling.

    Returns:
        List of LogisticMap instances at key bifurcation points
    """
    r_values = [2.8, 3.2, 3.5, 3.56, 3.7, 4.0]
    return [LogisticMap(r=r) for r in r_values]


if __name__ == "__main__":
    # Quick test
    logistic = create_chaotic_logistic()
    print(f"Created {logistic}")

    # Generate trajectory
    traj = logistic.trajectory(x0=0.1, num_iterations=100)
    print(f"Generated trajectory: min={traj.min():.4f}, max={traj.max():.4f}")

    # Bifurcation diagram
    r_vals, x_vals = logistic.bifurcation_diagram(num_r=100)
    print(f"Bifurcation diagram: {len(r_vals)} points")

    # Period detection
    for r in [2.8, 3.2, 3.5, 3.7]:
        lm = LogisticMap(r=r)
        period = lm.find_period()
        print(f"r={r}: period={period}, regime={lm.classify_regime()}")
