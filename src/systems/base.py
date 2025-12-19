"""
Base class for chaos systems.
Provides common interface for all dynamical systems.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.integrators import solve_ode


class ChaosSystem(ABC):
    """
    Abstract base class for all chaos systems.

    Subclasses must implement the derivative() method that defines
    the system's differential equations.
    """

    def __init__(self, name: str = "Chaos System"):
        """
        Initialize chaos system.

        Args:
            name: Name of the system
        """
        self.name = name

    @abstractmethod
    def derivative(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """
        Compute the derivative of the system state.

        This method defines the differential equations:
        dx/dt = f(x, t)

        Args:
            state: Current state vector [x, y, z, ...]
            t: Current time (optional, for time-dependent systems)

        Returns:
            Derivative vector [dx/dt, dy/dt, dz/dt, ...]
        """
        pass

    def solve(
        self,
        initial_state: np.ndarray,
        t_span: Tuple[float, float] = (0, 50),
        dt: float = 0.01,
        method: str = 'rk4'
    ) -> np.ndarray:
        """
        Solve the system and generate trajectory.

        Args:
            initial_state: Initial state vector
            t_span: Time span (t_start, t_end)
            dt: Time step
            method: Integration method ('rk4', 'euler', 'adaptive')

        Returns:
            Trajectory array of shape (n_points, n_dims)
        """
        times, trajectory = solve_ode(
            self.derivative,
            initial_state,
            t_span,
            method=method,
            dt=dt
        )
        return trajectory

    def solve_with_time(
        self,
        initial_state: np.ndarray,
        t_span: Tuple[float, float] = (0, 50),
        dt: float = 0.01,
        method: str = 'rk4'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the system and return both time and trajectory.

        Args:
            initial_state: Initial state vector
            t_span: Time span (t_start, t_end)
            dt: Time step
            method: Integration method

        Returns:
            Tuple of (time_array, trajectory_array)
        """
        return solve_ode(
            self.derivative,
            initial_state,
            t_span,
            method=method,
            dt=dt
        )

    def get_trajectory(
        self,
        initial_state: np.ndarray,
        num_points: int = 5000,
        transient: int = 0,
        dt: float = 0.01,
        method: str = 'rk4'
    ) -> np.ndarray:
        """
        Generate trajectory with transient removal.

        Args:
            initial_state: Initial state vector
            num_points: Number of points to return (after transient)
            transient: Number of initial points to discard
            dt: Time step
            method: Integration method

        Returns:
            Trajectory array without transient behavior
        """
        total_time = (num_points + transient) * dt
        trajectory = self.solve(
            initial_state,
            t_span=(0, total_time),
            dt=dt,
            method=method
        )

        # Remove transient
        if transient > 0:
            return trajectory[transient:]
        return trajectory

    def lyapunov_exponent(
        self,
        initial_state: np.ndarray,
        t_span: Tuple[float, float] = (0, 1000),
        dt: float = 0.01,
        epsilon: float = 1e-8
    ) -> float:
        """
        Estimate largest Lyapunov exponent.

        Positive Lyapunov exponent indicates chaos.

        Args:
            initial_state: Initial state vector
            t_span: Time span for estimation
            dt: Time step
            epsilon: Small perturbation size

        Returns:
            Estimated largest Lyapunov exponent
        """
        # Create nearby initial state
        perturbed_state = initial_state + epsilon * np.random.randn(len(initial_state))

        # Solve both trajectories
        _, traj1 = self.solve_with_time(initial_state, t_span, dt)
        _, traj2 = self.solve_with_time(perturbed_state, t_span, dt)

        # Calculate divergence
        divergence = np.linalg.norm(traj1 - traj2, axis=1)

        # Remove zeros to avoid log issues
        divergence = divergence[divergence > 0]

        if len(divergence) == 0:
            return 0.0

        # Lyapunov exponent
        t_total = t_span[1] - t_span[0]
        lyap = np.log(divergence[-1] / epsilon) / t_total

        return lyap

    def __repr__(self) -> str:
        """String representation of the system."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class DiscreteSystem(ABC):
    """
    Abstract base class for discrete-time dynamical systems (maps).

    Examples: Logistic map, Henon map, etc.
    """

    def __init__(self, name: str = "Discrete System"):
        """
        Initialize discrete system.

        Args:
            name: Name of the system
        """
        self.name = name

    @abstractmethod
    def iterate(self, state: np.ndarray, *params) -> np.ndarray:
        """
        Perform one iteration of the map.

        Args:
            state: Current state
            *params: System parameters

        Returns:
            Next state
        """
        pass

    def trajectory(
        self,
        initial_state: np.ndarray,
        num_iterations: int = 1000,
        transient: int = 0,
        *params
    ) -> np.ndarray:
        """
        Generate trajectory by iterating the map.

        Args:
            initial_state: Initial state
            num_iterations: Number of iterations
            transient: Number of initial iterations to discard
            *params: System parameters

        Returns:
            Trajectory array
        """
        total_iterations = num_iterations + transient
        dim = len(initial_state) if isinstance(initial_state, np.ndarray) else 1

        # Handle scalar case
        if dim == 1:
            trajectory = np.zeros(total_iterations)
            trajectory[0] = initial_state
            for i in range(1, total_iterations):
                trajectory[i] = self.iterate(trajectory[i-1], *params)
        else:
            trajectory = np.zeros((total_iterations, dim))
            trajectory[0] = initial_state
            for i in range(1, total_iterations):
                trajectory[i] = self.iterate(trajectory[i-1], *params)

        # Remove transient
        if transient > 0:
            return trajectory[transient:]
        return trajectory

    def __repr__(self) -> str:
        """String representation of the system."""
        return f"{self.__class__.__name__}(name='{self.name}')"
