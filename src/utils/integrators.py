"""
Numerical integration methods for solving differential equations.
Provides Runge-Kutta methods and Euler methods for chaos system simulations.
"""

import numpy as np
from typing import Callable, Tuple


def euler(derivative_func: Callable, state: np.ndarray, dt: float, *args) -> np.ndarray:
    """
    Euler method for numerical integration.

    Simple first-order integration method. Fast but less accurate.

    Args:
        derivative_func: Function that computes derivatives dy/dt = f(t, y)
        state: Current state vector [x, y, z, ...]
        dt: Time step
        *args: Additional arguments to pass to derivative_func

    Returns:
        New state after one time step
    """
    return state + dt * derivative_func(state, *args)


def rk4(derivative_func: Callable, state: np.ndarray, dt: float, *args) -> np.ndarray:
    """
    Fourth-order Runge-Kutta method (RK4).

    Classic RK4 integration - excellent balance of accuracy and performance.
    Standard method for chaos systems.

    Args:
        derivative_func: Function that computes derivatives dy/dt = f(t, y)
        state: Current state vector [x, y, z, ...]
        dt: Time step
        *args: Additional arguments to pass to derivative_func

    Returns:
        New state after one time step

    Mathematical formulation:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt*k1/2)
        k3 = f(t + dt/2, y + dt*k2/2)
        k4 = f(t + dt, y + dt*k3)
        y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = derivative_func(state, *args)
    k2 = derivative_func(state + 0.5 * dt * k1, *args)
    k3 = derivative_func(state + 0.5 * dt * k2, *args)
    k4 = derivative_func(state + dt * k3, *args)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_trajectory(
    derivative_func: Callable,
    initial_state: np.ndarray,
    t_span: Tuple[float, float],
    dt: float = 0.01,
    *args
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate full trajectory using RK4 integration.

    Args:
        derivative_func: Function that computes derivatives
        initial_state: Initial state vector
        t_span: Tuple of (t_start, t_end)
        dt: Time step (default: 0.01)
        *args: Additional arguments for derivative_func

    Returns:
        Tuple of (time_array, trajectory_array)
        - time_array: Array of time points
        - trajectory_array: Array of states at each time point
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt)

    # Initialize arrays
    times = np.linspace(t_start, t_end, num_steps)
    trajectory = np.zeros((num_steps, len(initial_state)))
    trajectory[0] = initial_state

    # Integration loop
    for i in range(1, num_steps):
        trajectory[i] = rk4(derivative_func, trajectory[i-1], dt, *args)

    return times, trajectory


def euler_trajectory(
    derivative_func: Callable,
    initial_state: np.ndarray,
    t_span: Tuple[float, float],
    dt: float = 0.01,
    *args
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate full trajectory using Euler integration.

    Args:
        derivative_func: Function that computes derivatives
        initial_state: Initial state vector
        t_span: Tuple of (t_start, t_end)
        dt: Time step (default: 0.01)
        *args: Additional arguments for derivative_func

    Returns:
        Tuple of (time_array, trajectory_array)
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt)

    times = np.linspace(t_start, t_end, num_steps)
    trajectory = np.zeros((num_steps, len(initial_state)))
    trajectory[0] = initial_state

    for i in range(1, num_steps):
        trajectory[i] = euler(derivative_func, trajectory[i-1], dt, *args)

    return times, trajectory


def adaptive_rk4(
    derivative_func: Callable,
    initial_state: np.ndarray,
    t_span: Tuple[float, float],
    dt_initial: float = 0.01,
    tolerance: float = 1e-6,
    *args
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive step-size RK4 integration.

    Adjusts time step based on local error estimate for better efficiency.

    Args:
        derivative_func: Function that computes derivatives
        initial_state: Initial state vector
        t_span: Tuple of (t_start, t_end)
        dt_initial: Initial time step
        tolerance: Error tolerance for adaptive stepping
        *args: Additional arguments for derivative_func

    Returns:
        Tuple of (time_array, trajectory_array)
    """
    t_start, t_end = t_span

    times = [t_start]
    trajectory = [initial_state.copy()]

    t = t_start
    state = initial_state.copy()
    dt = dt_initial

    while t < t_end:
        # Take one step with dt
        state_1 = rk4(derivative_func, state, dt, *args)

        # Take two steps with dt/2
        state_half = rk4(derivative_func, state, dt/2, *args)
        state_2 = rk4(derivative_func, state_half, dt/2, *args)

        # Estimate error
        error = np.linalg.norm(state_2 - state_1)

        if error < tolerance:
            # Accept step
            t += dt
            state = state_2
            times.append(t)
            trajectory.append(state.copy())

            # Increase step size if error is very small
            if error < tolerance / 10:
                dt = min(dt * 1.5, dt_initial * 10)
        else:
            # Reject step and reduce step size
            dt = dt * 0.5

        # Ensure we don't overshoot
        if t + dt > t_end:
            dt = t_end - t

    return np.array(times), np.array(trajectory)


# Convenience function
def solve_ode(
    derivative_func: Callable,
    initial_state: np.ndarray,
    t_span: Tuple[float, float],
    method: str = 'rk4',
    dt: float = 0.01,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified ODE solver interface.

    Args:
        derivative_func: Function that computes derivatives
        initial_state: Initial state vector
        t_span: Tuple of (t_start, t_end)
        method: Integration method ('rk4', 'euler', 'adaptive')
        dt: Time step
        **kwargs: Additional arguments

    Returns:
        Tuple of (time_array, trajectory_array)
    """
    if method == 'rk4':
        return rk4_trajectory(derivative_func, initial_state, t_span, dt)
    elif method == 'euler':
        return euler_trajectory(derivative_func, initial_state, t_span, dt)
    elif method == 'adaptive':
        tolerance = kwargs.get('tolerance', 1e-6)
        return adaptive_rk4(derivative_func, initial_state, t_span, dt, tolerance)
    else:
        raise ValueError(f"Unknown integration method: {method}")
