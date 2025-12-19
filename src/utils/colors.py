"""
Color utilities for chaos theory visualizations.
Provides color schemes, gradients, and mapping functions.
"""

import numpy as np
from typing import List, Tuple
import colorsys


# Manim color compatibility
BLUE = "#58C4DD"
GREEN = "#83C167"
YELLOW = "#FFFF00"
RED = "#FC6255"
PURPLE = "#9A72AC"
ORANGE = "#FF862F"
PINK = "#F2055C"
TEAL = "#5FCBC4"
MAROON = "#CA3433"
GOLD = "#FDFD96"


def interpolate_color(color1: str, color2: str, alpha: float) -> str:
    """
    Interpolate between two hex colors.

    Args:
        color1: First color (hex format "#RRGGBB")
        color2: Second color (hex format "#RRGGBB")
        alpha: Interpolation factor (0 = color1, 1 = color2)

    Returns:
        Interpolated color as hex string
    """
    # Convert hex to RGB
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

    # Interpolate
    r = int(r1 + (r2 - r1) * alpha)
    g = int(g1 + (g2 - g1) * alpha)
    b = int(b1 + (b2 - b1) * alpha)

    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"


def color_gradient(colors: List[str], num_colors: int) -> List[str]:
    """
    Create a smooth gradient between multiple colors.

    Args:
        colors: List of hex color strings
        num_colors: Number of colors to generate in the gradient

    Returns:
        List of hex color strings forming a gradient
    """
    if len(colors) < 2:
        return colors * num_colors

    gradient = []
    segment_size = num_colors // (len(colors) - 1)

    for i in range(len(colors) - 1):
        for j in range(segment_size):
            alpha = j / segment_size
            color = interpolate_color(colors[i], colors[i + 1], alpha)
            gradient.append(color)

    # Add the last color
    gradient.append(colors[-1])

    # Trim or extend to exact size
    while len(gradient) < num_colors:
        gradient.append(colors[-1])

    return gradient[:num_colors]


def velocity_to_color(velocity: float, v_min: float, v_max: float, colormap: str = 'plasma') -> str:
    """
    Map velocity to color.

    Args:
        velocity: Current velocity magnitude
        v_min: Minimum velocity in dataset
        v_max: Maximum velocity in dataset
        colormap: Colormap name ('plasma', 'viridis', 'cool_warm', 'rainbow')

    Returns:
        Hex color string
    """
    # Normalize velocity to [0, 1]
    if v_max - v_min > 0:
        t = (velocity - v_min) / (v_max - v_min)
    else:
        t = 0.5

    t = np.clip(t, 0, 1)

    if colormap == 'plasma':
        colors = [PURPLE, PINK, ORANGE, YELLOW]
    elif colormap == 'viridis':
        colors = [PURPLE, BLUE, TEAL, GREEN, YELLOW]
    elif colormap == 'cool_warm':
        colors = [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED]
    elif colormap == 'rainbow':
        # Full HSV rainbow
        h = t
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    else:
        colors = [BLUE, GREEN, YELLOW, RED]

    # Find which segment of the gradient
    segment_idx = int(t * (len(colors) - 1))
    segment_idx = min(segment_idx, len(colors) - 2)

    # Interpolate within segment
    segment_t = (t * (len(colors) - 1)) - segment_idx
    return interpolate_color(colors[segment_idx], colors[segment_idx + 1], segment_t)


def time_to_color(time_index: int, total_points: int, colormap: str = 'cool_warm') -> str:
    """
    Map time index to color for trajectory visualization.

    Args:
        time_index: Current time index
        total_points: Total number of points
        colormap: Colormap name

    Returns:
        Hex color string
    """
    t = time_index / max(total_points - 1, 1)
    return velocity_to_color(t, 0, 1, colormap)


def chaos_colormap(name: str = 'default') -> List[str]:
    """
    Get predefined colormaps for chaos visualizations.

    Args:
        name: Colormap name

    Returns:
        List of hex color strings

    Available colormaps:
        - 'default': Blue to yellow gradient
        - 'fire': Black to red to yellow
        - 'ocean': Dark blue to cyan
        - 'forest': Dark green to light green
        - 'sunset': Purple to orange
        - 'neon': Bright, vibrant colors
    """
    colormaps = {
        'default': [BLUE, TEAL, GREEN, YELLOW],
        'fire': ["#000000", MAROON, RED, ORANGE, YELLOW],
        'ocean': ["#000080", BLUE, TEAL, "#ADD8E6"],
        'forest': ["#013220", GREEN, "#90EE90"],
        'sunset': [PURPLE, PINK, ORANGE, GOLD],
        'neon': [PURPLE, PINK, BLUE, TEAL, GREEN],
        'plasma': [PURPLE, PINK, ORANGE, YELLOW],
        'cool': [BLUE, TEAL, GREEN],
        'warm': [ORANGE, RED, MAROON],
    }

    return colormaps.get(name, colormaps['default'])


def get_trajectory_colors(
    trajectory: np.ndarray,
    mode: str = 'velocity',
    colormap: str = 'plasma'
) -> List[str]:
    """
    Generate colors for entire trajectory.

    Args:
        trajectory: Array of shape (n_points, n_dims) with trajectory points
        mode: Color mode ('velocity', 'time', 'height')
        colormap: Colormap name

    Returns:
        List of hex color strings, one per trajectory point
    """
    n_points = len(trajectory)

    if mode == 'velocity':
        # Calculate velocity magnitudes
        velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        velocities = np.concatenate([[velocities[0]], velocities])  # Pad to match length
        v_min, v_max = velocities.min(), velocities.max()

        return [velocity_to_color(v, v_min, v_max, colormap) for v in velocities]

    elif mode == 'time':
        # Color by time/position in trajectory
        return [time_to_color(i, n_points, colormap) for i in range(n_points)]

    elif mode == 'height':
        # Color by z-coordinate (height)
        if trajectory.shape[1] >= 3:
            z_values = trajectory[:, 2]
            z_min, z_max = z_values.min(), z_values.max()
            return [velocity_to_color(z, z_min, z_max, colormap) for z in z_values]
        else:
            return [time_to_color(i, n_points, colormap) for i in range(n_points)]

    else:
        # Default to time-based coloring
        return [time_to_color(i, n_points, colormap) for i in range(n_points)]


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """
    Convert RGB (0-1 range) to hex color.

    Args:
        r, g, b: RGB values in range [0, 1]

    Returns:
        Hex color string
    """
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex color to RGB (0-1 range).

    Args:
        hex_color: Hex color string "#RRGGBB"

    Returns:
        Tuple of (r, g, b) values in range [0, 1]
    """
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (r, g, b)
