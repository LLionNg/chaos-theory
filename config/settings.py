"""
Configuration settings for the Chaos Theory project.

This module contains global settings that can be adjusted based on
the user's environment and preferences.
"""

import os
from pathlib import Path
import platform


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
EXAMPLES_DIR = PROJECT_ROOT / 'examples'

# Ensure output directory exists
OUTPUTS_DIR.mkdir(exist_ok=True)

# System detection
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == 'Windows'
IS_LINUX = SYSTEM == 'Linux'
IS_WSL = 'microsoft' in platform.uname().release.lower() if IS_LINUX else False

# Rendering settings
DEFAULT_QUALITY = '1080p'  # Options: '480p', '720p', '1080p', '4k'
DEFAULT_FPS = 60

# Manim configuration
MANIM_QUALITY_MAP = {
    '480p': 'low_quality',
    '720p': 'medium_quality',
    '1080p': 'high_quality',
    '4k': 'production_quality'
}

# Numerical integration settings
DEFAULT_INTEGRATOR = 'rk4'  # Options: 'rk4', 'euler', 'adaptive'
DEFAULT_DT = 0.01  # Default time step

# Visualization settings
DPI = 150  # DPI for matplotlib figures
FIGURE_SIZE = (12, 9)  # Default figure size (width, height)

# Color schemes (can be customized)
DEFAULT_COLORMAP = 'plasma'
TRAJECTORY_COLOR = '#58C4DD'  # Blue
HIGHLIGHT_COLOR = '#FC6255'  # Red

# Performance settings
MAX_TRAJECTORY_POINTS = 50000  # Maximum points to render
SUBSAMPLE_RATE = 1  # Subsample trajectories by this factor (1 = no subsampling)

# System-specific settings
CHAOS_SYSTEMS_CONFIG = {
    'lorenz': {
        'default_params': {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3},
        'default_initial': [1.0, 1.0, 1.0],
        'default_t_span': (0, 50),
        'dt': 0.01,
    },
    'rossler': {
        'default_params': {'a': 0.2, 'b': 0.2, 'c': 5.7},
        'default_initial': [1.0, 1.0, 1.0],
        'default_t_span': (0, 100),
        'dt': 0.01,
    },
    'logistic_map': {
        'default_params': {'r': 3.7},
        'default_initial': 0.1,
        'default_iterations': 1000,
    },
    'double_pendulum': {
        'default_params': {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0},
        'default_initial': [2.09, 0, 2.09, 0],  # About 120 degrees
        'default_t_span': (0, 20),
        'dt': 0.02,
    },
}

# Logging
VERBOSE = True

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("CHAOS THEORY PROJECT CONFIGURATION")
    print("=" * 60)
    print(f"System: {SYSTEM}")
    if IS_WSL:
        print("  Running in WSL (Windows Subsystem for Linux)")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUTS_DIR}")
    print(f"Default quality: {DEFAULT_QUALITY}")
    print(f"Default FPS: {DEFAULT_FPS}")
    print(f"Default integrator: {DEFAULT_INTEGRATOR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
