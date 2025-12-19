"""
Quick demo of the Lorenz attractor system.

This script demonstrates basic usage of the Lorenz system with matplotlib
visualization. Great for quick tests and understanding the system.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.systems.lorenz import LorenzSystem, create_classic_lorenz, demonstrate_butterfly_effect


def plot_lorenz_3d():
    """
    Plot Lorenz attractor in 3D.
    """
    print("Generating Lorenz attractor...")

    # Create system
    lorenz = create_classic_lorenz()

    # Generate trajectory
    initial_state = np.array([1.0, 1.0, 1.0])
    trajectory = lorenz.solve(initial_state, t_span=(0, 50), dt=0.01)

    print(f"Generated {len(trajectory)} points")

    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            linewidth=0.5, alpha=0.8, color='blue')

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Lorenz Attractor (σ=10, ρ=28, β=8/3)', fontsize=14, fontweight='bold')

    # Set viewing angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig('outputs/lorenz_3d.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/lorenz_3d.png")

    plt.show()


def plot_butterfly_effect():
    """
    Demonstrate the butterfly effect with two trajectories.
    """
    print("\nDemonstrating butterfly effect...")

    traj1, traj2 = demonstrate_butterfly_effect(perturbation=1e-8, t_span=(0, 40))

    # Calculate divergence
    divergence = np.linalg.norm(traj1 - traj2, axis=1)

    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6))

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(traj1[:, 0], traj1[:, 1], traj1[:, 2],
            linewidth=0.7, alpha=0.8, color='red', label='Trajectory 1')
    ax1.plot(traj2[:, 0], traj2[:, 1], traj2[:, 2],
            linewidth=0.7, alpha=0.8, color='blue', label='Trajectory 2')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Two Trajectories (initial diff: 1e-8)')
    ax1.legend()
    ax1.view_init(elev=25, azim=45)

    # Divergence plot
    ax2 = fig.add_subplot(122)
    time = np.linspace(0, 40, len(divergence))
    ax2.semilogy(time, divergence, color='purple', linewidth=2)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Distance between trajectories (log scale)', fontsize=12)
    ax2.set_title('Exponential Divergence (Butterfly Effect)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/butterfly_effect.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/butterfly_effect.png")

    plt.show()


def plot_projections():
    """
    Plot 2D projections of the Lorenz attractor.
    """
    print("\nGenerating 2D projections...")

    lorenz = create_classic_lorenz()
    trajectory = lorenz.solve(np.array([1, 1, 1]), t_span=(0, 50), dt=0.01)

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # XY projection
    axes[0].plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.5, alpha=0.7, color='blue')
    axes[0].set_xlabel('X', fontsize=11)
    axes[0].set_ylabel('Y', fontsize=11)
    axes[0].set_title('XY Projection', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # XZ projection
    axes[1].plot(trajectory[:, 0], trajectory[:, 2], linewidth=0.5, alpha=0.7, color='green')
    axes[1].set_xlabel('X', fontsize=11)
    axes[1].set_ylabel('Z', fontsize=11)
    axes[1].set_title('XZ Projection', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # YZ projection
    axes[2].plot(trajectory[:, 1], trajectory[:, 2], linewidth=0.5, alpha=0.7, color='red')
    axes[2].set_xlabel('Y', fontsize=11)
    axes[2].set_ylabel('Z', fontsize=11)
    axes[2].set_title('YZ Projection', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/lorenz_projections.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/lorenz_projections.png")

    plt.show()


def main():
    """
    Run all demos.
    """
    print("=" * 60)
    print("Lorenz Attractor Demo")
    print("=" * 60)

    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)

    # Run demos
    plot_lorenz_3d()
    plot_butterfly_effect()
    plot_projections()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("Check the 'outputs' folder for saved images.")
    print("=" * 60)


if __name__ == "__main__":
    main()
