"""
Bifurcation diagram demo for the Logistic Map.

This script generates the famous bifurcation diagram showing the
period-doubling route to chaos in the logistic map.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.systems.logistic_map import LogisticMap, create_chaotic_logistic


def plot_bifurcation_diagram():
    """
    Generate and plot the bifurcation diagram.
    """
    print("Generating bifurcation diagram...")
    print("This may take a moment...")

    logistic = LogisticMap()

    # Generate bifurcation data
    r_vals, x_vals = logistic.bifurcation_diagram(
        r_min=2.5,
        r_max=4.0,
        num_r=3000,
        num_iterations=1000,
        transient=200,
        x0=0.1
    )

    print(f"Generated {len(r_vals)} data points")

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot bifurcation diagram
    ax.plot(r_vals, x_vals, ',k', markersize=0.5, alpha=0.3)

    ax.set_xlabel('Growth Rate (r)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Population (x)', fontsize=14, fontweight='bold')
    ax.set_title('Logistic Map: Bifurcation Diagram\nPeriod-Doubling Route to Chaos',
                fontsize=16, fontweight='bold')

    # Add vertical lines for key bifurcation points
    ax.axvline(x=3.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Period-2 onset')
    ax.axvline(x=3.57, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Chaos onset')

    # Annotate regions
    ax.text(2.75, 0.9, 'Fixed\nPoint', fontsize=11, ha='center', color='green',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(3.25, 0.85, 'Period-2', fontsize=11, ha='center', color='blue',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(3.8, 0.9, 'Chaos', fontsize=11, ha='center', color='red',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_xlim(2.5, 4.0)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('outputs/bifurcation_diagram.png', dpi=200, bbox_inches='tight')
    print("Saved: outputs/bifurcation_diagram.png")

    plt.show()


def plot_cobweb_diagrams():
    """
    Plot cobweb diagrams for different r values.
    """
    print("\nGenerating cobweb diagrams...")

    r_values = [2.8, 3.3, 3.5, 3.7]
    titles = ['Fixed Point (r=2.8)', 'Period-2 (r=3.3)',
             'Period-4 (r=3.5)', 'Chaos (r=3.7)']

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, r, title in zip(axes, r_values, titles):
        logistic = LogisticMap(r=r)

        # Plot function
        x_range = np.linspace(0, 1, 500)
        y_func = r * x_range * (1 - x_range)

        ax.plot(x_range, y_func, 'b-', linewidth=2, label=f'f(x) = {r}x(1-x)')
        ax.plot([0, 1], [0, 1], 'y--', linewidth=2, label='y = x')

        # Generate and plot cobweb
        x_coords, y_coords = logistic.cobweb_plot_data(x0=0.1, num_iterations=50, r=r)
        ax.plot(x_coords, y_coords, 'r-', linewidth=1, alpha=0.7, label='Iteration path')

        ax.set_xlabel('$x_n$', fontsize=12)
        ax.set_ylabel('$x_{n+1}$', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('outputs/cobweb_diagrams.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/cobweb_diagrams.png")

    plt.show()


def plot_time_series():
    """
    Plot time series for different r values.
    """
    print("\nGenerating time series plots...")

    r_values = [2.8, 3.2, 3.5, 3.7]
    colors = ['green', 'blue', 'orange', 'red']
    labels = ['r=2.8 (Fixed)', 'r=3.2 (Period-2)', 'r=3.5 (Period-4)', 'r=3.7 (Chaos)']

    fig, ax = plt.subplots(figsize=(14, 6))

    for r, color, label in zip(r_values, colors, labels):
        logistic = LogisticMap(r=r)
        traj = logistic.trajectory(x0=0.1, num_iterations=100, transient=0, r=r)

        n_values = np.arange(len(traj))
        ax.plot(n_values, traj, color=color, linewidth=1.5, alpha=0.7, label=label)

    ax.set_xlabel('Iteration (n)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Population ($x_n$)', fontsize=13, fontweight='bold')
    ax.set_title('Logistic Map: Time Series for Different Growth Rates',
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig('outputs/logistic_timeseries.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/logistic_timeseries.png")

    plt.show()


def analyze_feigenbaum():
    """
    Analyze and display Feigenbaum constant.
    """
    print("\nAnalyzing Feigenbaum constant...")

    logistic = LogisticMap()

    # Approximate bifurcation points (can be calculated more precisely)
    bifurcation_points = [3.0, 3.449, 3.544, 3.5688, 3.5696]

    delta = logistic.feigenbaum_delta(bifurcation_points)

    print(f"\nFeigenbaum constant δ ≈ {delta:.4f}")
    print(f"Theoretical value: δ = 4.669201...")
    print(f"Error: {abs(delta - 4.669201) / 4.669201 * 100:.2f}%")

    # Show regime classification
    print("\nRegime classification:")
    test_r_values = [1.5, 2.5, 3.2, 3.5, 3.7, 4.0]
    for r in test_r_values:
        lm = LogisticMap(r=r)
        print(f"  r = {r}: {lm.classify_regime()}")


def main():
    """
    Run all bifurcation demos.
    """
    print("=" * 60)
    print("Logistic Map: Bifurcation Analysis")
    print("=" * 60)

    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # Run demos
    plot_bifurcation_diagram()
    plot_cobweb_diagrams()
    plot_time_series()
    analyze_feigenbaum()

    print("\n" + "=" * 60)
    print("All bifurcation demos completed!")
    print("Check the 'outputs' folder for saved images.")
    print("=" * 60)


if __name__ == "__main__":
    main()
