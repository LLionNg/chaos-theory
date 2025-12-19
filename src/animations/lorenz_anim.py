"""
Manim animations for the Lorenz Attractor.

This module contains high-quality educational animations demonstrating:
- The butterfly attractor formation
- Sensitive dependence on initial conditions
- Parameter variation effects
- 3D trajectory visualization with rotating camera
"""

from manim import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.systems.lorenz import LorenzSystem, create_classic_lorenz
from src.utils.colors import get_trajectory_colors, BLUE, GREEN, YELLOW, RED, PURPLE, TEAL


class LorenzAttractorScene(ThreeDScene):
    """
    Main Lorenz attractor visualization with rotating 3D view.
    """

    def construct(self):
        # Title
        title = Text("The Lorenz Attractor", font_size=48)
        title.to_edge(UP)
        subtitle = Text("The Butterfly Effect", font_size=32, color=BLUE)
        subtitle.next_to(title, DOWN)

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(Write(title), Write(subtitle))
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle))

        # Setup 3D axes
        axes = ThreeDAxes(
            x_range=[-25, 25, 5],
            y_range=[-35, 35, 5],
            z_range=[0, 50, 10],
            x_length=8,
            y_length=8,
            z_length=6,
        )

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes)

        # Generate Lorenz trajectory
        lorenz = create_classic_lorenz()
        initial_state = np.array([1.0, 1.0, 1.0])
        trajectory = lorenz.solve(initial_state, t_span=(0, 50), dt=0.005)

        # Subsample for performance
        trajectory = trajectory[::2]

        # Create parametric curve
        def lorenz_curve(t):
            idx = int(t * (len(trajectory) - 1))
            idx = np.clip(idx, 0, len(trajectory) - 1)
            point = trajectory[idx]
            return axes.c2p(point[0], point[1], point[2])

        # Draw trajectory
        curve = ParametricFunction(
            lorenz_curve,
            t_range=[0, 1],
            color=BLUE,
            stroke_width=2,
        )

        # Animate drawing
        self.play(Create(curve), run_time=10, rate_func=linear)
        self.wait()

        # Rotate camera around the attractor
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(10)
        self.stop_ambient_camera_rotation()

        self.wait(2)


class ButterflyEffectScene(ThreeDScene):
    """
    Demonstrate sensitive dependence on initial conditions.
    Two trajectories with tiny initial difference diverge exponentially.
    """

    def construct(self):
        # Title
        title = Text("The Butterfly Effect", font_size=48)
        subtitle = Text("Sensitive Dependence on Initial Conditions", font_size=28)
        subtitle.next_to(title, DOWN)

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(Write(title), Write(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Setup axes
        axes = ThreeDAxes(
            x_range=[-25, 25, 10],
            y_range=[-35, 35, 10],
            z_range=[0, 50, 10],
            x_length=8,
            y_length=8,
            z_length=6,
        )

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes)

        # Generate two trajectories with tiny difference
        lorenz = create_classic_lorenz()
        initial1 = np.array([1.0, 1.0, 1.0])
        initial2 = np.array([1.0, 1.0, 1.00001])  # Tiny difference

        traj1 = lorenz.solve(initial1, t_span=(0, 40), dt=0.005)
        traj2 = lorenz.solve(initial2, t_span=(0, 40), dt=0.005)

        # Subsample
        traj1 = traj1[::2]
        traj2 = traj2[::2]

        # Create curves
        def curve1_func(t):
            idx = int(t * (len(traj1) - 1))
            idx = np.clip(idx, 0, len(traj1) - 1)
            return axes.c2p(traj1[idx, 0], traj1[idx, 1], traj1[idx, 2])

        def curve2_func(t):
            idx = int(t * (len(traj2) - 1))
            idx = np.clip(idx, 0, len(traj2) - 1)
            return axes.c2p(traj2[idx, 0], traj2[idx, 1], traj2[idx, 2])

        curve1 = ParametricFunction(
            curve1_func,
            t_range=[0, 1],
            color=RED,
            stroke_width=3,
        )

        curve2 = ParametricFunction(
            curve2_func,
            t_range=[0, 1],
            color=BLUE,
            stroke_width=3,
        )

        # Animate both trajectories
        self.play(
            Create(curve1),
            Create(curve2),
            run_time=15,
            rate_func=linear
        )

        self.wait()

        # Rotate camera
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(8)
        self.stop_ambient_camera_rotation()

        # Add text explaining divergence
        explanation = Text(
            "Initial difference: 0.00001\nDivergence is exponential!",
            font_size=24,
            color=YELLOW
        )
        explanation.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(explanation)
        self.play(FadeIn(explanation))
        self.wait(3)


class LorenzPhaseSpaceScene(ThreeDScene):
    """
    Show multiple trajectories converging to the attractor.
    """

    def construct(self):
        title = Text("Strange Attractor", font_size=48)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))

        # Setup axes
        axes = ThreeDAxes(
            x_range=[-25, 25, 10],
            y_range=[-35, 35, 10],
            z_range=[0, 50, 10],
            x_length=8,
            y_length=8,
            z_length=6,
        )

        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        self.add(axes)

        # Generate multiple trajectories from different initial conditions
        lorenz = create_classic_lorenz()
        initial_states = lorenz.get_multiple_initial_states(num_states=5, spread=5.0)
        colors = [RED, BLUE, GREEN, YELLOW, PURPLE]

        curves = []
        for initial, color in zip(initial_states, colors):
            traj = lorenz.solve(initial, t_span=(0, 30), dt=0.005)
            traj = traj[::3]  # Subsample

            def make_curve_func(trajectory):
                def curve_func(t):
                    idx = int(t * (len(trajectory) - 1))
                    idx = np.clip(idx, 0, len(trajectory) - 1)
                    return axes.c2p(trajectory[idx, 0], trajectory[idx, 1], trajectory[idx, 2])
                return curve_func

            curve = ParametricFunction(
                make_curve_func(traj),
                t_range=[0, 1],
                color=color,
                stroke_width=2,
            )
            curves.append(curve)

        # Draw all curves simultaneously
        self.play(*[Create(curve) for curve in curves], run_time=12, rate_func=linear)
        self.wait()

        # Rotate
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(10)
        self.stop_ambient_camera_rotation()
        self.wait(2)


class LorenzParametersScene(Scene):
    """
    Show how parameters affect the system (2D visualization).
    """

    def construct(self):
        title = Text("Lorenz System Parameters", font_size=42)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.to_edge(UP))

        # Show equations
        equations = MathTex(
            r"\frac{dx}{dt} &= \sigma(y - x) \\",
            r"\frac{dy}{dt} &= x(\rho - z) - y \\",
            r"\frac{dz}{dt} &= xy - \beta z",
            font_size=36
        )
        self.play(Write(equations))
        self.wait(2)

        # Show classic parameters
        params = MathTex(
            r"\sigma = 10, \quad \rho = 28, \quad \beta = \frac{8}{3}",
            font_size=36,
            color=YELLOW
        )
        params.next_to(equations, DOWN, buff=0.5)
        self.play(Write(params))
        self.wait(3)

        # Explanation
        explanation = Text(
            "These classic parameters produce chaotic behavior",
            font_size=28,
            color=BLUE
        )
        explanation.next_to(params, DOWN, buff=0.5)
        self.play(FadeIn(explanation))
        self.wait(3)


class LorenzIntroScene(Scene):
    """
    Introduction scene with title and context.
    """

    def construct(self):
        # Main title
        title = Text("The Lorenz Attractor", font_size=60, weight=BOLD)
        self.play(Write(title))
        self.wait()

        # Subtitle
        subtitle = Text(
            "A Journey into Chaos Theory",
            font_size=36,
            color=BLUE
        )
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(subtitle))
        self.wait(2)

        # Move up
        self.play(
            title.animate.scale(0.7).to_edge(UP),
            FadeOut(subtitle)
        )

        # Historical context
        context = VGroup(
            Text("Discovered by Edward Lorenz in 1963", font_size=28),
            Text("Simplified model of atmospheric convection", font_size=28),
            Text("Demonstrates the 'Butterfly Effect'", font_size=28),
            Text("Sensitive dependence on initial conditions", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        context.move_to(ORIGIN)

        for line in context:
            self.play(FadeIn(line, shift=RIGHT))
            self.wait(0.8)

        self.wait(2)
        self.play(FadeOut(context), FadeOut(title))


# Utility function to render all scenes
def render_all_lorenz_scenes():
    """
    Convenience function to render all Lorenz scenes.
    Use from command line or programmatically.
    """
    scenes = [
        LorenzIntroScene,
        LorenzParametersScene,
        LorenzAttractorScene,
        ButterflyEffectScene,
        LorenzPhaseSpaceScene,
    ]
    return scenes


if __name__ == "__main__":
    # This allows testing individual scenes
    print("Lorenz animation scenes ready!")
    print("Render with: manim -pqh lorenz_anim.py LorenzAttractorScene")
