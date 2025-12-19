"""
Manim animations for the Rössler Attractor.

This module contains animations demonstrating:
- The scroll-like shape of the Rössler attractor
- 3D trajectory visualization
- Parameter variation effects
- Poincaré sections
"""

from manim import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.systems.rossler import RosslerSystem, create_classic_rossler
from src.utils.colors import BLUE, GREEN, YELLOW, RED, PURPLE, TEAL


class RosslerAttractorScene(ThreeDScene):
    """
    Main Rössler attractor visualization with rotating 3D view.
    """

    def construct(self):
        # Title
        title = Text("The Rössler Attractor", font_size=48)
        subtitle = Text("Chaotic Scroll Dynamics", font_size=32, color=PURPLE)
        subtitle.next_to(title, DOWN)

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(Write(title), Write(subtitle))
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle))

        # Setup 3D axes
        axes = ThreeDAxes(
            x_range=[-15, 15, 5],
            y_range=[-20, 15, 5],
            z_range=[0, 30, 5],
            x_length=7,
            y_length=7,
            z_length=6,
        )

        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        z_label = axes.get_z_axis_label("z")

        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)
        self.add(axes, x_label, y_label, z_label)

        # Generate Rössler trajectory
        rossler = create_classic_rossler()
        initial_state = rossler.get_default_initial_state()
        trajectory = rossler.solve(initial_state, t_span=(0, 100), dt=0.01)

        # Subsample for performance
        trajectory = trajectory[::3]

        # Create parametric curve
        def rossler_curve(t):
            idx = int(t * (len(trajectory) - 1))
            idx = np.clip(idx, 0, len(trajectory) - 1)
            point = trajectory[idx]
            return axes.c2p(point[0], point[1], point[2])

        # Draw trajectory
        curve = ParametricFunction(
            rossler_curve,
            t_range=[0, 1],
            color=PURPLE,
            stroke_width=2,
        )

        # Animate drawing
        self.play(Create(curve), run_time=12, rate_func=linear)
        self.wait()

        # Rotate camera around the attractor
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(12)
        self.stop_ambient_camera_rotation()

        self.wait(2)


class RosslerScrollScene(ThreeDScene):
    """
    Show the scroll-like structure of the Rössler attractor.
    """

    def construct(self):
        title = Text("Scroll Structure", font_size=44)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.to_edge(UP).scale(0.7))

        # Setup axes
        axes = ThreeDAxes(
            x_range=[-15, 15, 10],
            y_range=[-20, 15, 10],
            z_range=[0, 30, 10],
            x_length=7,
            y_length=7,
            z_length=6,
        )

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.add(axes)

        # Generate trajectory
        rossler = create_classic_rossler()
        trajectory = rossler.solve(
            rossler.get_default_initial_state(),
            t_span=(0, 150),
            dt=0.01
        )
        trajectory = trajectory[::4]

        # Create curve
        def curve_func(t):
            idx = int(t * (len(trajectory) - 1))
            idx = np.clip(idx, 0, len(trajectory) - 1)
            return axes.c2p(trajectory[idx, 0], trajectory[idx, 1], trajectory[idx, 2])

        curve = ParametricFunction(
            curve_func,
            t_range=[0, 1],
            color=TEAL,
            stroke_width=2,
        )

        self.play(Create(curve), run_time=15, rate_func=linear)
        self.wait()

        # Slowly rotate to show scroll structure
        self.move_camera(phi=60 * DEGREES, theta=0 * DEGREES, run_time=4)
        self.wait(2)
        self.move_camera(phi=80 * DEGREES, theta=90 * DEGREES, run_time=4)
        self.wait(2)


class RosslerMultipleTrajectories(ThreeDScene):
    """
    Show multiple trajectories converging to the attractor.
    """

    def construct(self):
        title = Text("Convergence to Attractor", font_size=42)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))

        # Setup axes
        axes = ThreeDAxes(
            x_range=[-15, 15, 10],
            y_range=[-20, 15, 10],
            z_range=[0, 30, 10],
            x_length=7,
            y_length=7,
            z_length=6,
        )

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes)

        # Generate multiple trajectories
        rossler = create_classic_rossler()
        initial_states = rossler.get_multiple_initial_states(num_states=5, spread=3.0)
        colors = [RED, BLUE, GREEN, YELLOW, PURPLE]

        curves = []
        for initial, color in zip(initial_states, colors):
            traj = rossler.solve(initial, t_span=(0, 80), dt=0.01)
            traj = traj[::4]

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

        # Draw all curves
        self.play(*[Create(curve) for curve in curves], run_time=12, rate_func=linear)
        self.wait()

        # Rotate
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(10)
        self.stop_ambient_camera_rotation()
        self.wait(2)


class RosslerParameterScene(Scene):
    """
    Show how parameter c affects the system.
    """

    def construct(self):
        title = Text("Parameter Variation Effect", font_size=44)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.to_edge(UP))

        # Show equations
        equations = MathTex(
            r"\frac{dx}{dt} &= -y - z \\",
            r"\frac{dy}{dt} &= x + ay \\",
            r"\frac{dz}{dt} &= b + z(x - c)",
            font_size=36
        )
        self.play(Write(equations))
        self.wait(2)

        # Classic parameters
        params = MathTex(
            r"a = 0.2, \quad b = 0.2, \quad c = 5.7",
            font_size=36,
            color=PURPLE
        )
        params.next_to(equations, DOWN, buff=0.5)
        self.play(Write(params))
        self.wait(2)

        # Explanation
        explanation = VGroup(
            Text("c < 2: Fixed point", font_size=26),
            Text("c ≈ 2-4: Limit cycle", font_size=26),
            Text("c > 4: Chaotic attractor", font_size=26, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        explanation.next_to(params, DOWN, buff=0.6)

        for line in explanation:
            self.play(FadeIn(line, shift=RIGHT))
            self.wait(0.7)

        self.wait(3)


class RosslerIntroScene(Scene):
    """
    Introduction to the Rössler attractor.
    """

    def construct(self):
        # Title
        title = Text("The Rössler Attractor", font_size=56, weight=BOLD)
        self.play(Write(title))
        self.wait()

        subtitle = Text("Elegant Chaotic Dynamics", font_size=36, color=PURPLE)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(subtitle))
        self.wait(2)

        self.play(FadeOut(subtitle), title.animate.scale(0.7).to_edge(UP))

        # Historical context
        context = VGroup(
            Text("Discovered by Otto Rössler in 1976", font_size=28),
            Text("Simpler than Lorenz system", font_size=28),
            Text("Exhibits scroll-like strange attractor", font_size=28),
            Text("Chaotic behavior with continuous spectrum", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        context.move_to(ORIGIN)

        for line in context:
            self.play(FadeIn(line, shift=RIGHT))
            self.wait(0.8)

        self.wait(3)
        self.play(FadeOut(context), FadeOut(title))


# Utility function
def render_all_rossler_scenes():
    """
    List of all Rössler attractor scenes.
    """
    return [
        RosslerIntroScene,
        RosslerParameterScene,
        RosslerAttractorScene,
        RosslerScrollScene,
        RosslerMultipleTrajectories,
    ]


if __name__ == "__main__":
    print("Rössler Attractor animation scenes ready!")
    print("Render with: manim -pqh rossler_anim.py RosslerAttractorScene")
