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
    Beautiful glowing scroll pattern with gradient colors.
    """

    def construct(self):
        # Title
        title = Text("The Rössler Attractor", font_size=48, weight=BOLD)
        subtitle = Text("Chaotic Scroll Dynamics", font_size=32, color=PURPLE)
        subtitle.next_to(title, DOWN)

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title, shift=DOWN), FadeIn(subtitle, shift=DOWN))
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle))

        # Setup minimal 3D axes (cleaner look)
        axes = ThreeDAxes(
            x_range=[-15, 15, 15],
            y_range=[-20, 15, 15],
            z_range=[0, 30, 15],
            x_length=10,
            y_length=10,
            z_length=7,
            axis_config={
                "stroke_color": GREY_D,
                "stroke_width": 1,
                "include_tip": False,
            }
        )

        # Camera pulled far back to see complete scroll attractor
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES, distance=20)
        self.add(axes)

        # Generate Rössler trajectory with longer time for full scroll
        rossler = create_classic_rossler()
        initial_state = rossler.get_default_initial_state()
        trajectory = rossler.solve(initial_state, t_span=(0, 150), dt=0.01)

        # Subsample for performance
        trajectory = trajectory[::3]

        # Convert trajectory to 3D points
        points = [axes.c2p(p[0], p[1], p[2]) for p in trajectory]

        # Create smooth curve with glowing effect
        curve = VMobject()
        curve.set_points_smoothly([points[0]] + points + [points[-1]])
        curve.set_stroke(width=3)
        curve.set_color_by_gradient(PURPLE_E, PURPLE_C, PINK, PURPLE_C, PURPLE_E)

        # Add glow effect by creating background copies
        glow_curve_1 = curve.copy()
        glow_curve_1.set_stroke(width=8, opacity=0.3)
        glow_curve_1.set_color_by_gradient(PURPLE, PINK, PURPLE)

        glow_curve_2 = curve.copy()
        glow_curve_2.set_stroke(width=16, opacity=0.15)
        glow_curve_2.set_color_by_gradient(PURPLE, PINK, PURPLE)

        # Animate drawing with glow
        self.play(
            Create(glow_curve_2),
            Create(glow_curve_1),
            Create(curve),
            run_time=15,
            rate_func=linear
        )
        self.wait()

        # Smooth camera rotation to showcase the scroll from multiple angles
        self.move_camera(phi=65 * DEGREES, theta=-120 * DEGREES, run_time=5)
        self.wait(2)
        self.move_camera(phi=85 * DEGREES, theta=0 * DEGREES, run_time=5)
        self.wait(2)

        # Final slow rotation
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(8)
        self.stop_ambient_camera_rotation()

        self.wait(2)


class RosslerScrollScene(ThreeDScene):
    """
    Show the scroll-like structure of the Rössler attractor with glowing effect.
    """

    def construct(self):
        title = Text("Scroll Structure", font_size=44, weight=BOLD)
        subtitle = Text("Elegant spiral dynamics", font_size=28, color=TEAL)
        subtitle.next_to(title, DOWN)

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title, shift=DOWN), FadeIn(subtitle, shift=DOWN))
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle))

        # Setup minimal axes
        axes = ThreeDAxes(
            x_range=[-15, 15, 15],
            y_range=[-20, 15, 15],
            z_range=[0, 30, 15],
            x_length=10,
            y_length=10,
            z_length=7,
            axis_config={
                "stroke_color": GREY_D,
                "stroke_width": 1,
                "include_tip": False,
            }
        )

        # Camera pulled far back for full scroll view
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES, distance=20)
        self.add(axes)

        # Generate trajectory
        rossler = create_classic_rossler()
        trajectory = rossler.solve(
            rossler.get_default_initial_state(),
            t_span=(0, 200),
            dt=0.01
        )
        trajectory = trajectory[::3]

        # Convert to points
        points = [axes.c2p(p[0], p[1], p[2]) for p in trajectory]

        # Create smooth curve with glow
        curve = VMobject()
        curve.set_points_smoothly([points[0]] + points + [points[-1]])
        curve.set_stroke(width=3)
        curve.set_color_by_gradient(TEAL_E, TEAL_C, BLUE_C, TEAL_C, TEAL_E)

        # Add glow layers
        glow_curve_1 = curve.copy()
        glow_curve_1.set_stroke(width=8, opacity=0.3)
        glow_curve_1.set_color_by_gradient(TEAL, BLUE, TEAL)

        glow_curve_2 = curve.copy()
        glow_curve_2.set_stroke(width=16, opacity=0.15)
        glow_curve_2.set_color_by_gradient(TEAL, BLUE, TEAL)

        # Animate with glow
        self.play(
            Create(glow_curve_2),
            Create(glow_curve_1),
            Create(curve),
            run_time=18,
            rate_func=linear
        )
        self.wait()

        # Smooth camera movements to show scroll structure
        self.move_camera(phi=65 * DEGREES, theta=-120 * DEGREES, run_time=5)
        self.wait(2)
        self.move_camera(phi=85 * DEGREES, theta=15 * DEGREES, run_time=5)
        self.wait(2)


class RosslerMultipleTrajectories(ThreeDScene):
    """
    Show multiple glowing trajectories converging to the strange attractor.
    Beautiful multi-colored scroll pattern.
    """

    def construct(self):
        title = Text("Strange Attractor", font_size=48, weight=BOLD)
        subtitle = Text("All paths lead to the scroll", font_size=32, color=PURPLE)
        subtitle.next_to(title, DOWN)

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title, shift=DOWN), FadeIn(subtitle, shift=DOWN))
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle))

        # Setup minimal axes
        axes = ThreeDAxes(
            x_range=[-15, 15, 15],
            y_range=[-20, 15, 15],
            z_range=[0, 30, 15],
            x_length=10,
            y_length=10,
            z_length=7,
            axis_config={
                "stroke_color": GREY_D,
                "stroke_width": 1,
                "include_tip": False,
            }
        )

        # Camera pulled far back for multiple trajectories
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES, distance=20)
        self.add(axes)

        # Generate multiple trajectories from different initial conditions
        rossler = create_classic_rossler()
        initial_states = rossler.get_multiple_initial_states(num_states=4, spread=3.0)
        colors = [RED_C, BLUE_C, GREEN_C, YELLOW_C]

        all_curves = []
        for initial, color in zip(initial_states, colors):
            traj = rossler.solve(initial, t_span=(0, 100), dt=0.01)
            traj = traj[::3]  # Subsample

            # Convert to points
            points = [axes.c2p(p[0], p[1], p[2]) for p in traj]

            # Create smooth curve with glow
            curve = VMobject()
            curve.set_points_smoothly([points[0]] + points + [points[-1]])
            curve.set_stroke(width=2.5, color=color)

            # Add glow layers
            glow = curve.copy().set_stroke(width=8, opacity=0.3, color=color)
            glow_outer = curve.copy().set_stroke(width=14, opacity=0.15, color=color)

            all_curves.extend([glow_outer, glow, curve])

        # Draw all curves simultaneously with beautiful effect
        self.play(*[Create(curve) for curve in all_curves], run_time=16, rate_func=linear)
        self.wait()

        # Smooth camera movements to showcase all trajectories
        self.move_camera(phi=65 * DEGREES, theta=-120 * DEGREES, run_time=5)
        self.wait(2)
        self.move_camera(phi=85 * DEGREES, theta=15 * DEGREES, run_time=5)
        self.wait(2)

        # Final rotation
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(8)
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
    Introduction to the Rössler attractor with smooth animations.
    """

    def construct(self):
        # Title
        title = Text("The Rössler Attractor", font_size=60, weight=BOLD)
        self.play(Write(title))
        self.wait()

        subtitle = Text("Elegant Chaotic Dynamics", font_size=36, color=PURPLE)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(subtitle, shift=DOWN))
        self.wait(2)

        # Move up smoothly
        self.play(
            title.animate.scale(0.7).to_edge(UP),
            FadeOut(subtitle)
        )

        # Historical context with smooth animations
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

        self.wait(2)
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
