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
    Beautiful glowing butterfly pattern with gradient colors.
    """

    def construct(self):
        # Title
        title = Text("The Lorenz Attractor", font_size=48, weight=BOLD)
        title.to_edge(UP)

        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(title, shift=DOWN))
        self.wait()
        self.play(FadeOut(title))

        # Setup minimal 3D axes (cleaner look)
        axes = ThreeDAxes(
            x_range=[-25, 25, 25],
            y_range=[-35, 35, 35],
            z_range=[0, 50, 25],
            x_length=10,
            y_length=10,
            z_length=7,
            axis_config={
                "stroke_color": GREY_D,
                "stroke_width": 1,
                "include_tip": False,
            }
        )

        # Camera pulled far back to ensure entire butterfly fits in frame
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES, distance=20)
        self.add(axes)

        # Generate Lorenz trajectory with longer time span for full butterfly
        lorenz = create_classic_lorenz()
        initial_state = np.array([1.0, 1.0, 1.0])
        trajectory = lorenz.solve(initial_state, t_span=(0, 100), dt=0.01)

        # Subsample for smooth performance
        trajectory = trajectory[::3]

        # Convert trajectory to 3D points
        points = [axes.c2p(p[0], p[1], p[2]) for p in trajectory]

        # Create smooth curve with glowing effect
        # Main curve with gradient
        curve = VMobject()
        curve.set_points_smoothly([points[0]] + points + [points[-1]])

        # Apply gradient coloring from blue to cyan to emphasize butterfly wings
        curve.set_stroke(width=3)
        curve.set_color_by_gradient(BLUE_E, BLUE_C, TEAL_C, BLUE_C, BLUE_E)

        # Add glow effect by creating background copies
        glow_curve_1 = curve.copy()
        glow_curve_1.set_stroke(width=8, opacity=0.3)
        glow_curve_1.set_color_by_gradient(BLUE, TEAL, BLUE)

        glow_curve_2 = curve.copy()
        glow_curve_2.set_stroke(width=16, opacity=0.15)
        glow_curve_2.set_color_by_gradient(BLUE, TEAL, BLUE)

        # Animate drawing with glow
        self.play(
            Create(glow_curve_2),
            Create(glow_curve_1),
            Create(curve),
            run_time=15,
            rate_func=linear
        )
        self.wait()

        # Smooth camera rotation to showcase the butterfly from multiple angles
        self.move_camera(phi=65 * DEGREES, theta=-120 * DEGREES, run_time=5)
        self.wait(2)
        self.move_camera(phi=85 * DEGREES, theta=0 * DEGREES, run_time=5)
        self.wait(2)

        # Final slow rotation
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(8)
        self.stop_ambient_camera_rotation()

        self.wait(2)


class ButterflyEffectScene(ThreeDScene):
    """
    Demonstrate sensitive dependence on initial conditions.
    Two glowing trajectories with tiny initial difference diverge exponentially.
    """

    def construct(self):
        # Title
        title = Text("The Butterfly Effect", font_size=48, weight=BOLD)
        subtitle = Text("Tiny differences, enormous consequences", font_size=32, color=YELLOW)
        subtitle.next_to(title, DOWN)

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title, shift=DOWN), FadeIn(subtitle, shift=DOWN))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Setup minimal axes
        axes = ThreeDAxes(
            x_range=[-25, 25, 25],
            y_range=[-35, 35, 35],
            z_range=[0, 50, 25],
            x_length=10,
            y_length=10,
            z_length=7,
            axis_config={
                "stroke_color": GREY_D,
                "stroke_width": 1,
                "include_tip": False,
            }
        )

        # Camera pulled far back to see both complete trajectories
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES, distance=20)
        self.add(axes)

        # Generate two trajectories with tiny difference
        lorenz = create_classic_lorenz()
        initial1 = np.array([1.0, 1.0, 1.0])
        initial2 = np.array([1.0, 1.0, 1.00001])  # 0.00001 difference

        traj1 = lorenz.solve(initial1, t_span=(0, 40), dt=0.01)
        traj2 = lorenz.solve(initial2, t_span=(0, 40), dt=0.01)

        # Subsample
        traj1 = traj1[::3]
        traj2 = traj2[::3]

        # Convert to points
        points1 = [axes.c2p(p[0], p[1], p[2]) for p in traj1]
        points2 = [axes.c2p(p[0], p[1], p[2]) for p in traj2]

        # Create smooth curves with glow
        curve1 = VMobject()
        curve1.set_points_smoothly([points1[0]] + points1 + [points1[-1]])
        curve1.set_stroke(width=3, color=RED_C)

        glow1 = curve1.copy().set_stroke(width=10, opacity=0.4, color=RED)
        glow1_outer = curve1.copy().set_stroke(width=18, opacity=0.2, color=RED)

        curve2 = VMobject()
        curve2.set_points_smoothly([points2[0]] + points2 + [points2[-1]])
        curve2.set_stroke(width=3, color=BLUE_C)

        glow2 = curve2.copy().set_stroke(width=10, opacity=0.4, color=BLUE)
        glow2_outer = curve2.copy().set_stroke(width=18, opacity=0.2, color=BLUE)

        # Animate both trajectories simultaneously
        self.play(
            Create(glow1_outer),
            Create(glow1),
            Create(curve1),
            Create(glow2_outer),
            Create(glow2),
            Create(curve2),
            run_time=18,
            rate_func=linear
        )
        self.wait()

        # Smooth camera movements to show divergence
        self.move_camera(phi=65 * DEGREES, theta=-120 * DEGREES, run_time=5)
        self.wait(2)

        # Add text explaining divergence
        explanation = Text(
            "Initial difference: 0.00001",
            font_size=28,
            color=YELLOW,
            weight=BOLD
        )
        explanation.to_edge(DOWN).shift(UP * 0.5)
        self.add_fixed_in_frame_mobjects(explanation)
        self.play(FadeIn(explanation))
        self.wait(3)


class LorenzPhaseSpaceScene(ThreeDScene):
    """
    Show multiple glowing trajectories converging to the strange attractor.
    Beautiful multi-colored butterfly pattern.
    """

    def construct(self):
        title = Text("Strange Attractor", font_size=48, weight=BOLD)
        subtitle = Text("All paths lead to the butterfly", font_size=32, color=TEAL)
        subtitle.next_to(title, DOWN)

        self.add_fixed_in_frame_mobjects(title, subtitle)
        self.play(FadeIn(title, shift=DOWN), FadeIn(subtitle, shift=DOWN))
        self.wait()
        self.play(FadeOut(title), FadeOut(subtitle))

        # Setup minimal axes
        axes = ThreeDAxes(
            x_range=[-25, 25, 25],
            y_range=[-35, 35, 35],
            z_range=[0, 50, 25],
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
        lorenz = create_classic_lorenz()
        initial_states = lorenz.get_multiple_initial_states(num_states=4, spread=8.0)
        colors = [RED_C, BLUE_C, GREEN_C, PURPLE_C]

        all_curves = []
        for initial, color in zip(initial_states, colors):
            traj = lorenz.solve(initial, t_span=(0, 50), dt=0.01)
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
