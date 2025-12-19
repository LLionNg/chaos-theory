"""
Manim animations for the Double Pendulum.

This module contains animations demonstrating:
- Chaotic motion of the double pendulum
- Sensitivity to initial conditions
- Energy conservation
- Beautiful trajectory patterns
"""

from manim import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.systems.double_pendulum import DoublePendulum, create_standard_double_pendulum
from src.utils.colors import BLUE, GREEN, YELLOW, RED, PURPLE, TEAL, ORANGE


class DoublePendulumScene(Scene):
    """
    Animate the double pendulum with glowing trail showing chaotic motion.
    """

    def construct(self):
        # Title with smooth entrance
        title = Text("Double Pendulum Chaos", font_size=48, weight=BOLD)
        subtitle = Text("Deterministic yet unpredictable", font_size=28, color=ORANGE)
        subtitle.next_to(title, DOWN)

        self.play(FadeIn(title, shift=DOWN), FadeIn(subtitle, shift=DOWN))
        self.wait()
        self.play(FadeOut(subtitle), title.animate.scale(0.7).to_edge(UP))

        # Create pendulum
        pendulum = create_standard_double_pendulum()
        initial = pendulum.get_default_initial_state(chaos=True)

        # Generate trajectory
        trajectory = pendulum.solve(initial, t_span=(0, 20), dt=0.02)

        # Get Cartesian coordinates
        (x1, y1), (x2, y2) = pendulum.get_cartesian_positions(trajectory)

        # Scale for visualization
        scale = 2.5
        origin = ORIGIN

        # Create pendulum components with enhanced visuals
        pivot = Dot(origin, radius=0.08, color=GREY_B)
        bob1 = Dot(radius=0.14, color=BLUE_C, fill_opacity=1)
        bob2 = Dot(radius=0.14, color=RED_C, fill_opacity=1)

        # Add glow to bobs
        bob1_glow = Dot(radius=0.25, color=BLUE, fill_opacity=0.3)
        bob2_glow = Dot(radius=0.25, color=RED, fill_opacity=0.3)

        rod1 = Line(origin, bob1.get_center(), color=WHITE, stroke_width=4)
        rod2 = Line(bob1.get_center(), bob2.get_center(), color=WHITE, stroke_width=4)

        # Glowing trail for second bob
        trail = TracedPath(bob2.get_center, stroke_color=ORANGE, stroke_width=3, dissipating_time=3.5)
        trail_glow = TracedPath(bob2.get_center, stroke_color=YELLOW, stroke_width=6,
                               stroke_opacity=0.3, dissipating_time=3.5)

        self.add(pivot, rod1, rod2, bob1_glow, bob1, bob2_glow, bob2, trail_glow, trail)

        # Animate pendulum motion
        def update_pendulum(mob, alpha):
            idx = int(alpha * (len(trajectory) - 1))
            idx = np.clip(idx, 0, len(trajectory) - 1)

            # Update positions
            pos1 = origin + scale * np.array([x1[idx], y1[idx], 0])
            pos2 = origin + scale * np.array([x2[idx], y2[idx], 0])

            bob1.move_to(pos1)
            bob2.move_to(pos2)
            bob1_glow.move_to(pos1)
            bob2_glow.move_to(pos2)
            rod1.put_start_and_end_on(origin, pos1)
            rod2.put_start_and_end_on(pos1, pos2)

        self.play(
            UpdateFromAlphaFunc(bob1, update_pendulum),
            run_time=20,
            rate_func=smooth
        )

        self.wait(2)


class DoublePendulumComparison(Scene):
    """
    Show multiple double pendulums with slightly different initial conditions.
    Demonstrates extreme sensitivity to initial conditions with enhanced visuals.
    """

    def construct(self):
        # Title with subtitle
        title = Text("Sensitive Dependence", font_size=44, weight=BOLD)
        subtitle = Text("Tiny differences, vastly different outcomes", font_size=26, color=YELLOW)
        subtitle.next_to(title, DOWN)

        title_group = VGroup(title, subtitle)
        title_group.to_edge(UP)

        self.play(FadeIn(title, shift=DOWN), FadeIn(subtitle, shift=DOWN))
        self.wait()

        # Keep title but fade subtitle
        self.play(FadeOut(subtitle))
        title.to_edge(UP)

        # Create pendulum
        pendulum = create_standard_double_pendulum()

        # Generate multiple trajectories
        num_pendulums = 4
        colors = [RED, BLUE, GREEN, YELLOW]
        initial_states = pendulum.get_multiple_initial_states(
            num_states=num_pendulums,
            perturbation=0.001  # Very small difference
        )

        # Generate trajectories
        trajectories = []
        for state in initial_states:
            traj = pendulum.solve(state, t_span=(0, 15), dt=0.02)
            trajectories.append(traj)

        # Create visual elements for each pendulum with enhanced styling
        scale = 2.0
        origins = [
            LEFT * 5 + UP * 1,
            RIGHT * 5 + UP * 1,
            LEFT * 5 + DOWN * 2,
            RIGHT * 5 + DOWN * 2
        ]

        pendulum_groups = []
        trails = []
        trail_glows = []

        for i, (origin, color) in enumerate(zip(origins, colors)):
            pivot = Dot(origin, radius=0.06, color=GREY_B)
            bob1 = Dot(radius=0.11, color=color, fill_opacity=1)
            bob2 = Dot(radius=0.11, color=color, fill_opacity=1)
            rod1 = Line(origin, bob1.get_center(), color=WHITE, stroke_width=2.5)
            rod2 = Line(bob1.get_center(), bob2.get_center(), color=WHITE, stroke_width=2.5)

            # Enhanced trails with glow
            trail = TracedPath(
                bob2.get_center,
                stroke_color=color,
                stroke_width=2,
                dissipating_time=2.5
            )
            trail_glow = TracedPath(
                bob2.get_center,
                stroke_color=color,
                stroke_width=4,
                stroke_opacity=0.3,
                dissipating_time=2.5
            )

            group = VGroup(pivot, rod1, rod2, bob1, bob2)
            pendulum_groups.append(group)
            trails.append(trail)
            trail_glows.append(trail_glow)

            self.add(pivot, rod1, rod2, bob1, bob2, trail_glow, trail)

        # Animate all pendulums
        def update_all_pendulums(alpha):
            idx = int(alpha * (len(trajectories[0]) - 1))
            idx = np.clip(idx, 0, len(trajectories[0]) - 1)

            for i, (traj, group, origin) in enumerate(zip(trajectories, pendulum_groups, origins)):
                (x1, y1), (x2, y2) = pendulum.get_cartesian_positions(traj)

                pivot, rod1, rod2, bob1, bob2 = group

                pos1 = origin + scale * np.array([x1[idx], y1[idx], 0])
                pos2 = origin + scale * np.array([x2[idx], y2[idx], 0])

                bob1.move_to(pos1)
                bob2.move_to(pos2)
                rod1.put_start_and_end_on(origin, pos1)
                rod2.put_start_and_end_on(pos1, pos2)

        # Run animation with smooth rate
        self.play(
            UpdateFromAlphaFunc(VGroup(*pendulum_groups), lambda m, a: update_all_pendulums(a)),
            run_time=15,
            rate_func=smooth
        )

        # Add explanatory text with smooth entrance
        text = Text("Initial difference: 0.001 radians",
                   font_size=26, color=YELLOW, weight=BOLD)
        text.to_edge(DOWN)
        self.play(FadeIn(text, shift=UP))
        self.wait(3)


class DoublePendulumTrajectory(Scene):
    """
    Show the trajectory of the second bob in 2D space with glowing effect.
    """

    def construct(self):
        title = Text("Chaotic Trajectory Pattern", font_size=44, weight=BOLD)
        subtitle = Text("Path of the second bob", font_size=28, color=BLUE)
        subtitle.next_to(title, DOWN)

        self.play(FadeIn(title, shift=DOWN), FadeIn(subtitle, shift=DOWN))
        self.wait()
        self.play(FadeOut(subtitle), title.animate.scale(0.8).to_edge(UP))

        # Setup axes with minimal styling
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=8,
            y_length=8,
            axis_config={
                "include_numbers": False,
                "stroke_color": GREY_D,
                "stroke_width": 1,
            },
        )

        self.play(Create(axes))

        # Generate trajectory
        pendulum = create_standard_double_pendulum()
        initial = pendulum.get_default_initial_state(chaos=True)
        trajectory = pendulum.solve(initial, t_span=(0, 30), dt=0.01)

        # Get Cartesian coordinates for second bob
        (x1, y1), (x2, y2) = pendulum.get_cartesian_positions(trajectory)

        # Create path - subsample for performance
        points = [axes.c2p(x2[i], y2[i]) for i in range(0, len(x2), 5)]

        # Create smooth glowing path
        path = VMobject()
        path.set_points_smoothly([points[0], points[0], points[0]])
        path.set_stroke(width=3, color=BLUE_C)

        # Glow layers
        path_glow = VMobject()
        path_glow.set_points_smoothly([points[0], points[0], points[0]])
        path_glow.set_stroke(width=6, color=BLUE, opacity=0.4)

        path_glow_outer = VMobject()
        path_glow_outer.set_points_smoothly([points[0], points[0], points[0]])
        path_glow_outer.set_stroke(width=10, color=BLUE, opacity=0.2)

        self.add(path_glow_outer, path_glow, path)

        # Animate path drawing
        def update_path(mob, alpha):
            idx = int(alpha * (len(points) - 1))
            if idx >= 2:
                path.set_points_smoothly(points[:idx])
                path_glow.set_points_smoothly(points[:idx])
                path_glow_outer.set_points_smoothly(points[:idx])

        self.play(
            UpdateFromAlphaFunc(path, update_path),
            run_time=22,
            rate_func=smooth
        )

        self.wait(2)


class DoublePendulumIntro(Scene):
    """
    Introduction to the double pendulum with smooth animations.
    """

    def construct(self):
        # Title
        title = Text("The Double Pendulum", font_size=60, weight=BOLD)
        self.play(Write(title))
        self.wait()

        subtitle = Text("A Classic Example of Deterministic Chaos", font_size=32, color=BLUE)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(subtitle, shift=DOWN))
        self.wait(2)

        # Move up smoothly
        self.play(
            title.animate.scale(0.7).to_edge(UP),
            FadeOut(subtitle)
        )

        # Show equations with smooth entrance
        equations = VGroup(
            Text("Equations of motion (Lagrangian):", font_size=28, weight=BOLD),
            MathTex(
                r"\frac{d^2\theta_1}{dt^2} = f_1(\theta_1, \theta_2, \omega_1, \omega_2)",
                font_size=30,
                color=BLUE
            ),
            MathTex(
                r"\frac{d^2\theta_2}{dt^2} = f_2(\theta_1, \theta_2, \omega_1, \omega_2)",
                font_size=30,
                color=RED
            ),
        ).arrange(DOWN, buff=0.5)

        self.play(FadeIn(equations, shift=DOWN))
        self.wait(2)

        # Key points with smooth animations
        points = VGroup(
            Text("• Two pendulums attached end-to-end", font_size=26),
            Text("• Nonlinear coupled equations", font_size=26),
            Text("• Chaotic for large angles", font_size=26),
            Text("• Extreme sensitivity to initial conditions", font_size=26),
            Text("• Energy is conserved (Hamiltonian system)", font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        points.next_to(equations, DOWN, buff=0.7)

        for point in points:
            self.play(FadeIn(point, shift=RIGHT), run_time=0.6)
            self.wait(0.2)

        self.wait(2)
        self.play(FadeOut(points), FadeOut(equations), FadeOut(title))


# Utility function
def render_all_pendulum_scenes():
    """
    List of all double pendulum scenes.
    """
    return [
        DoublePendulumIntro,
        DoublePendulumScene,
        DoublePendulumComparison,
        DoublePendulumTrajectory,
    ]


if __name__ == "__main__":
    print("Double Pendulum animation scenes ready!")
    print("Render with: manim -pqh pendulum_anim.py DoublePendulumScene")
