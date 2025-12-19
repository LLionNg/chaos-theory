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
    Animate the double pendulum with trail showing chaotic motion.
    """

    def construct(self):
        # Title
        title = Text("Double Pendulum Chaos", font_size=48)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.7).to_edge(UP))

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

        # Create pendulum components
        pivot = Dot(origin, radius=0.08, color=GRAY)
        bob1 = Dot(radius=0.12, color=BLUE)
        bob2 = Dot(radius=0.12, color=RED)
        rod1 = Line(origin, bob1.get_center(), color=WHITE, stroke_width=3)
        rod2 = Line(bob1.get_center(), bob2.get_center(), color=WHITE, stroke_width=3)

        # Trail for second bob
        trail = TracedPath(bob2.get_center, stroke_color=ORANGE, stroke_width=2, dissipating_time=3.0)

        self.add(pivot, rod1, rod2, bob1, bob2, trail)

        # Animate pendulum motion
        def update_pendulum(mob, alpha):
            idx = int(alpha * (len(trajectory) - 1))
            idx = np.clip(idx, 0, len(trajectory) - 1)

            # Update positions
            pos1 = origin + scale * np.array([x1[idx], y1[idx], 0])
            pos2 = origin + scale * np.array([x2[idx], y2[idx], 0])

            bob1.move_to(pos1)
            bob2.move_to(pos2)
            rod1.put_start_and_end_on(origin, pos1)
            rod2.put_start_and_end_on(pos1, pos2)

        self.play(
            UpdateFromAlphaFunc(bob1, update_pendulum),
            run_time=20,
            rate_func=linear
        )

        self.wait(2)


class DoublePendulumComparison(Scene):
    """
    Show multiple double pendulums with slightly different initial conditions.
    Demonstrates extreme sensitivity to initial conditions.
    """

    def construct(self):
        # Title
        title = Text("Sensitive Dependence: Double Pendulum", font_size=40)
        title.to_edge(UP)
        self.add(title)

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

        # Create visual elements for each pendulum
        scale = 2.0
        origins = [
            LEFT * 5 + UP * 1,
            RIGHT * 5 + UP * 1,
            LEFT * 5 + DOWN * 2,
            RIGHT * 5 + DOWN * 2
        ]

        pendulum_groups = []
        trails = []

        for i, (origin, color) in enumerate(zip(origins, colors)):
            pivot = Dot(origin, radius=0.06, color=GRAY)
            bob1 = Dot(radius=0.1, color=color)
            bob2 = Dot(radius=0.1, color=color)
            rod1 = Line(origin, bob1.get_center(), color=WHITE, stroke_width=2)
            rod2 = Line(bob1.get_center(), bob2.get_center(), color=WHITE, stroke_width=2)

            trail = TracedPath(
                bob2.get_center,
                stroke_color=color,
                stroke_width=1.5,
                dissipating_time=2.0
            )

            group = VGroup(pivot, rod1, rod2, bob1, bob2)
            pendulum_groups.append(group)
            trails.append(trail)

            self.add(pivot, rod1, rod2, bob1, bob2, trail)

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

        # Run animation
        self.play(
            UpdateFromAlphaFunc(VGroup(*pendulum_groups), lambda m, a: update_all_pendulums(a)),
            run_time=15,
            rate_func=linear
        )

        # Add text
        text = Text("Tiny initial differences lead to vastly different outcomes",
                   font_size=24, color=YELLOW)
        text.to_edge(DOWN)
        self.add(text)
        self.wait(3)


class DoublePendulumTrajectory(Scene):
    """
    Show the trajectory of the second bob in 2D space.
    """

    def construct(self):
        title = Text("Chaotic Trajectory Pattern", font_size=44)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.to_edge(UP))

        # Setup axes
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=8,
            y_length=8,
            axis_config={"include_numbers": False},
        )

        self.play(Create(axes))

        # Generate trajectory
        pendulum = create_standard_double_pendulum()
        initial = pendulum.get_default_initial_state(chaos=True)
        trajectory = pendulum.solve(initial, t_span=(0, 30), dt=0.01)

        # Get Cartesian coordinates for second bob
        (x1, y1), (x2, y2) = pendulum.get_cartesian_positions(trajectory)

        # Create path
        points = [axes.c2p(x2[i], y2[i]) for i in range(0, len(x2), 5)]  # Subsample

        # Draw path
        path = VMobject(color=BLUE)
        path.set_points_smoothly([points[0], points[0], points[0]])

        self.add(path)

        # Animate path drawing
        def update_path(mob, alpha):
            idx = int(alpha * (len(points) - 1))
            if idx >= 2:
                mob.set_points_smoothly(points[:idx])

        self.play(
            UpdateFromAlphaFunc(path, update_path),
            run_time=20,
            rate_func=linear
        )

        self.wait(2)


class DoublePendulumIntro(Scene):
    """
    Introduction to the double pendulum.
    """

    def construct(self):
        # Title
        title = Text("The Double Pendulum", font_size=56, weight=BOLD)
        self.play(Write(title))
        self.wait()

        subtitle = Text("A Classic Example of Deterministic Chaos", font_size=32, color=BLUE)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(subtitle))
        self.wait(2)

        self.play(FadeOut(subtitle), title.animate.to_edge(UP))

        # Show equations
        equations = VGroup(
            Text("Equations of motion (Lagrangian):", font_size=28),
            MathTex(
                r"\frac{d^2\theta_1}{dt^2} = f_1(\theta_1, \theta_2, \omega_1, \omega_2)",
                font_size=30
            ),
            MathTex(
                r"\frac{d^2\theta_2}{dt^2} = f_2(\theta_1, \theta_2, \omega_1, \omega_2)",
                font_size=30
            ),
        ).arrange(DOWN, buff=0.5)

        self.play(Write(equations))
        self.wait(2)

        # Key points
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

        self.wait(3)


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
