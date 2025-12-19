"""
Manim animations for the Logistic Map.

This module contains educational animations demonstrating:
- The famous bifurcation diagram
- Cobweb plots showing iteration process
- Period-doubling route to chaos
- Chaotic vs periodic behavior
"""

from manim import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.systems.logistic_map import LogisticMap, create_chaotic_logistic
from src.utils.colors import BLUE, GREEN, YELLOW, RED, PURPLE, TEAL


class BifurcationDiagram(Scene):
    """
    The famous bifurcation diagram showing period-doubling route to chaos.
    """

    def construct(self):
        # Title
        title = Text("Logistic Map: Bifurcation Diagram", font_size=44)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.6).to_edge(UP))

        # Setup axes
        axes = Axes(
            x_range=[2.5, 4.0, 0.5],
            y_range=[0, 1, 0.2],
            x_length=11,
            y_length=6,
            axis_config={"include_numbers": True},
            tips=False,
        )
        axes.shift(DOWN * 0.5)

        x_label = axes.get_x_axis_label("r", edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label("x", edge=LEFT, direction=LEFT)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait()

        # Generate bifurcation diagram data
        logistic = LogisticMap()
        r_vals, x_vals = logistic.bifurcation_diagram(
            r_min=2.5,
            r_max=4.0,
            num_r=2000,
            num_iterations=500,
            transient=100
        )

        # Create dots for bifurcation diagram
        dots = VGroup()
        for r, x in zip(r_vals[::10], x_vals[::10]):  # Subsample for performance
            point = axes.c2p(r, x)
            dot = Dot(point, radius=0.01, color=BLUE)
            dots.add(dot)

        # Animate creation of bifurcation diagram
        self.play(Create(dots), run_time=8, rate_func=linear)
        self.wait()

        # Highlight regions
        # Region 1: Fixed point (r < 3)
        region1 = axes.get_area(
            axes.plot(lambda r: 0.5, x_range=[2.5, 3.0]),
            x_range=[2.5, 3.0],
            color=GREEN,
            opacity=0.2
        )
        label1 = Text("Fixed Point", font_size=20, color=GREEN)
        label1.next_to(axes.c2p(2.75, 0.8), UP)

        self.play(FadeIn(region1), Write(label1))
        self.wait(2)

        # Region 2: Period-2 (3 < r < 3.45)
        region2_rect = Rectangle(
            width=axes.x_length * 0.45 / 1.5,
            height=axes.y_length,
            color=YELLOW,
            stroke_opacity=0.7,
            fill_opacity=0.1
        )
        region2_rect.move_to(axes.c2p(3.2, 0.5))
        label2 = Text("Period-2", font_size=20, color=YELLOW)
        label2.next_to(axes.c2p(3.2, 0.9), UP)

        self.play(Create(region2_rect), Write(label2))
        self.wait(2)

        # Region 3: Chaos (r > 3.57)
        region3_rect = Rectangle(
            width=axes.x_length * 0.43 / 1.5,
            height=axes.y_length,
            color=RED,
            stroke_opacity=0.7,
            fill_opacity=0.1
        )
        region3_rect.move_to(axes.c2p(3.8, 0.5))
        label3 = Text("Chaos", font_size=20, color=RED)
        label3.next_to(axes.c2p(3.8, 0.9), UP)

        self.play(Create(region3_rect), Write(label3))
        self.wait(3)

        self.play(
            FadeOut(region1), FadeOut(region2_rect), FadeOut(region3_rect),
            FadeOut(label1), FadeOut(label2), FadeOut(label3)
        )
        self.wait(2)


class CobwebPlot(Scene):
    """
    Cobweb plot showing the iteration process of the logistic map.
    """

    def construct(self):
        title = Text("Cobweb Plot: Visualizing Iteration", font_size=42)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.7).to_edge(UP))

        # Setup axes
        axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=7,
            y_length=7,
            axis_config={"include_numbers": True},
        )
        axes.shift(DOWN * 0.3)

        x_label = axes.get_x_axis_label("x_n", edge=DOWN)
        y_label = axes.get_y_axis_label("x_{n+1}", edge=LEFT)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # For r = 2.9 (stable fixed point)
        r = 2.9
        logistic = LogisticMap(r=r)

        # Plot the function
        func_curve = axes.plot(
            lambda x: r * x * (1 - x),
            x_range=[0, 1],
            color=BLUE
        )

        # Plot diagonal y = x
        diagonal = axes.plot(
            lambda x: x,
            x_range=[0, 1],
            color=YELLOW,
            stroke_width=2
        )

        self.play(Create(func_curve), Create(diagonal))
        self.wait()

        # Parameter text
        r_text = MathTex(f"r = {r}", font_size=36)
        r_text.to_corner(UR)
        self.play(Write(r_text))

        # Generate cobweb
        x_coords, y_coords = logistic.cobweb_plot_data(x0=0.1, num_iterations=20, r=r)

        # Draw cobweb
        cobweb = VGroup()
        for i in range(len(x_coords) - 1):
            line = Line(
                axes.c2p(x_coords[i], y_coords[i]),
                axes.c2p(x_coords[i+1], y_coords[i+1]),
                color=RED,
                stroke_width=2
            )
            cobweb.add(line)

        self.play(Create(cobweb), run_time=10, rate_func=linear)
        self.wait()

        # Show convergence
        convergence_text = Text("Converges to fixed point", font_size=28, color=GREEN)
        convergence_text.next_to(axes, DOWN)
        self.play(Write(convergence_text))
        self.wait(3)


class PeriodDoublingScene(Scene):
    """
    Show the period-doubling cascade.
    """

    def construct(self):
        title = Text("Period-Doubling Route to Chaos", font_size=42)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.to_edge(UP))

        # Show sequence of r values and their periods
        r_values = [2.8, 3.2, 3.5, 3.56, 3.57, 3.7]
        periods = [1, 2, 4, 8, "chaos", "chaos"]

        for idx, (r, period) in enumerate(zip(r_values, periods)):
            # Clear previous
            if idx > 0:
                self.clear()
                self.add(title)

            # Create axes
            axes = Axes(
                x_range=[0, 100, 20],
                y_range=[0, 1, 0.2],
                x_length=10,
                y_length=5,
                axis_config={"include_numbers": True},
            )
            axes.shift(DOWN * 0.5)

            x_label = axes.get_x_axis_label("n", edge=DOWN)
            y_label = axes.get_y_axis_label("x_n", edge=LEFT)

            self.play(Create(axes), Write(x_label), Write(y_label), run_time=0.5)

            # Generate trajectory
            logistic = LogisticMap(r=r)
            traj = logistic.trajectory(x0=0.1, num_iterations=100, transient=0, r=r)

            # Plot trajectory
            points = [axes.c2p(n, traj[n]) for n in range(len(traj))]
            dots = VGroup(*[Dot(p, radius=0.03, color=BLUE) for p in points])

            # Info text
            if period == "chaos":
                period_text = MathTex(f"r = {r}", r"\text{ (Chaotic)}", font_size=32)
            else:
                period_text = MathTex(f"r = {r}", f"\\text{{ (Period-{period})}}", font_size=32)
            period_text.to_corner(UR)

            self.play(Write(period_text))
            self.play(Create(dots), run_time=2)
            self.wait(1.5)


class LogisticMapIntro(Scene):
    """
    Introduction to the logistic map.
    """

    def construct(self):
        # Title
        title = Text("The Logistic Map", font_size=56, weight=BOLD)
        self.play(Write(title))
        self.wait()

        subtitle = Text("Simplicity Leading to Chaos", font_size=36, color=BLUE)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(subtitle))
        self.wait(2)

        self.play(FadeOut(subtitle), title.animate.to_edge(UP))

        # Show the equation
        equation = MathTex(
            "x_{n+1} = r \\cdot x_n \\cdot (1 - x_n)",
            font_size=48
        )
        self.play(Write(equation))
        self.wait(2)

        # Explanation
        explanation = VGroup(
            Text("Simple nonlinear difference equation", font_size=28),
            Text("Models population growth with limiting factor", font_size=28),
            Text("Parameter r controls growth rate", font_size=28),
            Text("Demonstrates period-doubling route to chaos", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        explanation.next_to(equation, DOWN, buff=0.8)

        for line in explanation:
            self.play(FadeIn(line, shift=RIGHT))
            self.wait(0.7)

        self.wait(3)


# Utility function
def render_all_logistic_scenes():
    """
    List of all logistic map scenes.
    """
    return [
        LogisticMapIntro,
        CobwebPlot,
        PeriodDoublingScene,
        BifurcationDiagram,
    ]


if __name__ == "__main__":
    print("Logistic Map animation scenes ready!")
    print("Render with: manim -pqh logistic_anim.py BifurcationDiagram")
