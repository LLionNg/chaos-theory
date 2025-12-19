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
    Enhanced with beautiful gradient coloring.
    """

    def construct(self):
        # Title with smooth entrance
        title = Text("Logistic Map: Bifurcation Diagram", font_size=44, weight=BOLD)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.6).to_edge(UP))

        # Setup axes with cleaner look
        axes = Axes(
            x_range=[2.5, 4.0, 0.5],
            y_range=[0, 1, 0.2],
            x_length=11,
            y_length=6,
            axis_config={
                "include_numbers": True,
                "font_size": 28,
                "stroke_color": GREY_C,
            },
            tips=False,
        )
        axes.shift(DOWN * 0.5)

        x_label = Text("r", font_size=32, weight=BOLD).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("x", font_size=32, weight=BOLD).next_to(axes.y_axis.get_end(), UP)

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

        # Create dots for bifurcation diagram with gradient coloring
        dots = VGroup()
        for i, (r, x) in enumerate(zip(r_vals[::10], x_vals[::10])):  # Subsample for performance
            point = axes.c2p(r, x)
            # Color gradient from blue to cyan based on r value
            color_interpolation = (r - 2.5) / (4.0 - 2.5)
            color = interpolate_color(BLUE_E, TEAL_C, color_interpolation)
            dot = Dot(point, radius=0.012, color=color, fill_opacity=0.8)
            dots.add(dot)

        # Animate creation of bifurcation diagram with smooth effect
        self.play(Create(dots), run_time=10, rate_func=smooth)
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
    Cobweb plot showing the iteration process with enhanced visuals.
    """

    def construct(self):
        title = Text("Cobweb Plot: Visualizing Iteration", font_size=42, weight=BOLD)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.7).to_edge(UP))

        # Setup axes with cleaner styling
        axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=7,
            y_length=7,
            axis_config={
                "include_numbers": True,
                "font_size": 26,
                "stroke_color": GREY_C,
            },
        )
        axes.shift(DOWN * 0.3)

        x_label = Text("x_n", font_size=28, weight=BOLD).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("x_n+1", font_size=28, weight=BOLD).next_to(axes.y_axis.get_end(), UP)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # For r = 2.9 (stable fixed point)
        r = 2.9
        logistic = LogisticMap(r=r)

        # Plot the function with glow
        func_curve = axes.plot(
            lambda x: r * x * (1 - x),
            x_range=[0, 1],
            color=BLUE_C,
            stroke_width=3
        )
        func_glow = axes.plot(
            lambda x: r * x * (1 - x),
            x_range=[0, 1],
            color=BLUE,
            stroke_width=6,
            stroke_opacity=0.3
        )

        # Plot diagonal y = x with glow
        diagonal = axes.plot(
            lambda x: x,
            x_range=[0, 1],
            color=YELLOW_C,
            stroke_width=3
        )
        diagonal_glow = axes.plot(
            lambda x: x,
            x_range=[0, 1],
            color=YELLOW,
            stroke_width=6,
            stroke_opacity=0.3
        )

        self.play(
            Create(func_glow), Create(func_curve),
            Create(diagonal_glow), Create(diagonal)
        )
        self.wait()

        # Parameter text
        r_text = MathTex(f"r = {r}", font_size=36, color=TEAL)
        r_text.to_corner(UR)
        self.play(Write(r_text))

        # Generate cobweb
        x_coords, y_coords = logistic.cobweb_plot_data(x0=0.1, num_iterations=20, r=r)

        # Draw cobweb with gradient
        cobweb = VGroup()
        for i in range(len(x_coords) - 1):
            # Fade color as iterations progress
            alpha = i / len(x_coords)
            color = interpolate_color(RED_C, PINK, alpha)
            line = Line(
                axes.c2p(x_coords[i], y_coords[i]),
                axes.c2p(x_coords[i+1], y_coords[i+1]),
                color=color,
                stroke_width=2.5
            )
            cobweb.add(line)

        self.play(Create(cobweb), run_time=12, rate_func=smooth)
        self.wait()

        # Show convergence
        convergence_text = Text("Converges to fixed point", font_size=28, color=GREEN, weight=BOLD)
        convergence_text.next_to(axes, DOWN)
        self.play(FadeIn(convergence_text, shift=UP))
        self.wait(3)


class PeriodDoublingScene(Scene):
    """
    Show the period-doubling cascade with enhanced visuals.
    """

    def construct(self):
        title = Text("Period-Doubling Route to Chaos", font_size=42, weight=BOLD)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.to_edge(UP))

        # Show sequence of r values and their periods
        r_values = [2.8, 3.2, 3.5, 3.56, 3.57, 3.7]
        periods = [1, 2, 4, 8, "chaos", "chaos"]
        colors = [GREEN_C, BLUE_C, TEAL_C, YELLOW_C, ORANGE, RED_C]

        for idx, (r, period, color) in enumerate(zip(r_values, periods, colors)):
            # Clear previous
            if idx > 0:
                self.clear()
                self.add(title)

            # Create axes with cleaner styling
            axes = Axes(
                x_range=[0, 100, 20],
                y_range=[0, 1, 0.2],
                x_length=10,
                y_length=5,
                axis_config={
                    "include_numbers": True,
                    "font_size": 24,
                    "stroke_color": GREY_C,
                },
            )
            axes.shift(DOWN * 0.5)

            x_label = Text("n", font_size=28, weight=BOLD).next_to(axes.x_axis.get_end(), RIGHT)
            y_label = Text("x_n", font_size=28, weight=BOLD).next_to(axes.y_axis.get_end(), UP)

            self.play(Create(axes), Write(x_label), Write(y_label), run_time=0.5)

            # Generate trajectory
            logistic = LogisticMap(r=r)
            traj = logistic.trajectory(x0=0.1, num_iterations=100, transient=0, r=r)

            # Plot trajectory with glow effect
            points = [axes.c2p(n, traj[n]) for n in range(len(traj))]
            dots = VGroup(*[Dot(p, radius=0.035, color=color, fill_opacity=0.9) for p in points])

            # Add subtle glow dots
            glow_dots = VGroup(*[Dot(p, radius=0.06, color=color, fill_opacity=0.2) for p in points])

            # Info text with color
            if period == "chaos":
                period_text = MathTex(f"r = {r}", r"\text{ (Chaotic)}", font_size=32, color=color)
            else:
                period_text = MathTex(f"r = {r}", f"\\text{{ (Period-{period})}}", font_size=32, color=color)
            period_text.to_corner(UR)

            self.play(FadeIn(period_text, shift=DOWN))
            self.play(Create(glow_dots), Create(dots), run_time=2.5)
            self.wait(1.5)


class LogisticMapIntro(Scene):
    """
    Introduction to the logistic map with smooth animations.
    """

    def construct(self):
        # Title
        title = Text("The Logistic Map", font_size=60, weight=BOLD)
        self.play(Write(title))
        self.wait()

        subtitle = Text("Simplicity Leading to Chaos", font_size=36, color=BLUE)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(subtitle, shift=DOWN))
        self.wait(2)

        # Move up smoothly
        self.play(
            title.animate.scale(0.8).to_edge(UP),
            FadeOut(subtitle)
        )

        # Show the equation with glow
        equation = MathTex(
            "x_{n+1} = r \\cdot x_n \\cdot (1 - x_n)",
            font_size=52,
            color=TEAL
        )
        self.play(Write(equation))
        self.wait(2)

        # Explanation with smooth entrance
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

        self.wait(2)
        self.play(FadeOut(explanation), FadeOut(equation), FadeOut(title))


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
