# Chaos Theory Visualization Project

A comprehensive educational project for visualizing chaos theory concepts using modern animation libraries. Features high-quality 1080p animations created with Manim, demonstrating beautiful mathematical chaos through the Lorenz Attractor, Logistic Map, Double Pendulum, and Rössler Attractor.

## Features

- **Four Chaos Systems**: Lorenz, Rössler, Logistic Map, and Double Pendulum
- **High-Quality Animations**: 1080p @ 60fps using Manim
- **Educational Focus**: Clear visualizations with annotations
- **Cross-Platform**: Works on Windows and Linux
- **Modular Architecture**: Easy to extend with new systems

## Gallery

(Animations will be rendered in the `outputs/` directory)

## Installation

### Prerequisites

**System Dependencies:**

For **Linux/WSL**:
```bash
sudo apt update
sudo apt install -y build-essential python3-dev libcairo2-dev libpango1.0-dev ffmpeg
```

For **Windows**:
- Install [FFmpeg](https://ffmpeg.org/download.html) and add to PATH
- Install [MiKTeX](https://miktex.org/download) or another LaTeX distribution (for Manim text rendering)

### Python Environment

1. Clone this repository:
```bash
git clone <repository-url>
cd chaos-theory
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Render Individual Animations

```bash
# Lorenz Attractor
manim -pqh src/animations/lorenz_anim.py LorenzAttractorScene

# Logistic Map Bifurcation
manim -pqh src/animations/logistic_anim.py BifurcationDiagram

# Double Pendulum
manim -pqh src/animations/pendulum_anim.py DoublePendulumScene

# Rössler Attractor
manim -pqh src/animations/rossler_anim.py RosslerAttractorScene
```

### Run Demo Scripts

```bash
# Quick Lorenz demo
python examples/lorenz_demo.py

# Generate bifurcation diagram
python examples/bifurcation_demo.py

# Render all animations
python examples/render_all.py
```

## Project Structure

```
chaos-theory/
├── src/
│   ├── systems/          # Chaos system implementations
│   │   ├── base.py      # Base class
│   │   ├── lorenz.py    # Lorenz attractor
│   │   ├── rossler.py   # Rössler attractor
│   │   ├── logistic_map.py  # Logistic map
│   │   └── double_pendulum.py  # Double pendulum
│   ├── animations/       # Manim animation scenes
│   ├── utils/           # Utility modules
│   └── interactive/     # Interactive viewers (future)
├── examples/            # Demo scripts
├── outputs/            # Generated animations
└── tests/              # Unit tests
```

## Chaos Systems

### Lorenz Attractor
The classic "butterfly effect" system discovered by Edward Lorenz in 1963. Demonstrates sensitive dependence on initial conditions in a 3D phase space.

**Parameters:**
- σ (sigma) = 10.0
- ρ (rho) = 28.0
- β (beta) = 8/3

**Equations:**
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

### Rössler Attractor
A continuous-time dynamical system that exhibits chaotic dynamics with a scroll-like shape.

**Parameters:**
- a = 0.2
- b = 0.2
- c = 5.7

**Equations:**
```
dx/dt = -y - z
dy/dt = x + ay
dz/dt = b + z(x - c)
```

### Logistic Map
A discrete-time system demonstrating the period-doubling route to chaos. Famous for its bifurcation diagram.

**Equation:**
```
x[n+1] = r * x[n] * (1 - x[n])
```

Where r is the growth rate parameter (0 to 4).

### Double Pendulum
A physical system consisting of two pendulums attached end-to-end, exhibiting chaotic motion for large initial angles.

## Customization

### Adjusting System Parameters

Edit the chaos system files in `src/systems/` to modify parameters:

```python
from src.systems.lorenz import LorenzSystem

# Custom parameters
system = LorenzSystem(sigma=10, rho=28, beta=8/3)
trajectory = system.solve(t_span=(0, 50), initial_state=[1, 1, 1])
```

### Animation Settings

Modify render quality in Manim commands:
- `-ql`: Low quality (480p, fast preview)
- `-qm`: Medium quality (720p)
- `-qh`: High quality (1080p)
- `-qk`: 4K quality (2160p, slow render)

Flags:
- `-p`: Preview after rendering
- `-s`: Save last frame as image
- `--format=gif`: Export as GIF instead of video

## Usage Examples

### Python API

```python
from src.systems.lorenz import LorenzSystem
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create system
lorenz = LorenzSystem()

# Generate trajectory
trajectory = lorenz.solve(t_span=(0, 50), initial_state=[1, 1, 1], dt=0.01)

# Plot with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
plt.show()
```

### Bifurcation Analysis

```python
from src.systems.logistic_map import LogisticMap

# Generate bifurcation diagram
logistic = LogisticMap()
r_values, x_values = logistic.bifurcation_diagram(
    r_min=2.5, r_max=4.0, num_r=1000, iterations=1000, last_n=100
)
```

## Mathematical Background

### What is Chaos Theory?

Chaos theory studies dynamical systems that are highly sensitive to initial conditions—a phenomenon popularly referred to as the "butterfly effect." Small differences in initial conditions yield widely diverging outcomes, making long-term prediction impossible despite being deterministic.

### Key Concepts

- **Sensitive Dependence on Initial Conditions**: Tiny changes lead to vastly different outcomes
- **Strange Attractors**: Complex structures in phase space that the system approaches
- **Period Doubling**: Route to chaos through successive bifurcations
- **Fractals**: Self-similar patterns at different scales

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Chaos Systems

1. Create a new class in `src/systems/` inheriting from `ChaosSystem`
2. Implement the `derivative()` method
3. Create corresponding animation in `src/animations/`
4. Add demo script in `examples/`

## Troubleshooting

### Manim Installation Issues

If you encounter LaTeX errors:
```bash
# Linux/WSL
sudo apt install texlive-full

# Windows: Install MiKTeX from https://miktex.org/
```

### FFmpeg Not Found

Ensure FFmpeg is in your system PATH:
```bash
# Test FFmpeg
ffmpeg -version
```

### Slow Rendering

- Use lower quality settings (`-ql` or `-qm`) for previews
- Reduce the number of frames or trajectory points
- Use GPU acceleration if available

## Credits

Created as an educational resource for understanding chaos theory through visualization.

Built with:
- [Manim](https://www.manim.community/) - Mathematical animation engine
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Scientific computing

## License

MIT License - Feel free to use for educational purposes.

## Contributing

Contributions welcome! Areas for improvement:
- Additional chaos systems (Chua's circuit, Duffing oscillator, etc.)
- Interactive real-time simulations
- More animation variations
- Educational annotations and explanations
- Performance optimizations

## Resources

- [Chaos: An Introduction to Dynamical Systems](https://www.springer.com/gp/book/9780387946771) by Alligood, Sauer, and Yorke
- [Nonlinear Dynamics and Chaos](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering) by Steven Strogatz
- [3Blue1Brown's Chaos Theory Video](https://www.youtube.com/watch?v=ovJcsL7vyrk)
