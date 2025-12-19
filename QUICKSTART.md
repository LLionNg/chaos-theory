# Quick Start Guide

Get up and running with the Chaos Theory project in minutes!

## Installation

### 1. System Dependencies

**For Linux/WSL:**
```bash
sudo apt update
sudo apt install -y build-essential python3-dev libcairo2-dev libpango1.0-dev ffmpeg
```

**For Windows:**
- Install [FFmpeg](https://ffmpeg.org/download.html) and add to PATH
- Install [MiKTeX](https://miktex.org/download) (for LaTeX rendering)

### 2. Python Setup

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Linux/WSL:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test

### Run a Demo Script

```bash
# Lorenz attractor visualization
python examples/lorenz_demo.py

# Bifurcation diagram
python examples/bifurcation_demo.py
```

These will create matplotlib plots and save them to `outputs/`.

## Render Your First Animation

### Single Animation (Quick Preview)

```bash
# Lorenz attractor (low quality for speed)
manim -pql src/animations/lorenz_anim.py LorenzAttractorScene
```

Flags:
- `-p`: Preview after rendering
- `-q`: Quality level
  - `l`: Low (480p) - fast preview
  - `m`: Medium (720p)
  - `h`: High (1080p) - recommended
  - `k`: 4K (2160p) - slow

### Render All Animations

```bash
# All animations at 1080p (takes a while!)
python examples/render_all.py

# Quick preview mode (one per system, low quality)
python examples/render_all.py --preview

# Specific system only
python examples/render_all.py --system lorenz

# Different quality
python examples/render_all.py --quality medium
```

## Quick Examples

### Python API Usage

```python
from src.systems.lorenz import LorenzSystem
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create Lorenz system
lorenz = LorenzSystem(sigma=10, rho=28, beta=8/3)

# Generate trajectory
trajectory = lorenz.solve(
    initial_state=[1, 1, 1],
    t_span=(0, 50),
    dt=0.01
)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
plt.show()
```

### Logistic Map

```python
from src.systems.logistic_map import LogisticMap
import matplotlib.pyplot as plt

# Create logistic map
logistic = LogisticMap(r=3.7)  # Chaotic regime

# Generate bifurcation diagram
r_vals, x_vals = logistic.bifurcation_diagram()

# Plot
plt.figure(figsize=(12, 8))
plt.plot(r_vals, x_vals, ',k', markersize=0.5)
plt.xlabel('r')
plt.ylabel('x')
plt.title('Bifurcation Diagram')
plt.show()
```

## Available Animations

### Lorenz Attractor
```bash
manim -pqh src/animations/lorenz_anim.py LorenzIntroScene
manim -pqh src/animations/lorenz_anim.py LorenzAttractorScene
manim -pqh src/animations/lorenz_anim.py ButterflyEffectScene
manim -pqh src/animations/lorenz_anim.py LorenzPhaseSpaceScene
```

### Logistic Map
```bash
manim -pqh src/animations/logistic_anim.py LogisticMapIntro
manim -pqh src/animations/logistic_anim.py BifurcationDiagram
manim -pqh src/animations/logistic_anim.py CobwebPlot
manim -pqh src/animations/logistic_anim.py PeriodDoublingScene
```

### Double Pendulum
```bash
manim -pqh src/animations/pendulum_anim.py DoublePendulumIntro
manim -pqh src/animations/pendulum_anim.py DoublePendulumScene
manim -pqh src/animations/pendulum_anim.py DoublePendulumComparison
```

### Rössler Attractor
```bash
manim -pqh src/animations/rossler_anim.py RosslerIntroScene
manim -pqh src/animations/rossler_anim.py RosslerAttractorScene
manim -pqh src/animations/rossler_anim.py RosslerScrollScene
```

## Troubleshooting

### Manim Not Found
```bash
# Make sure manim is installed
pip install manim

# Check version
manim --version
```

### LaTeX Errors
```bash
# Linux/WSL
sudo apt install texlive-full

# Or use text mode without LaTeX
manim --disable_caching -pqh <file> <scene>
```

### FFmpeg Not Found
```bash
# Test FFmpeg
ffmpeg -version

# If not found, install:
# Linux/WSL:
sudo apt install ffmpeg

# Windows: Download from ffmpeg.org and add to PATH
```

### Import Errors
```bash
# Make sure you're in the project root
cd chaos-theory

# And that virtual environment is activated
source venv/bin/activate  # Linux/WSL
# or
venv\Scripts\activate  # Windows
```

## Output Locations

- **Manim animations**: `media/videos/`
- **Demo plots**: `outputs/`
- **Temporary files**: `.pytest_cache/`, `__pycache__/`

## Next Steps

1. Explore the code in `src/systems/` to understand each chaos system
2. Check out `src/animations/` for animation code
3. Modify parameters in examples to see different behaviors
4. Create your own chaos system by inheriting from `ChaosSystem` base class
5. Build custom animations with Manim

## Resources

- [Manim Documentation](https://docs.manim.community/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Project README](README.md)

---

**Have fun exploring chaos!** 🦋
