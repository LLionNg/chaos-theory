"""
Render all Manim animations for the Chaos Theory project.

This script provides a convenient way to render all animations
at different quality levels.
"""

import subprocess
import os
import sys
from pathlib import Path


# Animation files and their scenes
ANIMATIONS = {
    'lorenz_anim.py': [
        'LorenzIntroScene',
        'LorenzParametersScene',
        'LorenzAttractorScene',
        'ButterflyEffectScene',
        'LorenzPhaseSpaceScene',
    ],
    'logistic_anim.py': [
        'LogisticMapIntro',
        'CobwebPlot',
        'PeriodDoublingScene',
        'BifurcationDiagram',
    ],
    'pendulum_anim.py': [
        'DoublePendulumIntro',
        'DoublePendulumScene',
        'DoublePendulumComparison',
        'DoublePendulumTrajectory',
    ],
    'rossler_anim.py': [
        'RosslerIntroScene',
        'RosslerParameterScene',
        'RosslerAttractorScene',
        'RosslerScrollScene',
        'RosslerMultipleTrajectories',
    ],
}

# Quality presets
QUALITY_FLAGS = {
    'low': '-ql',      # 480p, fast preview
    'medium': '-qm',   # 720p
    'high': '-qh',     # 1080p (recommended)
    '4k': '-qk',       # 2160p, slow render
}


def render_scene(file_path, scene_name, quality='high', preview=False):
    """
    Render a single Manim scene.

    Args:
        file_path: Path to the animation file
        scene_name: Name of the scene class
        quality: Quality preset ('low', 'medium', 'high', '4k')
        preview: Whether to preview after rendering
    """
    quality_flag = QUALITY_FLAGS.get(quality, '-qh')
    preview_flag = '-p' if preview else ''

    cmd = f"manim {quality_flag} {preview_flag} {file_path} {scene_name}"

    print(f"\n{'='*60}")
    print(f"Rendering: {scene_name}")
    print(f"Quality: {quality} ({quality_flag})")
    print(f"Command: {cmd}")
    print('='*60)

    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"✓ Successfully rendered {scene_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error rendering {scene_name}: {e}")
        return False


def render_all(quality='high', preview=False, specific_system=None):
    """
    Render all animations or animations for a specific system.

    Args:
        quality: Quality preset
        preview: Whether to preview each animation
        specific_system: Render only specific system ('lorenz', 'logistic', 'pendulum', 'rossler')
    """
    # Get animations directory
    project_root = Path(__file__).parent.parent
    anim_dir = project_root / 'src' / 'animations'

    # Filter animations if specific system requested
    animations_to_render = ANIMATIONS.copy()
    if specific_system:
        system_map = {
            'lorenz': 'lorenz_anim.py',
            'logistic': 'logistic_anim.py',
            'pendulum': 'pendulum_anim.py',
            'rossler': 'rossler_anim.py',
        }
        if specific_system in system_map:
            file_name = system_map[specific_system]
            animations_to_render = {file_name: ANIMATIONS[file_name]}
        else:
            print(f"Unknown system: {specific_system}")
            print(f"Available: {', '.join(system_map.keys())}")
            return

    # Statistics
    total_scenes = sum(len(scenes) for scenes in animations_to_render.values())
    rendered = 0
    failed = 0

    print("\n" + "="*60)
    print(f"RENDERING {total_scenes} SCENES AT {quality.upper()} QUALITY")
    print("="*60)

    # Render each animation
    for anim_file, scenes in animations_to_render.items():
        file_path = anim_dir / anim_file

        if not file_path.exists():
            print(f"\n✗ File not found: {file_path}")
            failed += len(scenes)
            continue

        print(f"\n{'#'*60}")
        print(f"# {anim_file.replace('.py', '').upper()}")
        print(f"# {len(scenes)} scenes")
        print(f"{'#'*60}")

        for scene in scenes:
            success = render_scene(str(file_path), scene, quality, preview=False)
            if success:
                rendered += 1
            else:
                failed += 1

    # Summary
    print("\n" + "="*60)
    print("RENDERING COMPLETE")
    print("="*60)
    print(f"✓ Successfully rendered: {rendered}/{total_scenes}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{total_scenes}")
    print("="*60)
    print(f"\nOutputs saved to: media/videos/")
    print("="*60)


def render_quick_preview():
    """
    Render a quick preview of one scene from each system at low quality.
    """
    print("\n" + "="*60)
    print("QUICK PREVIEW MODE")
    print("Rendering one scene from each system at low quality")
    print("="*60)

    project_root = Path(__file__).parent.parent
    anim_dir = project_root / 'src' / 'animations'

    preview_scenes = {
        'lorenz_anim.py': 'LorenzAttractorScene',
        'logistic_anim.py': 'BifurcationDiagram',
        'pendulum_anim.py': 'DoublePendulumScene',
        'rossler_anim.py': 'RosslerAttractorScene',
    }

    for anim_file, scene in preview_scenes.items():
        file_path = anim_dir / anim_file
        if file_path.exists():
            render_scene(str(file_path), scene, quality='low', preview=True)


def main():
    """
    Main function with command-line interface.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Render Chaos Theory animations with Manim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python render_all.py                    # Render all at high quality
  python render_all.py --quality medium   # Render all at 720p
  python render_all.py --system lorenz    # Render only Lorenz scenes
  python render_all.py --preview          # Quick preview (one scene each, low quality)

Available systems:
  lorenz, logistic, pendulum, rossler
        """
    )

    parser.add_argument(
        '--quality', '-q',
        choices=['low', 'medium', 'high', '4k'],
        default='high',
        help='Render quality (default: high/1080p)'
    )

    parser.add_argument(
        '--system', '-s',
        choices=['lorenz', 'logistic', 'pendulum', 'rossler'],
        help='Render only specific system'
    )

    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Quick preview mode (one scene per system, low quality)'
    )

    args = parser.parse_args()

    if args.preview:
        render_quick_preview()
    else:
        render_all(quality=args.quality, specific_system=args.system)


if __name__ == "__main__":
    # Check if manim is installed
    try:
        subprocess.run(['manim', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Manim is not installed or not in PATH")
        print("Install with: pip install manim")
        sys.exit(1)

    main()
