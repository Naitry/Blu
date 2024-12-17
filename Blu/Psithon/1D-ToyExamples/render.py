import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import json
import argparse
import os


def render_simulation(sim_dir: Path, output_dir: Path, output_format: str = 'mp4', fps: int = 30, dpi: int = 100):
    h5_path = sim_dir / "simulation_data.h5"
    config_path = sim_dir / "config.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    with h5py.File(h5_path, 'r') as f:
        x = f['x'][:]
        waves = f['waves'][:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Quantum Wave Evolution')

    prob_max = np.max(np.abs(waves)**2)
    wave_max = np.max(np.abs(waves))

    def animate(frame):
        ax1.clear()
        ax2.clear()

        psi = waves[frame]
        prob = np.abs(psi)**2

        ax1.plot(x, psi.real, 'b', label='Real')
        ax1.plot(x, psi.imag, 'r', label='Imaginary')
        ax1.set_ylim(-wave_max, wave_max)
        ax1.legend()
        ax1.set_title(f'Wave Function (Frame {frame})')

        ax2.plot(x, prob, 'k')
        ax2.set_ylim(0, prob_max)
        ax2.set_title('Probability Density')

        return ax1, ax2

    anim = animation.FuncAnimation(fig, animate, frames=len(waves), interval=1000 / fps)

    output_path = output_dir / f"{sim_dir.name}.{output_format}"
    if output_format == 'gif':
        anim.save(output_path, writer='pillow', fps=fps)
    else:
        anim.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi)

    plt.close()


def batch_render(input_dir: str, output_dir: str, output_format: str = 'mp4', fps: int = 30, dpi: int = 100):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    sim_dirs = [d for d in input_path.iterdir() if d.is_dir() and (d / "simulation_data.h5").exists()]

    for i, sim_dir in enumerate(sim_dirs, 1):
        print(f"Rendering simulation {i}/{len(sim_dirs)}: {sim_dir.name}")
        try:
            render_simulation(sim_dir, output_path, output_format, fps, dpi)
        except Exception as e:
            print(f"Error rendering {sim_dir.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing simulation folders")
    parser.add_argument("output_dir", help="Directory for rendered animations")
    parser.add_argument("--format", choices=['mp4', 'gif'], default='mp4')
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=100)

    args = parser.parse_args()
    batch_render(args.input_dir, args.output_dir, args.format, args.fps, args.dpi)
