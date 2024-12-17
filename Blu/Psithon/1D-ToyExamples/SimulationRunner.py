import torch
import os
from datetime import datetime
from typing import Dict, Any, List
import json
import h5py
import numpy as np
from QuantumSimulator import QuantumSimulator


class SimulationRunner:
    def __init__(self, base_save_dir: str = "quantum_sim_results"):
        self.base_save_dir = base_save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_random_wave_params(self,
                                    num_packets: int,
                                    x_range: tuple,
                                    p_range: tuple,
                                    sigma_range: tuple) -> List[Dict[str, float]]:
        packets = []
        for _ in range(num_packets):
            packets.append({
                "x0": np.random.uniform(*x_range),
                "p0": np.random.uniform(*p_range),
                "sigma": np.random.uniform(*sigma_range)
            })
        return packets

    def create_multi_packet_wave(self,
                                 simulator: QuantumSimulator,
                                 packet_params: List[Dict[str, float]]) -> torch.Tensor:
        psi = torch.zeros(simulator.nx, dtype=torch.complex64, device=simulator.device)
        for params in packet_params:
            psi += simulator.gaussian_packet(**params)
        norm = torch.sqrt(torch.sum(torch.abs(psi)**2) * simulator.dx)
        return psi / norm

    def run_multiple_simulations(self,
                                 num_simulations: int,
                                 step_method: str,
                                 num_steps: int,
                                 num_frames: int,
                                 sim_params: Dict[str, Any],
                                 wave_params: Dict[str, Any],
                                 num_packets: int = 1) -> List[str]:
        """Run multiple simulations with different random initializations"""
        save_dirs = []
        for sim_num in range(num_simulations):
            print(f"\nStarting simulation {sim_num + 1}/{num_simulations}")
            save_dir = self.run_simulation(
                step_method=step_method,
                num_steps=num_steps,
                num_frames=num_frames,
                sim_params=sim_params,
                wave_params=wave_params,
                num_packets=num_packets
            )
            save_dirs.append(save_dir)
        return save_dirs

    def run_simulation(self,
                       step_method: str,
                       num_steps: int,
                       num_frames: int,
                       sim_params: Dict[str, Any],
                       wave_params: Dict[str, Any],
                       num_packets: int = 1) -> str:
        """
        Run simulation with specified number of saved frames

        Args:
            step_method: Evolution method ('split_operator', 'direct', 'cayley')
            num_steps: Total number of simulation steps
            num_frames: Number of frames to save
            sim_params: Parameters for QuantumSimulator
            wave_params: Parameters for initial wave function
            num_packets: Number of random wave packets
        """
        save_interval = max(1, num_steps // (num_frames - 1))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.base_save_dir, f"{step_method}_sim_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        packet_params = self.generate_random_wave_params(
            num_packets=num_packets,
            x_range=wave_params.get("x_range", (-50, 50)),
            p_range=wave_params.get("p_range", (-0.5, 0.5)),
            sigma_range=wave_params.get("sigma_range", (10.0, 30.0))
        )

        config = {
            "step_method": step_method,
            "num_steps": num_steps,
            "num_frames": num_frames,
            "save_interval": save_interval,
            "sim_params": sim_params,
            "packet_params": packet_params
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        simulator = QuantumSimulator(**sim_params)

        with h5py.File(os.path.join(save_dir, "simulation_data.h5"), "w") as f:
            f.create_dataset("x", data=simulator.x.cpu().numpy())
            waves = f.create_dataset("waves", shape=(num_frames, simulator.nx), dtype=np.complex64)

            psi = self.create_multi_packet_wave(simulator, packet_params)
            waves[0] = psi.cpu().numpy()

            frame_idx = 1
            for step in range(num_steps):
                psi = getattr(simulator, f"step_{step_method}")(psi)

                if (step + 1) % save_interval == 0 and frame_idx < num_frames:
                    waves[frame_idx] = psi.cpu().numpy()
                    frame_idx += 1
                    simulator.plot_state(psi=psi, step=step)
                    print(f"Completed step {step + 1}/{num_steps}")

        return save_dir


if __name__ == "__main__":
    runner = SimulationRunner()

    sim_params = {
        "nx": 2000,
        "dx": 0.1,
        "dt": 0.10,
        "infinite": True,
    }

    wave_params = {
        "x_range": (-50, 50),
        "p_range": (-0.3, 0.3),
        "sigma_range": (1.0, 3.0)
    }

    results_dir = runner.run_simulation(
        step_method="direct",
        num_steps=100000,
        num_frames=1000,  # Will save 100 evenly spaced frames
        sim_params=sim_params,
        wave_params=wave_params,
        num_packets=1
    )
