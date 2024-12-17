from SimulationRunner import SimulationRunner

if __name__ == "__main__":
    runner = SimulationRunner()

    sim_params = {
        "nx": 2000,
        "dx": 0.1,
        "dt": 0.001,
        "infinite": True,
    }

    wave_params = {
        "x_range": (-50, 50),
        "p_range": (-2.0, 2.0),
        "sigma_range": (1.0, 3.0)
    }

    save_dirs = runner.run_multiple_simulations(
        num_simulations=200,
        step_method="cayley",
        num_steps=100000,
        num_frames=1000,
        sim_params=sim_params,
        wave_params=wave_params,
        num_packets=3
    )
