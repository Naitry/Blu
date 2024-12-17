import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class QuantumSimulator:
    def __init__(self,
                 nx: int = 2000,
                 dx: float = 0.1,
                 dt: float = 10.0,
                 hbar: float = 1.0,
                 infinite: bool = True,
                 wellWidth: Optional[float] = None):
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nx: int = nx
        self.dx: float = dx
        self.dt: float = dt
        self.hbar: float = hbar
        self.width: float = self.nx * self.dx
        self.x: torch.Tensor = torch.linspace(-self.width / 2, self.width / 2, nx, dtype=torch.complex64).to(self.device)
        self.k: torch.Tensor = 2 * np.pi * torch.fft.fftfreq(nx, dx).to(self.device)
        self.infinite: bool = infinite

        # Set default well width
        if wellWidth is None:
            self.wellWidth = float(self.width - 10)
        else:
            self.wellWidth = float(min(wellWidth, self.width - 10))

        # Initialize potential
        self.infiniteWellPotential = torch.zeros(nx, dtype=torch.complex64, device=self.device)
        if self.infinite:
            mask = torch.abs(self.x) > (self.wellWidth / 2)
            self.infiniteWellPotential[mask] = 1e6

        # Create Laplacian operator matrix for direct evolution
        self.laplacian = self._create_laplacian()

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))

    def _create_laplacian(self) -> torch.Tensor:
        """Create the Laplacian operator matrix using finite differences."""
        diagonals = torch.ones(self.nx - 1, dtype=torch.complex64, device=self.device)
        laplacian = torch.diag(-2 * torch.ones(self.nx, dtype=torch.complex64, device=self.device)) + \
            torch.diag(diagonals, 1) + \
            torch.diag(diagonals, -1)
        return laplacian / (self.dx ** 2)

    def gaussian_packet(self,
                        x0: float,
                        p0: float,
                        sigma: float) -> torch.Tensor:
        norm: float = (2 * np.pi * sigma**2)**(-0.25)
        psi: torch.Tensor = norm * torch.exp(-(self.x - x0)**2 / (4 * sigma**2))
        psi *= torch.exp(1j * p0 * self.x / self.hbar)
        return psi

    def kinetic_evolution(self, psi: torch.Tensor) -> torch.Tensor:
        """Improved kinetic energy evolution in momentum space"""
        psi_k = torch.fft.fft(psi)
        # The factor of 1/2 here is crucial - it's the m in p²/2m
        k2 = self.k**2
        exp_factor = torch.exp(-1j * self.hbar * k2 * self.dt / (4))  # Note the factor of 4
        return torch.fft.ifft(exp_factor * psi_k)

    def potential_evolution(self, psi: torch.Tensor) -> torch.Tensor:
        """Improved potential energy evolution"""
        # Use a smoother potential for better numerics
        V = self.infiniteWellPotential
        # The full time step for potential (not half)
        exp_factor = torch.exp(-1j * V * self.dt / self.hbar)
        return psi * exp_factor

    def step_split_operator(self, psi: torch.Tensor) -> torch.Tensor:
        """Improved split-operator method with proper ordering"""
        # First kinetic half-step
        psi = self.kinetic_evolution(psi)

        # Full potential step
        psi = self.potential_evolution(psi)

        # Second kinetic half-step
        psi = self.kinetic_evolution(psi)

        if self.infinite:
            mask = torch.abs(self.x) > (self.wellWidth / 2)
            psi[mask] = 0

        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(psi)**2) * self.dx)
        psi /= norm

        return psi

    def step_direct(self,
                    psi: torch.Tensor) -> torch.Tensor:
        """Direct evolution using the Schrödinger equation."""
        # Ensure psi is in the correct shape for matrix multiplication
        if len(psi.shape) == 1:
            psi = psi.reshape(-1, 1)

        kinetic = -0.5 * self.hbar * self.hbar * torch.matmul(self.laplacian, psi)
        potential = (self.infiniteWellPotential.reshape(-1, 1) * psi)
        dpsi_dt = -1j * (kinetic + potential) / self.hbar
        psi_new = psi + dpsi_dt * self.dt

        if self.infinite:
            mask = torch.abs(self.x) > (self.wellWidth / 2)
            psi_new[mask] = 0

        # Reshape back to 1D
        psi_new = psi_new.reshape(-1)

        # Normalize
        norm: float = torch.sqrt(torch.sum(torch.abs(psi_new)**2) * self.dx)
        psi_new /= norm

        return psi_new

    def step_cayley(self,
                    psi: torch.Tensor) -> torch.Tensor:
        """Evolution using Cayley's form from equation 11.154"""
        if len(psi.shape) == 1:
            psi = psi.reshape(-1, 1)

        # Form H = -ℏ²/2m ∇² + V
        H = -0.5 * self.hbar * self.hbar * self.laplacian + torch.diag(self.infiniteWellPotential)

        # Define α = Δt/(2ℏ)
        alpha = self.dt / (2 * self.hbar)

        # Form (1 ± iαH)
        I = torch.eye(self.nx,
                      dtype=torch.complex64,
                      device=self.device)
        A = I + 1j * alpha * H  # Left side of eq 11.154
        B = I - 1j * alpha * H  # Right side of eq 11.154

        # Solve (1 + iαH)ψ(t + Δt) = (1 - iαH)ψ(t)
        b = torch.matmul(B, psi)
        psi_new = torch.linalg.solve(A, b)

        psi_new = psi_new.reshape(-1)

        if self.infinite:
            mask = torch.abs(self.x) > (self.wellWidth / 2)
            psi_new[mask] = 0

        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(psi_new)**2) * self.dx)
        psi_new /= norm

        return psi_new

    def plot_state(self,
                   psi: torch.Tensor,
                   step: int) -> None:
        psi_cpu = psi.cpu()
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(self.x.cpu().real.numpy(), psi_cpu.real.numpy(), 'b', label='Real')
        self.ax1.plot(self.x.cpu().real.numpy(), psi_cpu.imag.numpy(), 'r', label='Imaginary')
        self.ax1.legend()
        self.ax1.set_title(f'Wave Function (Step {step})')

        prob = torch.abs(psi_cpu)**2
        self.ax2.plot(self.x.cpu().real.numpy(), prob.numpy(), 'k')
        self.ax2.set_title('Probability Density')
        plt.pause(0.01)

    def evolve(self,
               psi: torch.Tensor,
               steps: int,
               method: str = 'split_operator',
               plot_interval: int = 5) -> torch.Tensor:
        """
        Evolve the wave function using the specified method.
        """
        psi = psi.to(self.device)
        psi_history = torch.zeros((steps + 1, len(psi)), dtype=torch.complex64)
        psi_history[0] = psi.cpu()

        step_methods = {
            'split_operator': self.step_split_operator,
            'direct': self.step_direct,
            'cayley': self.step_cayley
        }

        if method not in step_methods:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(step_methods.keys())}")

        step_func = step_methods[method]

        for i in range(steps):
            psi = step_func(psi)
            psi_history[i + 1] = psi.cpu()
            if i % plot_interval == 0:
                self.plot_state(psi, i)

        return psi_history


if __name__ == "__main__":
    # Init Simulator
    sim = QuantumSimulator()

    # Create an initial field
    psi0 = sim.gaussian_packet(x0=-20, p0=0.2, sigma=20.0)

    # Compare different evolution methods
    methods = ['cayley']
    for method in methods:
        print(f"\nEvolving using {method} method...")
        evolution = sim.evolve(psi0, steps=1000000, method=method)
        plt.show()
