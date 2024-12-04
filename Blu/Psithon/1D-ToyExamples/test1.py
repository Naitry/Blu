import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class QuantumSimulator:
    def __init__(self,
                 nx: int = 10000,
                 dx: float = 0.1,
                 dt: float = 0.01,
                 hbar: float = 1.0,
                 infinite: bool = True,
                 wellWidth: Optional[int] = None):
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nx: int = nx
        self.dx: float = dx
        self.dt: float = dt
        self.hbar: float = hbar
        self.width: float = self.nx * self.dx
        self.x: torch.Tensor = torch.linspace(-self.width / 2, self.width / 2, nx, dtype=torch.complex64).to(self.device)
        self.k: torch.Tensor = 2 * np.pi * torch.fft.fftfreq(nx, dx).to(self.device)
        self.infinite: bool = infinite
        self.wellWidth: int
        self.infiniteWellPotential: torch.Tensor

        # make sure if infinite well is on, the well actually exists
        if self.infinite:
            if ((wellWidth is not None) and (wellWidth >= self.width)) or (wellWidth is None):
                wellWidth = self.width - 10
            self.infiniteWellPotential: torch.Tensor = torch.zeros_like(self.x)
            self.infiniteWellPotential[torch.abs(self.x) > wellWidth / 2] = 1e6

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))

    def gaussian_packet(self,
                        x0: float,
                        p0: float,
                        sigma: float) -> torch.Tensor:
        norm: float = (2 * np.pi * sigma**2)**(-0.25)
        psi: torch.Tensor = norm * torch.exp(-(self.x - x0)**2 / (4 * sigma**2))
        psi *= torch.exp(1j * p0 * self.x / self.hbar)
        return psi

    def kinetic_evolution(self,
                          psi: torch.Tensor):
        psi_k = torch.fft.fft(psi)
        k2 = self.k**2
        psi_k *= torch.exp(-1j * self.hbar * k2 * self.dt / (2))
        return torch.fft.ifft(psi_k)

    def potential_evolution(self,
                            psi: torch.Tensor):
        return psi * torch.exp(-1j * self.infiniteWellPotential * self.dt / self.hbar)

    def step(self,
             psi: torch.Tensor) -> torch.Tensor:
        psi = self.kinetic_evolution(psi)
        psi = self.potential_evolution(psi)
        psi = self.kinetic_evolution(psi)

        psi[torch.abs(self.x) > self.wellWidth / 2] = 0

        # Normalize
        norm: float = torch.sqrt(torch.sum(torch.abs(psi)**2) * self.dx)
        psi /= norm

        return psi

    def plot_state(self,
                   psi: torch.Tensor,
                   step: int) -> None:
        # move the data to the cpu for plotting
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
               steps: int ,
               plot_interval: int = 5000) -> torch.Tensor:
        psi: torch.Tensor = psi.to(self.device)
        psi_history: torch.Tensor = torch.zeros((steps + 1, len(psi)), dtype=torch.complex64)
        psi_history[0] = psi.cpu()

        for i in range(steps):
            psi = self.step(psi)
            psi_history[i + 1] = psi.cpu()
            if i % plot_interval == 0:
                self.plot_state(psi, i)

        return psi_history


# Init Simulator
sim: QuantumSimulator = QuantumSimulator()

# Create an initial field
psi0 = sim.gaussian_packet(x0=-20,
                           p0=0.2,
                           sigma=20.0)

# Evolve the field
evolution: torch.Tensor = sim.evolve(psi0, steps=300000)
