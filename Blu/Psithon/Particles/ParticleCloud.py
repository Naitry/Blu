import torch


class ParticleCloud:
    def __init__(self,
                 name: str):
        self.name: str = name
        self.particleCount: int = 0
        self.particles: torch.Tensor
