# Compute
import torch

# Blu
from Blu.Psithon.DefaultDefinitions import (BLU_PSITHON_defaultDataType)


class ParticleCloud:
    def __init__(self,
                 name: str,
                 device: torch.device,
                 dtype: torch.dtype = BLU_PSITHON_defaultDataType):
        self.name: str = name

        self.particles: torch.Tensor = torch.tensor([],
                                                    dtype=dtype,
                                                    device=device)
        # main object: a tensor of shape {numParticles, {2}, spatialDimensions}
        self.particleCount: int = 0
        self.spatialDimensions: int = 3
        self.dimensions: int = self.spatialDimensions + 1
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device
        # interaction values
        self.interactions: dict[str, torch.Tensor] = {}

    def addParticle(self,
                    position: torch.Tensor = None,
                    velocity: torch.Tensor = None) -> None:
        if position.shape is not [self.spatialDimensions]:
            print(f"Cannot use a position shape of {position.shape}, this particle cloud requires {[1, 1, self.spatialDimensions]}")
            return
        if velocity.shape is not [self.spatialDimensions]:
            print(f"Cannot use a velocity shape of {position.shape}, this particle cloud requires {[1, 1, self.spatialDimensions]}")
            return
        position = position or ((2 * torch.rand(size=(self.spatialDimensions),
                                                dtype=self.dtype,
                                                device=self.device)) - 1)
        velocity = velocity or ((2 * torch.rand(size=(self.spatialDimensions),
                                                dtype=self.dtype,
                                                device=self.device)) - 1)

        newParticle: torch.Tensor = torch.stack((position, velocity))
        self.particles = torch.cat([self.particles, newParticle.unsqueez(0)], dim=-0)

    def update(self,
               fields: torch.Tensor,
               coupling_matrixes: dict[str, torch.Tensor]):
        pass

    def setParticleCount(self):
        self.particleCount = self.particles.shape[0]
