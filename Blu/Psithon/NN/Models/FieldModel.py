from abc import ABC, abstractmethod
from torch import nn
from Blu.Psithon.Fields import Field


class FieldModel(ABC,
                 nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predictField(self,
                     )
