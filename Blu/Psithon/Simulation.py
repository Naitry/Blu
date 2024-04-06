from __future__ import annotations
import torch
from Blu.Psithon.Field import Field, loadFieldComponentDict


class Simulation():
    def __init__(self,
                 fields: dict[str, list[torch.Tensor]] = None):
        self.fields = dict[str, list[torch.Tensor]]
        if fields is None:
            self.fields = {}
        else:
            self.fields = fields

    def addFieldTimestep(self,
                         field: Field):
        if self.fields[field.name] is None:
            self.fields[field.name] = [Field.Tensor]
        else:
            self.fields[field.name].append(Field.Tensor)
        pass

    def addTensorTimestep(self,
                          fieldName: str,
                          tensor: torch.Tensor):
        if self.fields[fieldName] is None:
            self.fields[fieldName] = [tensor]
        else:
            self.fields[fieldName].append(tensor)
        pass

    def loadSimField(simDir: str,
                     fieldName: str) -> Simulation:
        realData: dict[int, torch.Tensor] = loadFieldComponentDict(filepath=fieldName,
                                                                   prefix="real")
        imaginaryData: dict[int, torch.Tensor] = loadFieldComponentDict(filepath=fieldName,
                                                                        prefix="imaginary")
        S: Simulation = Simulation()

        assert len(realData) == len(imaginaryData), "Real and Imaginary datasets do not have the same length"

        for timestep, realComponent in realData:
            imaginaryComponent = imaginaryData[timestep]

            fieldTensor: torch.Tensor = torch.complex(real=realComponent,
                                                      imag=imaginaryComponent)
            S.addTensorTimestep()
