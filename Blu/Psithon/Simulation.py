import torch
from Blu.Psithon.Field import Field


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
            self.fields[field.name] = [Field]
        else:
            self.fields[field.name].append(Field.Tensor)
        pass
