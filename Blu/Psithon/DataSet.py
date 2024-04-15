import torch
from torch.utils.data import Dataset
from Blu.Psithon.Simulation import Simulation


class QuantumFieldDatasetSingle(Dataset):
    def __init__(self, simulation, fieldName):
        self.simulation = simulation
        self.fieldName = fieldName

    def __len__(self):
        return len(self.simulation.fields[self.fieldName]) - 1

    def __getitem__(self, idx):
        current_step = self.simulation.fields[self.fieldName][idx]
        next_step = self.simulation.fields[self.fieldName][idx + 1]

        # Assuming current_step and next_step are complex-valued tensors of shape [height, width]
        current_step_real = current_step.real.unsqueeze(0)  # Adds channel dimension
        current_step_imag = current_step.imag.unsqueeze(0)  # Adds channel dimension
        next_step_real = next_step.real.unsqueeze(0)
        next_step_imag = next_step.imag.unsqueeze(0)

        # Stack along the channel dimension to get [2, height, width] tensors
        current_step = torch.cat((current_step_real, current_step_imag), dim=0)
        next_step = torch.cat((next_step_real, next_step_imag), dim=0)

        return current_step.float(), next_step.float()


class QuantumFieldDataset(Dataset):
    def __init__(self, simulations: list[Simulation], fieldName: str):
        self.simulations = simulations
        self.fieldName = fieldName
        self.indexMap = self._createIndexMap()

    def _createIndexMap(self):
        indexMap = []
        for simIndex, sim in enumerate(self.simulations):
            for timestepIndex in range(sim.numTimeSteps(self.fieldName) - 1):
                indexMap.append((simIndex, timestepIndex))
        return indexMap

    def __len__(self):
        return len(self.indexMap)

    def __getitem__(self, idx):
        simIndex, timestepIndex = self.indexMap[idx]
        current_step = self.simulations[simIndex].fields[self.fieldName][timestepIndex]
        next_step = self.simulations[simIndex].fields[self.fieldName][timestepIndex + 1]

        # Processing steps remain unchanged
        current_step_real = current_step.real.unsqueeze(0)
        current_step_imag = current_step.imag.unsqueeze(0)
        next_step_real = next_step.real.unsqueeze(0)
        next_step_imag = next_step.imag.unsqueeze(0)

        current_step = torch.cat((current_step_real, current_step_imag), dim=0)
        next_step = torch.cat((next_step_real, next_step_imag), dim=0)

        return current_step.float(), next_step.float()
