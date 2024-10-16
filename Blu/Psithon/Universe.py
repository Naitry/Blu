# System
import os

# Typing
from typing import (Optional,
                    Union)

# Data
import pandas as pd

# Compute
import torch

# Blu
from Blu.Psithon.Fields.Field import (Field,
                                      BLU_PSITHON_defaultDataType)
from Blu.Psithon.Particles.ParticleCloud import ParticleCloud


class Universe:
    def __init__(self,
                 device: torch.device,
                 spatialDimensions: int,
                 resolution: int,
                 scale: float,
                 speedLimit: float,
                 dt: float,
                 delta: float,
                 fields: Optional[list[Field]] = None,
                 particles: Optional[list[ParticleCloud]] = None,
                 dtype: torch.dtype = BLU_PSITHON_defaultDataType):
        """
        Universe constructor

        :param spatialDimensions: number of spatial dimensions in the Universe
        :param resolution: the resolution of the fields in each dimension
        :param dt: the regular magnitude of the forward timestep for the universe
        :param delta: the magnitude of the distance between points in the system, heavily effects stability
        :param fields: an optional set of fields, if input this will be the basis of the Universe
        :param dtype: the torch data type of the tensors
        :param device: the torch capable device which the universe will be created on
        :param simulationFolderPath: the path for simulation data to be saved to
        """
        self.spatialDimensions: int = spatialDimensions
        self.resolution: int = resolution
        self.scale: float = scale
        self.c: float = speedLimit
        self.dt: float = dt
        self.delta: float = delta
        self.fields: list[Field] = fields or []
        self.particles: list[ParticleCloud] = particles or []
        self.particles: list[ParticleCloud] = particles or []
        self.fields: list[Field] = fields or []
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

    def addField(self,
                 name: str) -> None:
        """
        add a new field to the Universe

        :param name: the string name of the field

        :return: none
        """
        self.fields.append(Field(name=name,
                                 device=self.device,
                                 dtype=self.dtype,
                                 spatialDimensions=self.spatialDimensions,
                                 resolution=self.resolution))

    def addParticleCloud(self,
                         name: str) -> None:
        """
        add a new field to the Universe

        :param name: the string name of the field

        :return: none
        """
        self.fields.append(ParticleCloud(name=name,
                                         device=self.device,
                                         dtype=self.dtype,
                                         spatialDimensions=self.spatialDimensions,
                                         resolution=self.resolution))

    def update(self,
               dt: Optional[float] = None,
               delta: Optional[float] = None) -> None:

        self.updateParticleClouds(dt=dt or self.dt,
                                  delta=delta or self.delta)
        self.updateFields(dt=dt or self.dt,
                          delta=delta or self.delta)

    def updateFields(self,
                     dt: Optional[float] = None,
                     delta: Optional[float] = None) -> None:
        """
        update each field in the universe

        :param dt: the magnitude of the timestep forward which will be taken
        :param delta: the magnitude of the distance between points in the system, heavily effects stability
        :param device: the device which the calculation will be made on
        this should usually be default, to keep the data on one device

        :return: none
        """
        for field in self.fields:
            field.update(dt=dt or self.dt,
                         delta=delta or self.delta)

    def updateParticleClouds(self,
                             dt: Optional[float] = None,
                             delta: Optional[float] = None) -> None:
        """
        update each field in the universe

        :param dt: the magnitude of the timestep forward which will be taken
        :param delta: the magnitude of the distance between points in the system, heavily effects stability
        :param device: the device which the calculation will be made on
        this should usually be default, to keep the data on one device

        :return: none
        """
        for cloud in self.particles:
            cloud.update(dt=dt or self.dt,
                         delta=delta or self.delta)

    def saveSimulation(self) -> None:
        """
        The saving process which will run in parallel to the main simulation process
        Takes frames of the simulation queue and saves them with field save functions

        :return: none
        """
        try:
            while True:
                data: Union[str | tuple] = self.simQueue.get()
                # CASE: stop signal received
                if data == "STOP":
                    print("Stop signal received!")
                    return
                else:
                    fields: list[Field]
                    entropies: list[float]
                    timestep: int
                    fields, entropies, timestep = data
                    for i, field in enumerate(fields):
                        filename = f"field_{i}.hdf5"
                        print(f"field {i}: t = {timestep}; entropy = {entropies[i]}")
                        filepath = os.path.join(self.simRunPath,
                                                filename)
                        imagePath = os.path.join(self.simRunPath,
                                                 "mostRecentTimestep.png")
                        field.saveImage(imagePath)
                        field.printField(clear=False)

                        # Save the field to an HDF5 file
                        field.saveHDF5(timestep=timestep,
                                       entropy=entropies[i],
                                       filepath=filepath)

        except Exception as e:
            print(f"Error in saving simulation: {e}")
            self.simResultQueue.put("Error")
            return

    def setSimRunID(self) -> None:
        """
        Scans the run catalog file to determine what the run ID should be changed to
        Updates the run ID of the universe when determined

        :return: none
        """
        catalogPath: str = os.path.join(self.simTargetPath,
                                        "runCatalog.csv")
        nextIndex: int
        if os.path.exists(catalogPath):
            try:
                # Read the existing run catalog
                runCatalog: pd.DataFrame = pd.read_csv(catalogPath)
                # Extract run IDs, assuming the format "Run_X" and X is an integer
                maxIndex: int = runCatalog['RunID'].str.extract('Run_([0-9]+)').astype(int).max().item()
                nextIndex = maxIndex + 1
            except Exception as e:
                print(f"Error reading run catalog: {e}")
                nextIndex = 0  # Default to 0 if any error occurs
        else:
            nextIndex = 1  # Start with 1 if no catalog exists

        self.simRunID = f"Run_{nextIndex}"
