# Multithreading
import multiprocessing as mp

# System
import os
import copy

# Typing
from typing import Optional, \
    Union

# Data
import pandas as pd

# Compute
import torch

# Blu
from Blu.Psithon.Field import Field, \
    BLU_PSITHON_defaultDimensions, \
    BLU_PSITHON_defaultDataType, \
    BLU_PSITHON_defaultResolution
from Blu.Psithon.GaussianWavePacket import GaussianWavePacket as GWavePacket

# Time
import time
from datetime import datetime


class Universe:

    def __init__(self,
                 spatialDimensions: int,
                 resolution: int,
                 dt: float,
                 delta: float,
                 fields: Optional[list[Field]] = None,
                 dtype: torch.dtype = BLU_PSITHON_defaultDataType,
                 device: torch.device = torch.device('cuda'),
                 simulationFolderPath: str = '/mnt/nfs/simulations/'):
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
        self.dt: float = dt
        self.delta: float = delta
        self.fields: list[Field] = fields
        self.fields = fields or []
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device
        # simulation variables
        self.simTargetPath: str = simulationFolderPath

        self.simStartTime: Optional[float] = None
        self.simRunID: Optional[str] = None
        self.simRunPath: Optional[str] = None
        self.simQueue: Optional[mp.Queue] = None
        self.simResultQueue: Optional[mp.Queue] = None

    def addField(self,
                 name: str) -> None:
        """
        add a new field to the Universe

        :param name: the string name of the field

        :return: none
        """
        self.fields.append(Field(name=name,
                                 spatialDimensions=2,
                                 resolution=self.resolution,
                                 dtype=self.dtype,
                                 device=self.device))

    def update(self,
               dt: Optional[float] = None,
               delta: Optional[float] = None,
               device: Optional[torch.device] = None) -> None:
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
                         delta=delta or self.delta,
                         device=device or self.device)

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
                    print(f"{timestep} {entropies[0]}")
                    for i, field in enumerate(fields):
                        filename = f"field_{i}.hdf5"
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

    def addSimRunEntry(self) -> None:
        catalogPath: str = os.path.join(self.simTargetPath, "runCatalog.csv")
        newEntry: pd.DataFrame = pd.DataFrame({
            'RunID': [self.simRunID],
            'StartTime': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        if os.path.exists(catalogPath):
            runCatalog: pd.DataFrame = pd.read_csv(catalogPath)
            updatedCatalog: pd.DataFrame = pd.concat([runCatalog, newEntry], ignore_index=True)
        else:
            updatedCatalog: pd.DataFrame = newEntry
        updatedCatalog.to_csv(catalogPath, index=False)

    def recordSimEnd(self) -> None:
        catalogPath: str = os.path.join(self.simTargetPath,
                                        "runCatalog.csv")
        if os.path.exists(catalogPath):
            runCatalog: pd.DataFrame = pd.read_csv(catalogPath)
            if self.simRunID in runCatalog['RunID'].values:
                # If the column does not exist, it will be added
                runCatalog.loc[runCatalog['RunID'] == self.simRunID, "EndTime"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                runCatalog.to_csv(catalogPath,
                                  index=False)
                print(f"Recorded end time for {self.simRunID}.")
            else:
                print(f"No entry found for {self.simRunID}.")
        else:
            print("Catalog does not exist.")

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

    def runSimulation(self,
                      numSteps: int = 1e7,
                      fps: int = 60,
                      simulationLength: float = 10.0) -> None:
        """
        the main simulation process for the universe
        proceeds through a for loop, simulating each step and placing data into a queue
        only places a fraction of the frames actually simulated into the queue
        this fraction is calculated based on fps, sim length, and number of sim steps

        :param numSteps: number of iterations which the simulation will take
        :param fps: number of iterations per second which will be saved
        :param simulationLength: length in seconds of the simulation

        :return: none
        """
        # 1. Init
        # set run ID
        print(1)
        self.setSimRunID()
        self.simRunPath = self.simTargetPath + self.simRunID
        self.addSimRunEntry()
        os.makedirs(self.simRunPath,
                    exist_ok=True)
        totalFrames: int = int(fps * simulationLength)
        saveInterval: int = numSteps // totalFrames

        entropies: list[float] = []

        # 2. Multithreading queues
        # Create multiprocessing queues
        print(2)
        self.simQueue = mp.Queue()
        self.simResultQueue = mp.Queue()

        # 3. Main sim loop
        print(3)
        self.simStartTime: float = time.time()
        outputProcess: mp.Process = mp.Process(target=self.saveSimulation)
        outputProcess.start()
        # iterate through each step in the simulation
        for step in range(int(numSteps)):
            # CASE: step should be saved
            if step % saveInterval == 0:
                cpuFields: list[Field] = []
                # iterate through fields and calculate entropies
                for field in self.fields:
                    entropy = field.calculateEntropy()
                    print(f"e {entropy}")
                    entropies.append(entropy)
                    cpuField = copy.deepcopy(field)
                    cpuField.tensor = cpuField.tensor.cpu()
                    cpuFields.append(cpuField)
                self.simQueue.put((cpuFields,
                                   entropies,
                                   step))
            self.update()

        self.recordSimEnd()
