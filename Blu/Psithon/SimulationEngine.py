# blu
from Blu.Psithon.Universe import Universe
from Blu.Psithon.Fields import Field
from Blu.Utils.Hardware import (getDevice,
                                getDeviceList)

# system
import os
import multiprocessing as mp
import time
from datetime import datetime
import copy

# I/O
import pandas as pd

# typing
from typing import (Optional,
                    Union)

# compute
import torch


class SimulationEngine:
    def __init__(self,
                 resolution: int = 1000,
                 spatialDimensions: int = 2,
                 simulationFolderPath: str = './simulations/'):
        mp.set_start_method('spawn')

        # get a list of available devices and set an active device
        self.devices: list[torch.device] = getDeviceList()
        self.activeDevice: torch.device = getDevice()

        # simulation attributes
        self.spatialDimentions: int = spatialDimensions
        self.resolution: int = resolution

        # create a universe which the simulations will take place in
        self.U: Universe = Universe(spatialDimensions=self.spatialDimentions,
                                    resolution=resolution,
                                    scale=3.0e10,
                                    speedLimit=3.0e8,
                                    dt=1e-7,
                                    delta=1e-1,
                                    device=self.activeDevice)

        # simulation variables
        self.simTargetPath: str = simulationFolderPath
        self.catalogPath: str = os.path.join(self.simTargetPath,
                                             "runCatalog.csv")

        self.simStartTime: Optional[float] = None
        self.simStepsPerSecond: float = 0.0
        self.simRunID: Optional[str] = None
        self.simRunPath: Optional[str] = None
        self.simQueue: Optional[mp.Queue] = None
        self.simResultQueue: Optional[mp.Queue] = None

    def addSimRunEntry(self) -> None:
        # create a new entry for the catalog
        newEntry: pd.DataFrame = pd.DataFrame({
            'RunID': [self.simRunID],
            'StartTime': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })

        # CASE: simulation catalog exists
        if os.path.exists(self.catalogPath):
            # load the catalog and append the new entry
            runCatalog: pd.DataFrame = pd.read_csv(self.catalogPath)
            updatedCatalog: pd.DataFrame = pd.concat([runCatalog,
                                                      newEntry],
                                                     ignore_index=True)
        else:
            # create a fresh catalog with just the new entry
            updatedCatalog: pd.DataFrame = newEntry

        # write out the updated catalog
        updatedCatalog.to_csv(self.catalogPath, index=False)

    def recordSimEnd(self) -> None:
        # CASE: simulation catalog exists
        if os.path.exists(self.catalogPath):
            runCatalog: pd.DataFrame = pd.read_csv(self.catalogPath)
            # CASE: Entry for the current run ID exists
            if self.simRunID in runCatalog['RunID'].values:
                # If the column does not exist, it will be added
                runCatalog.loc[runCatalog['RunID'] == self.simRunID, "EndTime"] = datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S')
                runCatalog.to_csv(self.catalogPath,
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
        nextIndex: int
        # CASE: simulation catalog exists
        if os.path.exists(self.catalogPath):
            try:
                # Read the existing run catalog
                runCatalog: pd.DataFrame = pd.read_csv(self.catalogPath)
                # Extract run IDs, assuming the format "Run_X" and X is an integer
                maxIndex: int = runCatalog['RunID'].str.extract('Run_([0-9]+)').astype(int).max().item()
                nextIndex = maxIndex + 1
            except Exception as e:
                print(f"Error reading run catalog: {e}")
                nextIndex = 0  # Default to 0 if any error occurs
        else:
            nextIndex = 1  # Start with 1 if no catalog exists

        self.simRunID = f"Run_{nextIndex}"

    def saveSimulation(self) -> None:
        """
        The saving process which will run in parallel to the main simulation process
        Takes frames of the simulation queue and saves them with field save functions

        :return: none
        """
        try:
            # CASE: still saving
            while True:
                # get a data point from the output queue
                data: Union[str | tuple] = self.simQueue.get()

                # CASE: stop signal received
                if data == "STOP":
                    print("Stop signal received!")
                    return
                else:
                    # output data fields
                    fields: list[Field]
                    entropies: list[float]
                    timestep: int

                    # unpack data from queue
                    fields, entropies, timestep = data

                    print(f"Simulation step {timestep}:")
                    # iterate through each field
                    for i, field in enumerate(fields):
                        # set the file name based on the field
                        filename: str = f"field_{i}.hdf5"
                        print(f"field {i}|| t = {timestep}; entropy = {entropies[i]}")

                        # set filepath for the simulation data
                        filepath = os.path.join(self.simRunPath,
                                                filename)
                        # set filepath for the most recent timestep image
                        imagePath = os.path.join(self.simRunPath,
                                                 "mostRecentTimestep.png")

                        # print the most recent timestep to the console
                        field.printField()

                        # save the most recent timestep as an image
                        field.saveImage(imagePath)

                        # Save the field to an HDF5 file
                        field.saveHDF5(timestep=timestep,
                                       entropy=entropies[i],
                                       filepath=filepath)

        except Exception as e:
            print(f"Error in saving simulation: {e}")
            self.simResultQueue.put("Error")
            return

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
        self.setSimRunID()
        # get the total sim run path
        self.simRunPath = self.simTargetPath + self.simRunID
        # add the sim run to the record
        self.addSimRunEntry()
        # make a new directory for the simulation
        os.makedirs(self.simRunPath,
                    exist_ok=True)

        # calculate total simulation frames
        totalFrames: int = int(fps * simulationLength)
        # calculate interval at which frames should be saved at
        saveInterval: int = numSteps // totalFrames

        initialEntropies: list[float] = []
        entropies: list[float] = []
        cpuFields: list[Field] = []

        # iterate through each field
        for field in self.U.fields:
            # calculate and store initial entropies
            entropy: float = field.calculateEntropy()
            initialEntropies.append(entropy)

        # 2. Multithreading queues
        # Create multiprocessing queues
        self.simQueue: mp.Queue = mp.Queue()
        self.simResultQueue: mp.Queue = mp.Queue()

        # 3. Main sim loop
        # set simulation start time
        self.simStartTime: float = time.time()
        # create the output process
        outputProcess: mp.Process = mp.Process(target=self.saveSimulation)
        # start the output process
        outputProcess.start()
        # iterate through each step in the simulation
        for step in range(int(numSteps)):
            # CASE: step should be saved
            if step % saveInterval == 0:
                # update the steps/second
                self.simStepsPerSecond = step / (time.time() - self.simStartTime)
                # clear the lists
                entropies = []
                cpuFields = []
                # iterate through fields
                for i, field in enumerate(self.U.fields):
                    # calculate and store field entropy
                    entropy: float = field.calculateEntropy()
                    entropies.append(entropy)
                    # deepcopy field and move it to cpu
                    cpuField: Field = copy.deepcopy(field)
                    cpuField.field = cpuField.field.cpu()
                    # store field
                    cpuFields.append(cpuField)
                # add the results to the output queue
                self.simQueue.put((cpuFields,
                                   entropies,
                                   step))
            # update the state of the universe
            self.U.update()
        # record the end of the simulation to the log file
        self.recordSimEnd()
