from Universe import Universe
from Fields import Field
import pandas as pd
import datetime
import os
import multiprocessing as mp
import time
import copy
from typing import (Optional,
                    Union)


class SimulationEngine:
    def __init__(self,
                 simulationFolderPath: str = '/mnt/nfs/simulations/'):
        self.U = Universe()
        # simulation variables
        self.simTargetPath: str = simulationFolderPath
        self.simStartTime: Optional[float] = None
        self.simRunID: Optional[str] = None
        self.simRunPath: Optional[str] = None
        self.simQueue: Optional[mp.Queue] = None
        self.simResultQueue: Optional[mp.Queue] = None

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
                runCatalog.loc[runCatalog['RunID'] == self.simRunID, "EndTime"] = datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S')
                runCatalog.to_csv(catalogPath,
                                  index=False)
                print(f"Recorded end time for {self.simRunID}.")
            else:
                print(f"No entry found for {self.simRunID}.")
        else:
            print("Catalog does not exist.")

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
        for field in self.fields:
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
                # clear the lists
                entropies = []
                cpuFields = []
                # iterate through fields and calculate entropies
                for i, field in enumerate(self.fields):
                    entropy = field.calculateEntropy()
                    print(f"Initial entropy: {initialEntropies[i]}")
                    entropies.append(entropy)
                    cpuField = copy.deepcopy(field)
                    cpuField.field = cpuField.field.cpu()
                    cpuFields.append(cpuField)
                self.simQueue.put((cpuFields,
                                   entropies,
                                   step))
            self.update(dt=self.dt,
                        delta=self.delta)

        self.recordSimEnd()

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
