import multiprocessing as mp
from Blu.Psithon.Universe import Universe
from Blu.Utils.Hardware import getDevice
import random
import torch


def main():
    # Simulation parameters
    numSimulationRuns: int = 10
    minParticles: int = 100
    maxParticles: int = 150
    maxPacketSize: int = 200
    minPacketSize: int = 70
    resolution: int = 1000
    safeRegion: int = maxPacketSize // 2

    device: torch.device = getDevice()

    for simulationRun in range(numSimulationRuns):
        U: Universe = Universe(spatialDimensions=2,
                               resolution=resolution,
                               scale=3.0e10,
                               speedLimit=3.0e8,
                               dt=1e-6,
                               delta=1e-1,
                               simulationFolderPath="./simulations/",
                               device=device)

        U.addField(name="2D_boson_field")

        # Generate a random number of particles
        numParticles: int = random.randint(minParticles,
                                           maxParticles)

        for _ in range(numParticles):
            packetSize: int = random.randint(minPacketSize,
                                             maxPacketSize)
            position: list[int] = [random.randint(safeRegion,
                                                  resolution - safeRegion),
                                   random.randint(safeRegion,
                                                  resolution - safeRegion)]
            k: list[float] = [random.uniform(-1, 1),
                              random.uniform(-1, 1)]

            # Add wave packet with random parameters
            U.fields[0].addWavePacket(packetSize=packetSize,
                                      k=k,
                                      position=position)

        # Optionally print field information
        U.fields[0].printField(clear=False)

        # Run the simulation
        U.runSimulation(numSteps=int(3e7))


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
