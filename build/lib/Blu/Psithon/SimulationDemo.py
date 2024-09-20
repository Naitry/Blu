import multiprocessing as mp
from Blu.Psithon.Universe import Universe
import random


def main():
    # Simulation parameters
    numSimulationRuns: int = 10  # Example: Loop the simulation 10 times
    minParticles: int = 100
    maxParticles: int = 150
    maxPacketSize: int = 200  # Maximum size for a wave packet
    minPacketSize: int = 70
    resolution: int = 1000  # Resolution of the universe
    safeRegion: int = maxPacketSize // 2

    for simulationRun in range(numSimulationRuns):
        U: Universe = Universe(spatialDimensions=2,
                               resolution=resolution,
                               dt=1e-6,
                               delta=1e-1,
                               simulationFolderPath="/mnt/nfs/raid_mount/simulations/")

        U.addField(name="2D_boson_field")

        # Generate a random number of particles
        numParticles: int = random.randint(minParticles, maxParticles)

        for _ in range(numParticles):
            packetSize: int = random.randint(minPacketSize, maxPacketSize)  # Randomize packet size
            position: list[int] = [random.randint(safeRegion, resolution - safeRegion), random.randint(safeRegion, resolution - safeRegion)]
            k: list[float] = [random.uniform(-1, 1), random.uniform(-1, 1)]  # Randomize k vector

            # Add wave packet with random parameters
            U.fields[0].addWavePacket(packetSize=packetSize,
                                      k=k,
                                      position=position)

        # Optionally print field information
        U.fields[0].printField(clear=False)

        # Run the simulation
        U.runSimulation(numSteps=int(3e7))

        # Here you can save or process the results of the simulation before the next run starts


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
