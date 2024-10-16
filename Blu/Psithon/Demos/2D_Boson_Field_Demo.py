from Blu.Psithon.Universe import Universe
from Blu.Utils.Hardware import getDevice
import torch


def main():
    # constants
    resolution: int = 1000
    packetSize: int = 1
    packetPosition: list[int] = [0, 0]
    k: list[float] = [1, 0]
    device: torch.device = getDevice()

    # initialize the universe
    U: Universe = Universe(spatialDimensions=2,
                           resolution=resolution,
                           scale=3.0e10,
                           speedLimit=3.0e8,
                           dt=1e-6,
                           delta=1e-1,
                           simulationFolderPath="./simulations/",
                           device=device)

    # add the boson field
    U.addField(name="2D_boson_field")

    # Add wave packet
    U.fields[0].addWavePacket(packetSize=packetSize,
                              k=k,
                              position=packetPosition)

    U.fields[0].printField(clear=False)

    # Run the simulation
    U.runSimulation(numSteps=int(3e7))


if __name__ == '__main__':
    main()
