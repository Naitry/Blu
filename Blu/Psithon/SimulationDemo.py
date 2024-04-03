import multiprocessing as mp
from Blu.Psithon.Universe import Universe


def main():
    U: Universe = Universe(spatialDimensions=2,
                           resolution=1000,
                           dt=1e-6,
                           delta=1e-1,
                           simulationFolderPath="/mnt/nfs/simulations/")

    U.addField(name="2D_boson_field")
    U.fields[0].addWavePacket(packetSize=500,
                              k=[-0.5, -0.5],
                              position=[750, 750])
    U.fields[0].addWavePacket(packetSize=500,
                              k=[0.5, 0.5],
                              position=[250, 250])
    U.fields[0].printField(clear=False)
    U.runSimulation(numSteps=int(3e7))


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
