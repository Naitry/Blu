import multiprocessing as mp
from Blu.Psithon.Universe import Universe


def main():
    U: Universe = Universe(spatialDimensions=2,
                           resolution=1000,
                           dt=1e-7,
                           delta=1e-1)

    U.addField(name="2D_boson_field")
    U.fields[0].addWavePacket(packetSize=500,
                              k=[0.5, 0.5])
    U.fields[0].printField(clear=False)
    U.runSimulation()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
