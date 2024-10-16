from SimulationEngine import SimulationEngine


def main():
    engine: SimulationEngine = SimulationEngine()

    # constants
    resolution: int = 1000
    packetSize: int = 1
    packetPosition: list[int] = [0, 0]
    k: list[float] = [1, 0]

    # add the boson field
    engine.U.addField(name="2D_boson_field")

    # Add wave packet
    engine.U.fields[0].addWavePacket(packetSize=packetSize,
                                     k=k,
                                     position=packetPosition)

    engine.U.fields[0].printField(clear=False)

    # Run the simulation
    engine.runSimulation(numSteps=int(3e7))


if __name__ == '__main__':
    main()
