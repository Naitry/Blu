from Blu.Psithon.SimulationEngine import SimulationEngine


def main():
    engine: SimulationEngine = SimulationEngine(resolution=1000)

    # add the boson field
    engine.U.addField(name="2D_boson_field")

    # Add wave packet
    [engine.U.fields[0].addRandomWavePacket() for _ in range(2)]

    # Run the simulation
    engine.runSimulation(numSteps=int(3e7))


if __name__ == '__main__':
    main()
