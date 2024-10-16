from Blu.Psithon.SimulationEngine import SimulationEngine


def main():
    engine: SimulationEngine = SimulationEngine()

    # add the boson field
    engine.U.addField(name="2D_boson_field")

    # Add wave packet
    engine.U.fields[0].addWavePacket(packetSize=100.0,
                                     k=[1.0,
                                        0.0],
                                     position=[50.0,
                                               50.0])

    engine.U.fields[0].printField(clear=False)

    # Run the simulation
    engine.runSimulation(numSteps=int(3e7))


if __name__ == '__main__':
    main()
