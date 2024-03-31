from Blu.Psithon.Field import Field

def test_1PsithonField():
    F: Field = Field(name="Test_Field")
    F.addWavePacket(packetSize=500,
                    k=[0.5,0.5])
    F.printField(clear=False)

