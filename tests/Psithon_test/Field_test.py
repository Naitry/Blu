from Blu.Psithon.Fields.Field import Field
import torch


def test_1PsithonField():
    F: Field = Field(name="Test_Field",
                     device=torch.device("cpu"))
    F.addRandomWavePacket()
    F.printField(clear=False)


test_1PsithonField()
