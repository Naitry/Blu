from Blu.Psithon.model import UNet
from Blu.Psithon.Simulation import Simulation
from Blu.Psithon.Field import Field
import torch
import torch.optim as optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet_model: UNet = UNet(n_channels=2, n_classes=2)
optimizer = optim.Adam(unet_model.parameters(), lr=0.001)

# Loading the checkpoint
checkpoint = torch.load("/home/naitry/Dev/Blu/Blu/Psithon/unet_checkpoint.pth")
unet_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

fieldName: str = "field_0"
unet_model = unet_model.to(device)
sim: Simulation = Simulation()

sim.loadSimField("/mnt/nfs/raid_mount/simulations/Run_30/", "field_0")


# Assuming initial data is complex
initial_step = sim.fields[fieldName][0].tensor  # first timestep, for example

generatedSim: Simulation = Simulation()

current_step = initial_step

print(type(current_step))
print(current_step.is_complex())

f: Field = Field(name=fieldName,
                 field=current_step,
                 spatialDimensions=len(current_step.shape),
                 resolution=current_step.shape[0],
                 dtype=current_step.dtype,
                 device=current_step.device)

# Save the current output as images
generatedSim.addFieldTimestep(f=f)

startTime = time.time()

for i in range(1, 600):  # Perform 10 forward predictions

    current_step_real = current_step.real.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    current_step_imag = current_step.imag.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    current_step = torch.cat([current_step_real, current_step_imag], dim=1).to(device)  # [1, 2, H, W]

    # Perform the forward pass
    with torch.no_grad():
        output = unet_model(current_step)

    # Prepare output for the next iteration
    current_step = output.detach()  # Detach from the current computation graph

    current_step_real = current_step[:, 0, :, :].squeeze(0)
    current_step_imag = current_step[:, 1, :, :].squeeze(0)
    current_step = torch.complex(real=current_step_real,
                                 imag=current_step_imag)
    f: Field = Field(name=fieldName,
                     field=current_step,
                     spatialDimensions=len(current_step.shape),
                     resolution=current_step.shape[0],
                     dtype=current_step.dtype,
                     device=current_step.device)

    f.printField(clear=False)

    # Save the current output as images
    generatedSim.addFieldTimestep(f=f)

duration = time.time() - startTime
print("U-Net simulation complete in ", duration, " seconds")
generatedSim.saveFieldToVideo(fieldName=fieldName,
                              outputFilePath="./simRender.mp4",
                              fps=30)
