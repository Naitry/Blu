from Blu.Psithon.model import UNet
from Blu.Psithon.Simulation import Simulation
from Blu.Psithon.DataSet import QuantumFieldDataset, QuantumFieldDatasetSingle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import cm
from PIL import Image
import time


def save_image(tensor, filepath_prefix):
    tensor = tensor.detach().cpu()  # Move tensor back to CPU for processing

    # Adjust for tensors without explicit batch/channel dimensions
    if tensor.dim() == 4:  # [B, C, H, W]
        realField = tensor[:, 0, :, :].numpy()  # First channel for real part
        imagField = tensor[:, 1, :, :].numpy()  # Second channel for imaginary part
    elif tensor.dim() == 3:  # [C, H, W] - single image without batch dim
        realField = tensor[0, :, :].numpy()
        imagField = tensor[1, :, :].numpy()
    elif tensor.dim() == 2:  # [H, W] - single channel, likely real part only
        # If only a 2D tensor is passed, consider it as the real part with no imaginary component
        realField = tensor.numpy()
        imagField = np.zeros_like(realField)  # No imaginary part
    else:
        raise ValueError("Unsupported tensor shape for visualization.")

    absField = np.sqrt(realField**2 + imagField**2)  # Calculate the magnitude for the absolute field

    def array_to_image(arr, cmap=cm.viridis):
        normalized_arr = arr / np.max(arr) if np.max(arr) > 0 else arr
        return Image.fromarray(np.uint8(cmap(normalized_arr) * 255))

    # Convert fields to images using Pillow
    abs_img = array_to_image(absField.squeeze(), cm.viridis)  # Adjust colormap as needed

    abs_img.save(f"{filepath_prefix}_abs.png")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
torch.set_num_threads(int(torch.get_num_threads() * 1.9))
print(f"Training on device: {device}")

simulationDirs = [
    "/mnt/nfs/raid_mount/simulations/Run_24/",
    "/mnt/nfs/raid_mount/simulations/Run_25/",
    "/mnt/nfs/raid_mount/simulations/Run_26/",
    "/mnt/nfs/raid_mount/simulations/Run_27/",
    "/mnt/nfs/raid_mount/simulations/Run_28/",
    "/mnt/nfs/raid_mount/simulations/Run_29/",
    "/mnt/nfs/raid_mount/simulations/Run_30/",
    "/mnt/nfs/raid_mount/simulations/Run_31/",
    "/mnt/nfs/raid_mount/simulations/Run_32/",
    "/mnt/nfs/raid_mount/simulations/Run_33/"
]


sims: list[Simulation] = []
for simDir in simulationDirs:
    sim: Simulation = Simulation()
    sim.loadSimField(simDir, "field_0")
    sims.append(sim)

dataset = QuantumFieldDataset(sims, 'field_0')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)

# Initialize your model (assuming n_channels=2 for complex numbers and n_classes=2 for output)
unet_model = UNet(n_channels=2, n_classes=2)
unet_model = unet_model.to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(unet_model.parameters(), lr=0.001)


def train_model(model, dataloader, optimizer, criterion, epochs=1):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Record the start time of the epoch
        epoch_start_time = time.time()

        for i, (current_step, next_step) in enumerate(dataloader):
            current_step = current_step.to(device)
            next_step = next_step.to(device)

            optimizer.zero_grad()
            output = model(current_step)
            loss = criterion(output, next_step)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate and print the epoch duration
        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds')

        # Saving a checkpoint with more than just the model state_dict
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Add more components as needed
        }
        torch.save(checkpoint, "./unet_checkpoint.pth")
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}')


train_model(unet_model, dataloader, optimizer, criterion, epochs=20)

print("training complete")
