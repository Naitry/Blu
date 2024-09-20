import csv
import torch
from torch import optim
from torch.nn import MSELoss, KLDivLoss
from torch.nn import functional as F
import numpy as np
from matplotlib import cm
from PIL import Image
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from Blu.Psithon.Universe import Universe
from Blu.Psithon.Field import Field
from Blu.Psithon.autoencoderModel import HybridAutoencoder



def generate_random_field(resolution, minParticles, maxParticles, minPacketSize, maxPacketSize, safeRegion):
    """Generates a random field with wave packets at random positions and wave vectors."""
    field = Field(name="random_field", resolution=resolution)

    numParticles = random.randint(minParticles, maxParticles)
    for _ in range(numParticles):
        packetSize = random.randint(minPacketSize, maxPacketSize)
        position = [
            random.randint(safeRegion, resolution - safeRegion),
            random.randint(safeRegion, resolution - safeRegion)
        ]
        k = [random.uniform(-1, 1), random.uniform(-1, 1)]

        field.addWavePacket(packetSize=packetSize, k=k, position=position)

    return field.tensor


def complex_mse_loss(output, target):
    return torch.sum((output.real - target.real) ** 2 + (output.imag - target.imag) ** 2)

def custom_loss(output, target, alpha=0.1):
    # MSE loss for direct comparison
    mse_loss = complex_mse_loss(output, target)

    # Additional penalty for multiple high-intensity spots
    # Assuming 'output' is complex and we're interested in its magnitude
    output_mag = torch.sqrt(output.real**2 + output.imag**2)
    target_mag = torch.sqrt(target.real**2 + target.imag**2)

    # Define a threshold to detect high-intensity spots
    threshold = 0.5 * torch.max(target_mag)
    output_spots = torch.where(output_mag > threshold, 1, 0)
    target_spots = torch.where(target_mag > threshold, 1, 0)

    # Calculate the penalty for extra spots
    extra_spots = torch.abs(output_spots.sum() - target_spots.sum())
    spots_penalty = extra_spots * alpha

    return mse_loss + spots_penalty

def complex_ssim(output: torch.Tensor, target: torch.Tensor, val_range: float = 1) -> torch.Tensor:
    """Compute the mean structural similarity index between two complex images."""
    from pytorch_msssim import ssim  # Ensure you have pytorch-msssim installed

    # Calculate SSIM for real and imaginary parts separately
    real_ssim = ssim(output.real.unsqueeze(1), target.real.unsqueeze(1), data_range=val_range)
    imag_ssim = ssim(output.imag.unsqueeze(1), target.imag.unsqueeze(1), data_range=val_range)

    # Combine the SSIM scores
    return 0.5 * (real_ssim + imag_ssim)

def dynamic_threshold_spots(output_mag: torch.Tensor, target_mag: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    # Dynamic threshold based on the mean and std of the target magnitude
    mean_val, std_val = target_mag.mean(), target_mag.std()
    threshold = mean_val + alpha * std_val

    output_spots = (output_mag > threshold).float()
    target_spots = (target_mag > threshold).float()

    # Calculate the penalty for incorrect spots (both excess and missing)
    spot_penalty = F.mse_loss(output_spots, target_spots)

    return spot_penalty

def enhanced_loss(output: torch.Tensor, target: torch.Tensor, alpha: float = 0.5, beta: float = 0.5) -> torch.Tensor:
    # Basic complex MSE loss
    mse_loss = torch.mean((output.real - target.real) ** 2 + (output.imag - target.imag) ** 2)

    # Compute SSIM
    ssim_loss = complex_ssim(output, target)

    # Spot penalty
    output_mag = torch.sqrt(output.real**2 + output.imag**2)
    target_mag = torch.sqrt(target.real**2 + target.imag**2)
    spot_penalty = dynamic_threshold_spots(output_mag, target_mag)

    # Combine losses with weights to balance their contributions
    total_loss = alpha * mse_loss + (1 - alpha) * ssim_loss + beta * spot_penalty

    return total_loss


def generateField(autoencoder, device: str):
    """
    Generate a field of frames from an autoencoder given the number of samples and latent space dimensions.

    Args:
        autoencoder: The autoencoder model which includes a decoder.
        numSamples: The number of latent space samples to generate.
        latentChannels: The number of channels in the latent space (e.g., 2 for two channels).
        device: The device type ('cpu' or 'cuda') where the tensors should be processed.

    Returns:
        torch.Tensor: The decoded frames from the autoencoder.
    """
    numSamples = 1;
    # Define the latent space dimensions
    latentHeight, latentWidth = 250, 250
    # Generate random latent vectors with the shape (numSamples, latentChannels, latentHeight, latentWidth)
    latentVectors = torch.randn(numSamples, autoencoder.BC, latentHeight, latentWidth).to(device)
    # Decode the latent vectors using the autoencoder's decoder
    fieldFrame = autoencoder.decoder(latentVectors).squeeze(0).squeeze(0)
    field = Field(name="GeneratedField", field=fieldFrame, resolution=resolution, device=device)
    return field

def interpolate_fields(autoencoder, start_vector, end_vector, steps, device, resolution):
    start_vector = start_vector.to(device)
    end_vector = end_vector.to(device)
    t_values = torch.linspace(0, 1, steps).to(device)
    vectors = [start_vector + (end_vector - start_vector) * t for t in t_values]

    fields = []
    for v in vectors:
        decoded = autoencoder.decoder(v).detach()

        # Ensure the output has correct dimensions
        decoded = decoded.squeeze()  # Remove unnecessary dimensions
        if decoded.ndim > 2:
            decoded = decoded.squeeze(0)  # Adjust this based on your specific dimensionality

        # Verify final shape and type before creating Field

        # Create a Field object and attempt to save an image
        field = Field(name="GeneratedField", field=decoded, resolution=resolution, device=device)
        fields.append(field)
        field.saveImage(filepath=f"./output_{fields.index(field)}.png")

    return fields




def train_model(autoencoder, i, epochs, batch_size, resolution, minParticles, maxParticles, minPacketSize, maxPacketSize, safeRegion, device):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
    startVector = torch.randn(5, device=device)  # Start vector for interpolation
    endVector = torch.randn(5, device=device)  # End vector for interpolation
    testFieldA = Field(name="testField",
                      field=generate_random_field(resolution, minParticles, maxParticles, minPacketSize, maxPacketSize, safeRegion),
                      resolution=resolution,
                      device=device)
    testFieldB = Field(name="testField",
                      field=generate_random_field(resolution, minParticles, maxParticles, minPacketSize, maxPacketSize, safeRegion),
                      resolution=resolution,
                      device=device)

    testFieldA.saveImage(filepath=f"./testA.png")
    testFieldB.saveImage(filepath=f"./testB.png")

    with open(f'training_{i}Channel.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average Loss'])

        for epoch in range(epochs):
            total_loss = 0
            for _ in range(batch_size):
                field_data = generate_random_field(resolution, minParticles, maxParticles, minPacketSize, maxPacketSize, safeRegion)
                field_data = field_data.unsqueeze(0).unsqueeze(0).to(device)
                optimizer.zero_grad()
                output = autoencoder(field_data)
                loss = custom_loss(output, field_data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            average_loss = total_loss / batch_size
            print(f'Epoch {epoch+1}, Avg Loss: {average_loss}')
            writer.writerow([epoch + 1, average_loss])  # Log epoch and average loss to CSV


            if (epoch % 50 == 0):
                startVector = autoencoder.encoder(testFieldA.tensor.unsqueeze(0).unsqueeze(0).to(device))
                endVector = autoencoder.encoder(testFieldB.tensor.unsqueeze(0).unsqueeze(0).to(device))
                guess_fields = interpolate_fields(autoencoder, startVector, endVector, 5, device, resolution)
                guessA = Field(name="guessA", field=autoencoder.forward(testFieldA.tensor.unsqueeze(0).unsqueeze(0).to(device)).squeeze(0).squeeze(0))
                guessB = Field(name="guessB", field=autoencoder.forward(testFieldB.tensor.unsqueeze(0).unsqueeze(0).to(device)).squeeze(0).squeeze(0))
                random = generateField(autoencoder, device)
                guessA.saveImage(filepath=f"./guessA.png")
                guessB.saveImage(filepath=f"./guessB.png")
                random.saveImage(filepath=f"./testDecode.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(16, 17):
    autoencoder = HybridAutoencoder(i).to(device)

    # Example parameters
    epochs = 2000
    batch_size = 100
    resolution = 1000
    minParticles = 1
    maxParticles = 1
    minPacketSize = 100
    maxPacketSize = 200
    safeRegion = 100

    train_model(autoencoder, i, epochs, batch_size, resolution, minParticles, maxParticles, minPacketSize, maxPacketSize, safeRegion, device)


