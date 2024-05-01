import torch
from torch import optim
from torch.nn import MSELoss, KLDivLoss
import random
from torch.utils.data import DataLoader
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


def train_model(autoencoder, epochs, batch_size, resolution, minParticles, maxParticles, minPacketSize, maxPacketSize, safeRegion):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    mse_loss = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for _ in range(batch_size):  # Generate a batch of new fields
            field_data = generate_random_field(resolution, minParticles, maxParticles, minPacketSize, maxPacketSize, safeRegion)
            field_data = field_data.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension

            optimizer.zero_grad()
            output = autoencoder(field_data)

            loss = mse_loss(output, field_data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Avg Loss: {total_loss / batch_size}')

autoencoder = HybridAutoencoder()
# Example parameters
epochs = 10
batch_size = 10
resolution = 1000
minParticles = 1
maxParticles = 5
minPacketSize = 50
maxPacketSize = 200
safeRegion = 100

train_model(autoencoder, epochs, batch_size, resolution, minParticles, maxParticles, minPacketSize, maxPacketSize, safeRegion)


