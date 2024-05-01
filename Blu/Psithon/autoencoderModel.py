# Autoencoder input of 1000x1000
# hybridized convolutional and fourier layers with the goal of the latent space having a high correlation to the spectral and spacial characteristics of the field


# Fourier Layer
class FourierLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features):
        super(FourierLayer, self).__init__()
        # Weight matrix
        self.weights = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        # Applying FFT
        x_fft = torch.fft.fft(torch.complex(x, torch.zeros_like(x)))
        # Element-wise multiplication in the frequency domain
        x_transformed = torch.fft.ifft(x_fft * self.weights).real
        return x_transformed

# Hybrid Encode Decode Layer
# Single architecture for encoding and decoding which combines fourier neural layers and convolutional layers in parallel
class HybridLayer(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kerrnel_size):
        super(HybridLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.fft = FourierLayer(input_channels * kernel_size * kernel_size, output_channels * kernel_size * kernel_size)
        self.output_channels = output_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # Convolution path
        x_conv = self.conv(x)

        # Fourier path
        # First reshape and then apply Fourier layer
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, height * width)
        x_fft = self.fft(x_flat)
        x_fft = x_fft.view(batch_size, self.output_channels, height, width)

        # Combining the outputs
        x_combined = torch.cat((x_conv, x_fft), dim=1)  # Concatenate along the channel dimension
        return x_combined


class HybridAutoencoder(nn.Module):
    def __init__(self):
        super(HybridAutoencoder, self).__init__()
        # Assuming the input is a 2D field of shape (1, 1000, 1000)
        # Encoder
        self.encoder = nn.Sequential(
            HybridLayer(1, 16, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Reducing dimension
            nn.Linear(500*500*32, 256),  # Adjust size accordingly
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 12)  # Compressed representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 500*500*32),  # Adjust size accordingly
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsampling
            HybridLayer(32, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

