# autoencoder input of 1000x1000
# hybridized convolutional and fourier layers with the goal of the latent space having a high correlation to the spectral and spacial characteristics of the field
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.real_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.imag_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.real_bias = nn.Parameter(torch.Tensor(out_features))
        self.imag_bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.real_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.imag_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.real_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.real_bias, -bound, bound)
        nn.init.uniform_(self.imag_bias, -bound, bound)

    def forward(self, input):
        # Check if input is complex, convert if not
        if not input.is_complex():
            input = torch.complex(input, torch.zeros_like(input))

        real = F.linear(input.real, self.real_weight, self.real_bias) - F.linear(input.imag, self.imag_weight)
        imag = F.linear(input.real, self.imag_weight, self.real_bias) + F.linear(input.imag, self.real_weight)
        return torch.complex(real, imag)


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.real_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.imag_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.real_bias = nn.Parameter(torch.zeros(out_channels))
        self.imag_bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        if not input.is_complex():
            input = torch.complex(input, torch.zeros_like(input))

        real = F.conv2d(input.real, self.real_weight, self.real_bias, self.stride, self.padding) - \
            F.conv2d(input.imag, self.imag_weight, None, self.stride, self.padding)
        imag = F.conv2d(input.real, self.imag_weight, self.real_bias, self.stride, self.padding) + \
            F.conv2d(input.imag, self.real_weight, None, self.stride, self.padding)
        return torch.complex(real, imag)


class FourierLayer(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.real_weights = nn.Parameter(torch.randn(height, width))
        self.imag_weights = nn.Parameter(torch.randn(height, width))

    def forward(self, x):
        x_fft = torch.fft.fft2(x)
        real = x_fft.real * self.real_weights - x_fft.imag * self.imag_weights
        imag = x_fft.real * self.imag_weights + x_fft.imag * self.real_weights
        x_weighted = torch.complex(real, imag)
        return torch.fft.ifft2(x_weighted)


class HybridLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.conv = ComplexConv2d(input_channels, output_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.fft = FourierLayer(1000, 1000)

    def forward(self, x):
        x_conv = self.conv(x)
        x_fft = self.fft(x)
        x_combined = torch.cat((x_conv, x_fft), dim=1)
        return x_combined


class ComplexReLU(nn.Module):
    def forward(self, input):
        real = F.leaky_relu(input.real)
        imag = F.leaky_relu(input.imag)
        return torch.complex(real, imag)


class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, input):
        real = F.max_pool2d(input.real, self.kernel_size, self.stride, self.padding)
        imag = F.max_pool2d(input.imag, self.kernel_size, self.stride, self.padding)
        return torch.complex(real, imag)


class ComplexUpsample(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear'):
        super(ComplexUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        if not input.is_complex():
            input = torch.complex(input, torch.zeros_like(input))
        real_upsampled = F.interpolate(input.real, scale_factor=self.scale_factor, mode=self.mode)
        imag_upsampled = F.interpolate(input.imag, scale_factor=self.scale_factor, mode=self.mode)
        return torch.complex(real_upsampled, imag_upsampled)


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        if self.training:
            real_mask = (torch.rand(input.real.size(), device=input.real.device) > self.p).float()
            imag_mask = (torch.rand(input.imag.size(), device=input.imag.device) > self.p).float()
            if self.inplace:
                input.real.mul_(real_mask)
                input.imag.mul_(imag_mask)
                return input
            else:
                return torch.complex(input.real * real_mask, input.imag * imag_mask)
        return input


class ComplexToChannels(nn.Module):
    def __init__(self):
        super(ComplexToChannels, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Splits a complex tensor into its real and imaginary parts, and stacks them as separate channels
        return torch.stack((x.real, x.imag), dim=1)  # Shape: [batch, 2, height, width]


class ChannelsToComplex(nn.Module):
    def __init__(self):
        super(ChannelsToComplex, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes input x has shape [batch, 2, height, width] where x[:, 0] is real and x[:, 1] is imaginary
        return torch.complex(x[:, 0], x[:, 1])  # Recombines two channels into a complex tensor


class squeeze0(nn.Module):
    def __init__(self):
        super(squeeze0, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0)


class unsqueeze0(nn.Module):
    def __init__(self):
        super(unsqueeze0, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0)


class HybridAutoencoder(nn.Module):
    def __init__(self, BottleneckChannels: int):
        super(HybridAutoencoder, self).__init__()
        self.BC: int = BottleneckChannels
        # Encoder
        self.encoder = nn.Sequential(
            squeeze0(),
            ComplexToChannels(),
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(16, self.BC, kernel_size=3, stride=2, padding=1), nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.BC, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1, output_padding=0), nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1, output_padding=0), nn.ReLU(True),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            ChannelsToComplex(),
            unsqueeze0()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_encoded = self.encoder(x)
        x_reconstructed = self.decoder(x_encoded)

        return x_reconstructed


class HybridAutoencoder_deep(nn.Module):
    def __init__(self, BottleneckChannels: int):
        super(HybridAutoencoder, self).__init__()
        self.bottleneckChannels: int = BottleneckChannels
        # Encoder
        self.encoder = nn.Sequential(
            squeeze0(),
            ComplexToChannels(),
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.Conv2d(16, self.bottleneckChannels, kernel_size=3, stride=2, padding=1), nn.ReLU(True),
            # nn.Conv2d(32, 16, kernel_size=7, stride=5, padding=1),  nn.ReLU(True),
            # nn.Conv2d(16, 16, kernel_size=6, stride=5, padding=1),  nn.ReLU(True),
            # nn.Conv2d(16, 32, kernel_size=6, stride=5, padding=1),  nn.ReLU(True),
            # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  nn.ReLU(True),
            # nn.Flatten()
        )
        # Decoder
        self.decoder = nn.Sequential(
            # nn.Unflatten(1,(64, 1, 1)),
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   nn.ReLU(True),
            # nn.ConvTranspose2d(32, 16, kernel_size=6, stride=5, padding=1, output_padding=1),   nn.ReLU(True),
            # nn.ConvTranspose2d(16, 16, kernel_size=6, stride=5, padding=1, output_padding=1),   nn.ReLU(True),
            # nn.ConvTranspose2d(16, 32, kernel_size=6, stride=5, padding=1, output_padding=1),   nn.ReLU(True),
            nn.ConvTranspose2d(self.bottleneckChannels, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1, output_padding=0), nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1, output_padding=0), nn.ReLU(True),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            ChannelsToComplex(),
            unsqueeze0()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert complex input to two-channel real tensor
        x_encoded = self.encoder(x)
        x_reconstructed = self.decoder(x_encoded)

        return x_reconstructed


class HybridAutoencoder_STAR(nn.Module):
    def __init__(self):
        super(HybridAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            squeeze0(),
            ComplexToChannels(),
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # input is 1000x1000x2, output is 500x500x16
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # output is 250x250x32
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),  # output is 125x125x2
            nn.ReLU(True),
            nn.Flatten()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 125, 125)),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # output is 250x250x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # output is 500x500x16
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # output is 1000x1000x2
            nn.Tanh(),  # Using Tanh to keep the output in the range [-1,1] which is typical for normalized image data
            ChannelsToComplex(),
            unsqueeze0()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert complex input to two-channel real tensor
        x_encoded = self.encoder(x)
        x_reconstructed = self.decoder(x_encoded)

        return x_reconstructed


class HybridAutoencoder_6(nn.Module):
    def __init__(self):
        super(HybridAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # [1x1000x1000]
            squeeze0(),
            # [1000x1000]
            ComplexToChannels(),
            # [2x1000x1000]
            nn.Conv2d(2, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [10x500x500]
            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [20x250x250]
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [40x125x125]
            # nn.Conv2d(40, 80, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            # [80x63x63]
            # nn.Conv2d(80, 160, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            # [160x32x32]
            # nn.Conv2d(160, 320, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(True),
            # [320x16x16]
            # nn.Conv2d(320, 640, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            # [640x8x8]
            # nn.Conv2d(640, 1280, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            # [1280x4x4]
            # nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            # [1280x2x2]
            # nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            # [1280x1x1]
        )

        self.decoder = nn.Sequential(
            # [1280x1x1]
            # nn.ConvTranspose2d(1280, 1280, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(True),
            # [1280x2x2]
            # nn.ConvTranspose2d(1280, 1280, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(True),
            # [1280x4x4]
            # nn.ConvTranspose2d(1280, 640, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(True),
            # [640x8x8]
            # nn.ConvTranspose2d(640, 320, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(True),
            # [320x16x16]
            # nn.ConvTranspose2d(320, 160, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(True),
            # [160x32x32]
            # nn.ConvTranspose2d(160, 80, kernel_size=3, stride=2, padding=1, output_padding=0),
            # nn.ReLU(True),
            # [80x63x63] - Note, output shape is approximate due to stride and kernel size choices
            # nn.ConvTranspose2d(80, 40, kernel_size=3, stride=2, padding=1, output_padding=0),
            # nn.ReLU(True),
            # [40x125x125]
            nn.ConvTranspose2d(40, 20, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # [20x250x250]
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # [10x500x500]
            nn.ConvTranspose2d(10, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # [2x1000x1000]
            ChannelsToComplex(),
            # [1000x1000]
            unsqueeze0(),
            # [1x1000x1000]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert complex input to two-channel real tensor
        x_encoded = self.encoder(x)
        x_reconstructed = self.decoder(x_encoded)

        return x_reconstructed


class HybridAutoencoder_5(nn.Module):
    def __init__(self):
        super(HybridAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # [1x1000x1000]
            squeeze0(),
            # [2x1000x1000]
            ComplexToChannels(),
            # [2x1000x1000]
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [16x500x500]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [32x250x250]
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [64x125x125]
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [32x63x63]
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [16x32x32]
            nn.Conv2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # [8x16x16]
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [4x8x8]
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # [2x8x8]
            nn.Flatten(),
            # [128]
            nn.Linear(128, 100),
            nn.ReLU(True)
            # [100]
        )
        # Decoder
        self.decoder = nn.Sequential(
            # [100]
            nn.Linear(100, 64 * 7 * 7),
            nn.ReLU(True),
            # [3236]
            nn.Unflatten(1, (64, 7, 7)),
            # [64x7x7]
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [64x13x13]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # [32x25x25]
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # [16x50x50}
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=5, padding=1),
            nn.ReLU(True),
            # [16x250x250]
            nn.ConvTranspose2d(16, 8, kernel_size=6, stride=4, padding=1),
            nn.ReLU(True),
            # [8x1000x1000]
            nn.ConvTranspose2d(8, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            # [2x1000x1000]
            ChannelsToComplex(),
            # [1000x1000]
            unsqueeze0()
            # [1x1000x1000]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert complex input to two-channel real tensor
        x_encoded = self.encoder(x)
        x_reconstructed = self.decoder(x_encoded)

        return x_reconstructed


class HybridAutoencoder_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.latentDimensions: int = 10
        self.encoder = nn.Sequential(
            nn.Flatten(),
            ComplexLinear(1000000, 200),
            ComplexReLU(),
            ComplexLinear(200, self.latentDimensions)
        )
        self.decoder = nn.Sequential(
            ComplexLinear(self.latentDimensions, 200),
            ComplexReLU(),
            ComplexLinear(200, 1000000),
            nn.Unflatten(1, (1, 1000, 1000))
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded


class HybridAutoencoder_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.latentDimensions: int = 10
        self.encoder = nn.Sequential(
            nn.Flatten(),
            ComplexReLU(),
            ComplexLinear(1000000, self.latentDimensions)
        )
        self.decoder = nn.Sequential(
            ComplexLinear(self.latentDimensions, 50),
            ComplexReLU(),
            ComplexLinear(50, 200),
            ComplexReLU(),
            ComplexLinear(200, 1000000),
            nn.Unflatten(1, (1, 1000, 1000))
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded


class HybridAutoencoder_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.latentDimensions: int = 10
        self.encoder = nn.Sequential(
            HybridLayer(1, 32, 3),  # Reduced number of filters and kernel size
            ComplexReLU(),
            ComplexMaxPool2d(4, 4),  # Increased pooling size
            nn.Flatten(),
            ComplexLinear(2062500, self.latentDimensions),  # Adjusted to match new flattened size
        )
        self.decoder = nn.Sequential(
            ComplexLinear(self.latentDimensions, 1000000),
            ComplexReLU(),
            nn.Unflatten(1, (16, 250, 250)),  # Adjusted dimensions
            ComplexUpsample(scale_factor=4,
                            mode='nearest'),
            ComplexConv2d(16, 1, 5, padding=2),  # Single convolution layer
            nn.Sigmoid()
        )

    def forward(self, x):
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        x = self.encoder(x)
        x = self.decoder(x)
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))  # Ensure output is complex
        return x


class HybridAutoencoder_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.latentDimensions: int = 10
        self.encoder = nn.Sequential(
            nn.Flatten(),
            ComplexReLU(),
            ComplexLinear(1000000, 250),  # Adjusted to match new flattened size
            ComplexReLU(),
            ComplexLinear(250, self.latentDimensions),  # Adjusted to match new flattened size
        )
        self.decoder = nn.Sequential(
            ComplexLinear(self.latentDimensions, 250),
            ComplexReLU(),
            ComplexLinear(250, 1000000),
            ComplexReLU(),
            nn.Unflatten(1, (1, 1000, 1000)),  # Adjusted dimensions
            ComplexReLU(),
            ComplexConv2d(1, 1, 5, padding=2),  # Single convolution layer
            nn.Sigmoid()
        )

    def forward(self, x):
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        x = self.encoder(x)
        x = self.decoder(x)
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))  # Ensure output is complex
        return x
