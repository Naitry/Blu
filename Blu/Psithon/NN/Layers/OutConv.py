from torch import nn


class OutConv(nn.Module):
    """Convolution to produce the final output"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
