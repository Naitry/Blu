import torch
from typing import Optional


def GaussianWavePacket(packetSize: int,
                       dimensions: int = 2,
                       sigma: float = 20.0,
                       k: Optional[torch.Tensor] = None,
                       dtype: torch.dtype = torch.float32,
                       device: torch.device = torch.device('mps')) -> torch.Tensor:
    """
    Generate an N-dimensional Gaussian wave packet.

    :param:	packetSize (int): Size of the wave packet.
    :param: dimensions (int): Number of dimensions.
    :param: sigma (float): Value between 0 and 100 determining the size of the wave packet
    :param: k (torch.Tensor): Wave vector, should be of the same dimension as 'dimensions'.
    :param: dtype (torch.dtype): Data type for the tensor.
    :param: device (torch.device): Device on which the tensor will be allocated.
    :return: torch.Tensor: The N-dimensional Gaussian wave packet.
    """
    if k is None:
        k: torch.Tensor = torch.zeros(size=[dimensions, ],
                                      dtype=torch.cfloat,
                                      device=device)
    else:
        k = k.to(device=device)

    # Validate the dimension of k
    if len(k) != dimensions:
        raise ValueError("Length of k must match the number of dimensions")

    # Create a range tensor for one dimension
    rangeTensor: torch.Tensor = torch.arange(start=-packetSize // 2,
                                             end=packetSize // 2,
                                             step=1,
                                             dtype=dtype,
                                             device=device)

    # Create an N-dimensional grid
    packetGrid: torch.Tensor = torch.stack(torch.meshgrid([rangeTensor] * dimensions,
                                                          indexing='ij'),
                                           dim=-1)

    # Convert grid to complex numbers
    complexGrid: torch.Tensor = torch.complex(real=packetGrid,
                                              imag=torch.zeros_like(packetGrid))

    # Calculate the squared sum
    squaredSum: torch.Tensor = torch.sum(input=complexGrid ** 2,
                                         dim=-1)

    size: float = packetSize * (sigma / 300.0)

    # Calculate the Gaussian wave packet
    wavePacket: torch.Tensor = torch.exp(input=-squaredSum / (2 * size ** 2))

    # Calculate the complex exponential term using broadcasting
    k = k.reshape((1,) * dimensions + (-1,))
    exponentialComponent: torch.Tensor = torch.exp(-1j * torch.sum(k * complexGrid,
                                                                   dim=-1))

    # Combine both parts
    wavePacket *= exponentialComponent

    return wavePacket
