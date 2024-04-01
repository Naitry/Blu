# for forward type annotation (recursive data structure with annotation)
from __future__ import annotations

# typing
from typing import Optional

# compute
import torch
import numpy as np

# Blu components
from Blu.Math.DifferentialGeometry import laplacianLegacy as Laplacian
from Blu.Psithon.GaussianWavePacket import GaussianWavePacket
from Blu.Utils.Terminal import clearTerminal, getTerminalSize, arrayToText, arrayToTextColored

# Data output
from PIL import Image
from matplotlib import cm
import h5py

# Suppress Warning
import warnings

# Function to suppress specific UserWarnings
warnings.filterwarnings("ignore", message="ComplexHalf support is experimental and many operators don't support it yet.*")

# Default simulation size
BLU_PSITHON_defaultDimensions: int = 2
BLU_PSITHON_defaultResolution: int = 1000

# Primary data type for the field
BLU_PSITHON_defaultDataType: torch.dtype = torch.cfloat

# Floating point data type which will be able to represent one of the complex components
BLU_PSITHON_defaultDataTypeComponent: torch.dtype = torch.float16


class Field:

    def __init__(self,
                 name: Optional[str] = None,
                 field: Optional[torch.Tensor] = None,
                 spatialDimensions: int = BLU_PSITHON_defaultDimensions,
                 resolution: int = BLU_PSITHON_defaultResolution,
                 dtype: torch.dtype = BLU_PSITHON_defaultDataType,
                 device: torch.device = torch.device('cpu')):
        self.name: Optional[str] = name
        self.tensor: torch.Tensor
        self.spatialDimensions: int = spatialDimensions
        self.resolution: int = resolution
        if field is None:
            self.tensor = torch.zeros(size=[resolution] * spatialDimensions,
                                      dtype=dtype,
                                      device=device,
                                      requires_grad=False)
        else:
            self.tensor = field

    def addWavePacket(self,
                      packetSize: int,
                      sigma: float = 20.0,
                      k: list[float] = None,
                      position: Optional[list[float]] = None,
                      dtype: torch.dtype = torch.float32,
                      device: torch.device = torch.device('cpu')) -> None:
        """
        Place a Gaussian wave packet into the field at a specified position.

        :param packetSize: Size of the wave packet.
        :param sigma: Controls the size of the wave packet.
        :param k: Wave vector, defining the direction and speed of the packet.
        :param position: The position at which to place the center of the wave packet in the field.
        :param dtype: Data type for the wave packet tensor.
        :param device: Device on which the wave packet will be generated.
        :return: None. The function modifies the field in place.
        """
        if position is None:
            position = [self.resolution // 2] * self.spatialDimensions

        # Ensure that the position and wave vector lists have the same number of dimensions as the field
        if len(position) != self.tensor.dim():
            raise ValueError("Position must have the same number of dimensions as the field")
        if len(k) != self.tensor.dim():
            raise ValueError("Wave vector (k) must have the same number of dimensions as the field")

        # Generate the Gaussian wave packet
        wavePacket = GaussianWavePacket(packetSize=packetSize,
                                        dimensions=self.tensor.dim(),
                                        sigma=sigma,
                                        k=torch.tensor(data=k,
                                                       dtype=torch.float32,
                                                       device=device),
                                        dtype=dtype,
                                        device=device)

        # Initialize slices for the field and the wave packet
        fieldSlices = []
        wavePacketSlices = []

        # Construct slices based on the specified position and the packet size
        for dim, pos in enumerate(position):
            startPos = max(0,
                           pos - packetSize // 2)
            endPos = startPos + packetSize

            # Adjust start and end positions if they are out of the field's boundaries
            startPos = max(min(startPos,
                               self.tensor.size(dim) - 1),
                           0)
            endPos = max(min(endPos,
                             self.tensor.size(dim)),
                         0)

            # Calculate the slice for the wave packet
            wavePacketStart = max(0,
                                  packetSize // 2 - pos)
            wavePacketEnd = wavePacketStart + (endPos - startPos)

            # Append the slices to the listsL
            fieldSlices.append(slice(startPos,
                                     endPos))
            wavePacketSlices.append(slice(wavePacketStart,
                                          wavePacketEnd))

        # Place the wave packet into the field at the specified position
        # The ellipsis (...) allows for slicing in N dimensions
        self.tensor[tuple(fieldSlices)] = wavePacket[tuple(wavePacketSlices)]

    def calculateEntropy(self) -> float:
        # Calculate the probability distribution from the wave function
        probabilityDistribution = (torch.abs(self.tensor) ** 2).type(torch.float64)
        # Ensure normalization
        probabilityDistribution.divide_(torch.sum(probabilityDistribution))

        # Calculate the Shannon entropy
        # Add a small number to avoid log(0)
        entropy = -torch.sum(probabilityDistribution * torch.log(probabilityDistribution + 1e-12))
        print(entropy)

        return entropy.item()

    def update(self,
               dt: float,
               delta: float,
               device: torch.device) -> Field:
        """
        Update the field for an n-dimensional space.

        :param: self: The input field as an n-dimensional torch tensor.
        :param: device: The torch device on which to perform the calculations.
        :param: delta: The spacing between points in the field.
        :param: dt: Time step for the update.
        :return: The updated field as an n-dimensional torch tensor.
        """

        # Initialize the potential for n-dimensions (assuming initPotential is updated to handle n-dim inputs)
        V = torch.zeros_like(self.tensor)

        # Calculate the laplacian for n-dimensions (assuming Laplacian is updated for n-dim inputs)
        laplaceField: torch.Tensor = Laplacian(field=self.tensor,
                                               delta=delta)

        # Iterate based on the time dependent Schrödinger equation
        self.tensor.add_(-1j * dt * (-0.5 * laplaceField))

        # Apply boundary conditions for n-dimensions
        for dim in range(self.tensor.dim()):
            # Set the first and last index along each dimension to 0
            self.tensor.index_fill_(dim,
                                    torch.tensor([0, self.tensor.size(dim) - 1],
                                                 device=device),
                                    0)

        return self

    def saveHDF5(self,
                 timestep: int,
                 entropy: Optional[float],
                 filepath: str) -> None:
        # Convert tensor to ComplexFloat if it's not already
        if self.tensor.dtype == torch.complex32:  # ComplexHalf in PyTorch is torch.complex32
            converted_tensor = self.tensor.to(torch.complex64)  # Convert to ComplexFloat
        else:
            converted_tensor = self.tensor

        entropy = entropy or self.calculateEntropy()
        with h5py.File(filepath,
                       'a') as f:
            # Convert and save real and imaginary components as float32 (numpy's default)
            f.create_dataset(f'real_{timestep}',
                             data=converted_tensor.real.numpy())
            f.create_dataset(f'imaginary_{timestep}',
                             data=converted_tensor.imag.numpy())
            # Assuming 'potential' was meant to be a separate, real-valued dataset;
            # adjust accordingly if it's meant to be part of the complex tensor
            # For demonstration, saving it as is, but ensure it's properly handled according to your needs
            f.create_dataset(f'potential_{timestep}',
                             data=np.full_like(converted_tensor.real.numpy(),
                                               fill_value=0.0))
            f.create_dataset(f'name_{timestep}',
                             data=np.array(self.name).astype('S'))  # Assuming self.name is a string
            f.create_dataset(f'entropy_{timestep}',
                             data=np.array(entropy).astype(np.float32))

    def saveImage(self,
                  filepath: str) -> None:
        absField: np.ndarray = torch.abs(self.tensor).cpu().numpy()
        realField: np.ndarray = torch.real(self.tensor).cpu().numpy()
        imagField: np.ndarray = torch.imag(self.tensor).cpu().numpy()

        def arrayToImage(arr: np.ndarray,
                         cmap):
            if arr.max() > 0:  # Avoid division by zero
                normalized_arr = arr / arr.max()
            else:  # Handle the case where the array max is 0
                normalized_arr = arr
            return Image.fromarray(np.uint8(cmap(normalized_arr) * 255))

        # Convert fields to images using Pillow
        absImg: Image = arrayToImage(absField,
                                     cm.twilight_shifted)
        realImg: Image = arrayToImage(realField,
                                      cm.cool)
        imagImg: Image = arrayToImage(imagField,
                                      cm.spring)

        combinedImg: Image = Image.new(mode='RGB',
                                       size=(2 * self.tensor.size(0), self.tensor.size(1)))
        combinedImg.paste(im=absImg,
                          box=(0, 0))
        combinedImg.paste(im=realImg,
                          box=(self.tensor.size(0), 0))

        combinedImg.save(filepath)

    def printField(self,
                   clear: bool = True) -> None:
        if clear:
            clearTerminal()
            # Get terminal size and adjust for aspect ratio of the tensor
        columns, lines = getTerminalSize()
        aspect_ratio = self.tensor.size(1) / self.tensor.size(0)
        text_width = columns
        text_height = int(text_width * aspect_ratio)
        absField: np.ndarray = torch.abs(self.tensor).cpu().numpy()
        print(arrayToTextColored(arr=absField,
                          width=text_width,
                          height=text_height),
              end="")

    def loadFromHDF5(self,
                     filePath: str) -> None:
        """
        Loads a tensor from an HDF5 file that contains real, imaginary, and potential components.

        filePath: Path to the HDF5 file to load the data from.

        Returns:
        A complex tensor reconstructed from the loaded data.
        """
        with h5py.File(filePath,
                       'r') as f:
            realPart = torch.Tensor(f['real'][:])
            imagPart = torch.Tensor(f['imaginary'][:])
            self.tensor = torch.complex(real=realPart,
                                        imag=imagPart)
            self.name = f['name'][:]


def loadFieldFromHDF5(filePath: str,
                      spatialDimensions: int = BLU_PSITHON_defaultDimensions,
                      resolution: int = BLU_PSITHON_defaultResolution,
                      dtype: torch.dtype = BLU_PSITHON_defaultDataType,
                      device: torch.device = torch.device('cpu')) -> Field:
    """
    Loads a complex field from an HDF5 file that contains real, imaginary, and potential components.

    Args:
    filePath: Path to the HDF5 file to load the data from.

    Returns:
    A complex tensor reconstructed from the loaded data.
    """
    field = Field(spatialDimensions=spatialDimensions,
                  resolution=resolution,
                  dtype=dtype,
                  device=device)
    field.loadFromHDF5(filePath=filePath)
    return field
