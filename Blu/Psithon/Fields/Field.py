# forward type annotation
from __future__ import annotations

# Typing
from typing import Optional

# Compute
import torch
import numpy as np

# Blu
from Blu.Math.DifferentialGeometry import laplacianLegacy as Laplacian
from Blu.Psithon.Fields.GaussianWavePacket import GaussianWavePacket
from Blu.Utils.Terminal import (clearTerminal,
                                getTerminalSize,
                                arrayToTextColored)
from Blu.Psithon.DefaultDefinitions import (BLU_PSITHON_defaultRank,
                                            BLU_PSITHON_defaultDataType,
                                            BLU_PSITHON_defaultDimensions,
                                            BLU_PSITHON_defaultResolution)

# Rendering and Output
from PIL import Image
from matplotlib import cm
import h5py

# Warnings
import warnings

# Function to suppress specific UserWarnings
warnings.filterwarnings("ignore",
                        message="ComplexHalf support is experimental and many operators don't support it yet.*")


class Field:
    def __init__(self,
                 device: torch.device,
                 name: str,
                 field: Optional[torch.Tensor] = None,
                 spatialDimensions: int = BLU_PSITHON_defaultDimensions,
                 fieldRank: int = BLU_PSITHON_defaultRank,
                 resolution: int = BLU_PSITHON_defaultResolution,
                 dtype: torch.dtype = BLU_PSITHON_defaultDataType):
        self.name = name
        # main object: a tensor
        self.field: torch.Tensor
        self.spatialDimensions: int = spatialDimensions
        self.dimensions: int = spatialDimensions + 1
        self.resolution: int = resolution
        self.dtype: torch.dtype = dtype
        if field is None:
            self.field = torch.zeros(size=[self.dimensions**(fieldRank - 1)] + [resolution] * self.spatialDimensions,
                                     dtype=dtype,
                                     device=device,
                                     requires_grad=False)
        else:
            self.field = field

    def addWavePacket(self,
                      packetSize: int,
                      sigma: float = 20.0,
                      k: list[float] = None,
                      position: Optional[list[float]] = None,
                      dtype: torch.dtype = torch.float32,
                      device: torch.device = torch.device('mps')) -> None:
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
        if len(position) != self.field.dim():
            raise ValueError("Position must have the same number of dimensions as the field")
        if len(k) != self.field.dim():
            raise ValueError("Wave vector (k) must have the same number of dimensions as the field")

        # Generate the Gaussian wave packet
        wavePacket = GaussianWavePacket(packetSize=packetSize,
                                        dimensions=self.field.dim(),
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
                               self.field.size(dim) - 1),
                           0)
            endPos = max(min(endPos,
                             self.field.size(dim)),
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
        self.field[tuple(fieldSlices)] += wavePacket[tuple(wavePacketSlices)]

    def calculateEntropy(self) -> float:
        # Calculate the probability distribution from the wave function
        probabilityDistribution = (torch.abs(self.field) ** 2).type(torch.float32)
        # Ensure normalization
        probabilityDistribution.divide_(torch.sum(probabilityDistribution))

        # Calculate the Shannon entropy
        # Add a small number to avoid log(0)
        entropy = -torch.sum(probabilityDistribution * torch.log(probabilityDistribution + 1e-12))
        print(entropy)

        return entropy.item()

    def update(self,
               device: torch.device,
               dt: float,
               delta: float) -> Field:
        """
        Update the field for an n-dimensional space.

        :param: self: The input field as an n-dimensional torch tensor.
        :param: device: The torch device on which to perform the calculations.
        :param: delta: The spacing between points in the field.
        :param: dt: Time step for the update.
        :return: The updated field which contains San n-dimensional torch tensor.
        """

        # Initialize potential
        V = torch.zeros_like(self.field)

        # Calculate laplacian
        laplaceField: torch.Tensor = Laplacian(field=self.field,
                                               delta=delta)

        # Iterate according to the time dependent Schrödinger equation
        self.field.add_(-1j * dt * (-0.5 * laplaceField + V))

        # Apply boundary conditions for n-dimensions
        for dim in range(self.field.dim()):
            # Set the first and last index along each dimension to 0
            self.field.index_fill_(dim,
                                   torch.tensor([0, self.field.size(dim) - 1],
                                                device=device),
                                   0)
        return self

    def saveHDF5(self,
                 timestep: int,
                 entropy: Optional[float],
                 filepath: str) -> None:
        # Convert tensor to ComplexFloat if it's not already
        if self.field.dtype == torch.complex32:  # ComplexHalf in PyTorch is torch.complex32
            converted_tensor = self.field.to(torch.complex64)  # Convert to ComplexFloat
        else:
            converted_tensor = self.field

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
        print(self.field.size())
        absField: np.ndarray = torch.abs(self.field).cpu().detach().numpy()
        realField: np.ndarray = torch.real(self.field).cpu().detach().numpy()
        imagField: np.ndarray = torch.imag(self.field).cpu().detach().numpy()

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
                                       size=(2 * self.field.size(0), self.field.size(1)))
        combinedImg.paste(im=absImg,
                          box=(0, 0))
        combinedImg.paste(im=realImg,
                          box=(self.field.size(0), 0))

        combinedImg.save(filepath)

    def printField(self,
                   clear: bool = True) -> None:
        if clear:
            clearTerminal()
            # Get terminal size and adjust for aspect ratio of the tensor
        columns, lines = getTerminalSize()
        aspect_ratio = self.field.size(1) / self.field.size(0)
        text_width = columns
        text_height = int(text_width * aspect_ratio)
        absField: np.ndarray = torch.abs(self.field).cpu().numpy()
        print(arrayToTextColored(arr=absField,
                                 width=text_width,
                                 height=text_height),
              end="")

    def loadFromHDF5(self,
                     filePath: str,
                     timestep: Optional[int] = None) -> None:
        """
        Loads a tensor from an HDF5 file. If timestep is provided, it attempts to load datasets
        for that specific timestep.

        Args:
        filePath: Path to the HDF5 file to load the data from.
        timestep: Specific timestep to load. If None, attempts to load the first timestep found.
        """
        with h5py.File(filePath, 'r') as f:
            # Attempt to dynamically determine the dataset names if timestep is provided or not
            real_dataset_name = f'real_{timestep}' if timestep is not None else 'real'
            imaginary_dataset_name = f'imaginary_{timestep}' if timestep is not None else 'imaginary'
            name_dataset_name = f'name_{timestep}' if timestep is not None else 'name'

            # Check if the expected datasets exist
            if real_dataset_name not in f or imaginary_dataset_name not in f:
                raise KeyError(
                    f"Required datasets '{real_dataset_name}' or '{imaginary_dataset_name}' not found in file.")

            realPart = torch.Tensor(f[real_dataset_name][:])
            imagPart = torch.Tensor(f[imaginary_dataset_name][:])
            self.field = torch.complex(real=realPart, imag=imagPart)

            if name_dataset_name in f:
                self.name = f[name_dataset_name][:].astype(str)


def loadFieldFromHDF5(filePath: str,
                      spatialDimensions: int = 2,
                      resolution: int = 1000,
                      dtype: torch.dtype = torch.cfloat,
                      device: torch.device = torch.device('cpu'),
                      timestep: Optional[int] = None) -> Field:
    """
    Function to load a field from an HDF5 file, possibly for a specific timestep.

    Args:
    filePath: The path to the HDF5 file.
    spatialDimensions, resolution, dtype, device: Parameters for the Field initialization.
    timestep: Specific timestep to load. If None, attempts to load the first timestep found.

    Returns:
    A Field object loaded with data from the HDF5 file.
    """
    field = Field(spatialDimensions=spatialDimensions, resolution=resolution, dtype=dtype, device=device)
    field.loadFromHDF5(filePath=filePath, timestep=timestep)
    return field


def loadFieldComponentDict(filePath: str,
                           prefix: str) -> dict[int, torch.Tensor]:
    """
    List all timesteps for datasets with a given prefix in an HDF5 file.

    Args:
        filePath: The path to the HDF5 file.
        prefix: The prefix to filter datasets by (e.g., 'real' or 'imaginary').

    Returns:
        A list of timesteps (as integers) in chronological order.
    """
    data: dict[int, torch.Tensor] = {}

    def filterDatasets(name: str,
                       obj: h5py.Dataset):
        if isinstance(obj, h5py.Dataset) and name.startswith(prefix):
            # Extract timestep from the dataset name
            _, timestep_str = name.split('_')
            try:
                timestep = int(timestep_str)
                data[timestep] = torch.tensor(np.array(obj))
            except ValueError:
                # Handle cases where the conversion fails
                print(f"Warning: Found dataset with non-integer timestep: {name}")

    with h5py.File(filePath, 'r') as file:
        file.visititems(filterDatasets)

    return sorted(data.items())


def loadSimulation(filepath: str):
    realData: dict[int, torch.Tensor] = loadFieldComponentDict(filepath=filepath,
                                                               prefix="real")
    imaginaryData: dict[int, torch.Tensor] = loadFieldComponentDict(filepath=filepath,
                                                                    prefix="imaginary")
    pass
