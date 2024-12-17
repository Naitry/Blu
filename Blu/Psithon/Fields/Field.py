# forward type annotation
from __future__ import annotations
import warnings
import h5py
from matplotlib import cm, colors
from PIL import Image
from Blu.Psithon.DefaultDefinitions import (BLU_PSITHON_defaultRank,
                                            BLU_PSITHON_defaultDataType,
                                            BLU_PSITHON_defaultDimensions,
                                            BLU_PSITHON_defaultResolution)

# 3Typing
from typing import Optional

# Compute
import torch
import numpy as np

# Blu
from Blu.Math.DifferentialGeometry import laplacianLegacy as Laplacian
from Blu.Psithon.Fields.GaussianWavePacket import GaussianWavePacket
from Blu.Utils.Terminal import (clearTerminal,
                                getTerminalSize,
                                tensorToTextColored)
from Blu.Utils.Functions import getCurrentFunctionName

import random

# Suppress specific UserWarnings
warnings.filterwarnings("ignore",
                        message="ComplexHalf support is experimental and many operators don't support it yet.*")


class Field:
    def __init__(self,
                 name: str,
                 device: torch.device,
                 dtype: torch.dtype = BLU_PSITHON_defaultDataType,
                 spatialDimensions: int = BLU_PSITHON_defaultDimensions,
                 resolution: int = BLU_PSITHON_defaultResolution,
                 fieldRank: int = BLU_PSITHON_defaultRank,
                 field: Optional[torch.Tensor] = None):
        # set field name
        self.name: str = name

        # set torch variables
        self.field: torch.Tensor
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        # set field shape and size
        self.spatialDimensions: int = spatialDimensions
        self.dimensions: int = spatialDimensions + 1
        self.resolution: int = resolution

        # CASE: field argument is none
        if field is None:
            # generate field
            rankPortion: list = [self.dimensions**(fieldRank - 1)]
            spatialPortion: list = [resolution] * self.spatialDimensions
            fieldShape: list = rankPortion + spatialPortion

            # new empty field
            self.field = torch.zeros(size=fieldShape,
                                     dtype=dtype,
                                     device=device,
                                     requires_grad=False)
        else:
            # assign field to the input
            self.field = field

    def addWavePacket(self,
                      packetSize: int,
                      k: list[float] = None,
                      position: Optional[list[float]] = None) -> None:
        """
        Place a Gaussian wave packet into the field at a specified position

        :param packetSize: Size of the wave packet.
        :param sigma: Controls the size of the wave packet.
        :param k: Wave vector, defining the direction and speed of the packet.
        :param position: The position at which to place the center of the wave packet in the field.
        :param dtype: Data type for the wave packet tensor.
        :return: None. The function modifies the field in place.
        """
        # CASE: position argument is none
        if position is None:
            # set position to the middle of the field
            position = [self.resolution // 2] * self.dimensions

        positionLength: int = len(position)
        waveVectorLength: int = len(k)
        fieldDims: int = self.field.dim() - 1

        # CASE: length of position != # of field dimensions
        if positionLength != fieldDims:
            line1: str = "Position must have the same number of dimensions as the field\n"
            line2: str = "positionLength: %d != fieldDims: %d" % (positionLength, fieldDims)
            raise ValueError(line1 + line2)

        # CASE: length of wave vector != # of field dimensions
        elif waveVectorLength != fieldDims:
            line1: str = "Wave vector (k) must have the same number of dimensions as the field\n"
            line2: str = "wavevectorlength: %d != fieldDims: %d" % (positionLength, fieldDims)
            raise ValueError(line1 + line2)

        # Generate the Gaussian wave packet
        wavePacket: torch.Tensor = GaussianWavePacket(packetSize=packetSize,
                                                      dimensions=fieldDims,
                                                      sigma=40.0,
                                                      k=torch.tensor(data=k,
                                                                     dtype=self.dtype,
                                                                     device=self.device),
                                                      device=self.device)

        # Initialize slices for the field and the wave packet
        fieldSlices: list = []
        wavePacketSlices: list = []

        # Construct slices based on the specified position and the packet size
        # iterate through each position
        for dim, pos in enumerate(position):
            startPos: int = max(0,
                                int(pos - float(packetSize) / 2.0))
            endPos: int = startPos + packetSize

            # Calculate the slice for the wave packet
            wavePacketStart: int = max(0,
                                       int(float(packetSize) / 2.0 - pos))
            wavePacketEnd: int = wavePacketStart + (endPos - startPos)

            # Append the slices to the listsL
            fieldSlices.append(slice(int(startPos),
                                     int(endPos)))
            wavePacketSlices.append(slice(int(wavePacketStart),
                                          int(wavePacketEnd)))

        # Place the wave packet into the field at the specified position
        self.field[0][tuple(fieldSlices)] += wavePacket[tuple(wavePacketSlices)]

    def addRandomWavePacket(self,
                            kMagnitude: float = 0.3) -> None:
        """
        Place a Gaussian wave packet into the field at a random position
        """
        dims: int = len(self.field.shape) - 1
        size: int = random.randrange(10, int(self.resolution))
        k: list[int] = [random.uniform(-kMagnitude, kMagnitude) for _ in range(dims)]
        position: list[float] = [random.uniform(0 + size // 2,
                                                int(self.resolution) - size // 2) for _ in range(dims)]
        self.addWavePacket(packetSize=size,
                           k=k,
                           position=position)

    def calculateEntropy(self) -> float:
        # Calculate the probability distribution from the wave function
        probabilityDistribution: torch.Tensor = (torch.abs(self.field) ** 2).type(torch.float32)
        # Normalize
        probabilityDistribution.divide_(torch.sum(probabilityDistribution))

        # Calculate the Shannon entropy
        # Add a small number to avoid log(0)
        entropy: torch.Tensor = -torch.sum(probabilityDistribution * torch.log(probabilityDistribution + 1e-12))

        return entropy.item()

    def update(self,
               dt: float,
               delta: float) -> Field:
        """
        Update the field for an n-dimensional space.

        :param: self: The input field as an n-dimensional torch tensor.
        :param: delta: The spacing between points in the field.
        :param: dt: Time step for the update.
        :return: The updated field which contains San n-dimensional torch tensor.
        """
        if self.field.shape[0] == 1:
            subField: torch.Tensor = self.field[0]
            # Initialize potential
            v: torch.Tensor = torch.zeros_like(subField)

            # Calculate laplacian
            laplaceField: torch.Tensor = Laplacian(field=subField,
                                                   delta=delta)

            # update the field according to the time dependent Schrödinger equation
            subField.add_(-1j * dt * (-0.5 * laplaceField + v))

            # iterate over each dimenshion and apply boundary conditions
            for dim in range(subField.dim()):
                # Set the first and last index along each dimension to 0
                subField.index_fill_(dim,
                                     torch.tensor([0, self.field.size(dim) - 1],
                                                  device=self.device),
                                     0)
        else:
            print(f"{getCurrentFunctionName()} not yet supported of this field rank")
        return self

    def saveHDF5(self,
                 timestep: int,
                 entropy: Optional[float],
                 filepath: str) -> None:
        if self.field.shape[0] == 1:
            subField: torch.Tensor = self.field[0]

            # Convert tensor to ComplexFloat if it's not already
            if self.field.dtype == torch.complex32:  # ComplexHalf in PyTorch is torch.complex32
                subField = subField.to(torch.complex64)  # Convert to ComplexFloat

            entropy = entropy or self.calculateEntropy()
            with h5py.File(filepath,
                           'a') as f:
                # Convert and save real and imaginary components as float32 (numpy's default)
                f.create_dataset(f'real_{timestep}',
                                 data=subField.real.numpy())
                f.create_dataset(f'imaginary_{timestep}',
                                 data=subField.imag.numpy())
                f.create_dataset(f'potential_{timestep}',
                                 data=np.full_like(subField.real.numpy(),
                                                   fill_value=0.0))
                f.create_dataset(f'name_{timestep}',
                                 data=np.array(self.name).astype('S'))
                f.create_dataset(f'entropy_{timestep}',
                                 data=np.array(entropy).astype(np.float32))
        else:
            print(f"{getCurrentFunctionName()} not yet supported of this field rank")

    def saveImage(self, filepath: str) -> None:
        if self.field.shape[0] == 1:
            subField: torch.Tensor = self.field[0]

            # split the representations of the field into 2 np arrays all on cpu
            absField: np.ndarray = torch.abs(subField).cpu().detach().numpy()
            imagField: np.ndarray = torch.imag(subField).cpu().detach().numpy()

            def create_blue_transparent_colormap():
                # Create a colormap that goes from transparent to blue
                blues = [(1, 0.3, 0, 0), (1, 0.3, 0, 1)]  # (R, G, B, A)
                n_bins = 100  # Number of divisions in the colormap
                cmap = colors.LinearSegmentedColormap.from_list("blue_transparent", blues, N=n_bins)
                return cmap

            def arrayToImage(arr: np.ndarray, cmap):
                if arr.max() > 0:  # Avoid division by zero
                    normalized_arr = arr / arr.max()
                else:  # Handle the case where the array max is 0
                    normalized_arr = arr
                return Image.fromarray(np.uint8(cmap(normalized_arr) * 255))

            # Convert fields to images using Pillow
            absImg: Image = arrayToImage(absField, cm.Blues)

            # Create and use the blue-to-transparent colormap for imaginary field
            blue_transparent_cmap = create_blue_transparent_colormap()
            imagImg: Image = arrayToImage(imagField, blue_transparent_cmap)

            # Convert absolute image to RGBA
            absImg = absImg.convert("RGBA")

            # Ensure both images have the same size
            absImg = absImg.resize(imagImg.size)

            # Combine the images
            combinedImg = Image.alpha_composite(absImg, imagImg)

            combinedImg.save(filepath)
        else:
            print(f"{getCurrentFunctionName()} not yet supported of this field rank")

    def printField(self,
                   clear: bool = True) -> None:
        if self.field.shape[0] == 1:
            subField: torch.Tensor = self.field[0]

            # CASE: terminal should be cleared
            if clear:
                clearTerminal()

            # get terminal size
            columns: int
            lines: int
            columns, lines = getTerminalSize()

            # adjust for aspect ratio of the tensor
            aspectRatio: float = subField.size(1) / subField.size(0)
            textWidth: int = columns
            textHeight: int = int(textWidth * aspectRatio)

            # take the absolute value of the field
            absField: torch.Tensor = torch.abs(subField)

            # convert to colored text and print the field
            print(tensorToTextColored(tensor=absField,
                                      width=textWidth,
                                      height=textHeight),
                  end="")
        else:
            print(f"{getCurrentFunctionName()} not yet supported of this field rank")

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
