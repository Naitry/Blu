from __future__ import annotations
import torch
from Blu.Psithon.Field import Field, loadFieldComponentDict
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import cm
from typing import Optional


class Simulation():
    def __init__(self,
                 fields: dict[str, list[Field]] = None):
        self.fields = dict[str, list[Field]]
        if fields is None:
            self.fields = {}
        else:
            self.fields = fields

    def addFieldTimestep(self,
                         f: Field):
        if f.name not in self.fields:
            self.fields[f.name] = [f]
        else:
            self.fields[f.name].append(f)
        pass

    def numTimeSteps(self,
                     fieldName: str) -> int:
        return len(self.fields[fieldName])

    def loadSimField(self,
                     simDir: str,
                     fieldName: str) -> Simulation:

        filePath: str = simDir + fieldName + ".hdf5"
        print(filePath)
        realData: dict[int, torch.Tensor] = loadFieldComponentDict(filePath=filePath,
                                                                   prefix="real")
        imaginaryData: dict[int, torch.Tensor] = loadFieldComponentDict(filePath=filePath,
                                                                        prefix="imaginary")

        assert len(realData) == len(imaginaryData), "Real and Imaginary datasets do not have the same length"

        for i, data in enumerate(realData):
            timestep, realComponent = data
            imaginaryComponent = imaginaryData[i][1]
            fieldTensor: torch.Tensor = torch.complex(real=realComponent,
                                                      imag=imaginaryComponent)
            self.addFieldTimestep(f=Field(name=fieldName,
                                          field=fieldTensor,
                                          spatialDimensions=len(fieldTensor.shape),
                                          resolution=fieldTensor.shape[0],
                                          dtype=fieldTensor.dtype,
                                          device=fieldTensor.device))

    def tensorToImage(self, tensor: torch.Tensor, cmap) -> Image:
        """Converts a tensor to a PIL Image using a colormap."""
        # Remove any singleton dimensions (e.g., [1, 1, H, W] -> [H, W])
        tensor = tensor.squeeze()

        # Convert the tensor to a NumPy array
        tensor_np = tensor.cpu().numpy()

        # Normalize the tensor to be in the range [0, 1]
        normalized_tensor = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())

        # Apply colormap and convert to an RGB image
        mapped_tensor = cm.get_cmap(cmap)(normalized_tensor)[:, :, :3]  # Keep only RGB, discard alpha if present

        # Convert the NumPy array to PIL Image and return
        return Image.fromarray((mapped_tensor * 255).astype(np.uint8))

    def generateEntropyPlot(self, entropies: list[float], width: int, height: int) -> Image:
        """Generates and returns an entropy plot as a PIL Image."""
        # Explicitly specify figure size and DPI for better control
        dpi = 100
        figWidth = width / dpi
        figHeight = height / dpi
        fig, ax = plt.subplots(figsize=(figWidth, figHeight), dpi=dpi)

        ax.plot(entropies, color='blue')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Entropy')

        # Use plt.subplots_adjust as an alternative to plt.tight_layout
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def saveFieldToVideo(self,
                         fieldName: str,
                         outputFilePath: str,
                         dim: Optional[(int, int)] = None,
                         fps: int = 60):
        """Saves the field's components and entropy to a video file."""

        assert fieldName in self.fields, "Target field does not exist in this simulation"
        field: list[Field] = self.fields[fieldName]

        if dim is None:
            width, height = dim = (2 * field[0].resolution, 2 * field[0].resolution)
        print(len(field))
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoDims = (width, height)  # Assuming each component and the entropy plot are of equal height
        video = cv2.VideoWriter(outputFilePath, fourcc, fps, videoDims)
        entropies: list[float] = []

        # For each timestep, decompose the field, generate images, and write to video
        for timestep, f in enumerate(field):
            entropies.append(f.calculateEntropy())

            # Decompose the tensor into its components
            absField = torch.abs(f.tensor)
            realField = torch.real(f.tensor)
            imagField = torch.imag(f.tensor)

            # Convert each component to an image
            absImg = self.tensorToImage(absField, 'twilight_shifted')
            realImg = self.tensorToImage(realField, 'cool')
            imagImg = self.tensorToImage(imagField, 'spring')

            # Generate the entropy plot for the current timestep
            entropyPlot = self.generateEntropyPlot(entropies, width // 2, height // 2)

            # Combine component images and entropy plot into one image
            combinedImg = Image.new('RGB', (width, height))
            combinedImg.paste(absImg, (0, 0))
            combinedImg.paste(realImg, (width // 2, 0))
            combinedImg.paste(imagImg, (0, height // 2))
            combinedImg.paste(entropyPlot, (width // 2 , height // 2))

            # Convert PIL Image to OpenCV format and write to video
            combinedImg_cv = cv2.cvtColor(np.array(combinedImg), cv2.COLOR_RGB2BGR)
            video.write(combinedImg_cv)

        # Release the video writer
        video.release()


def genField0Vid() -> None:
    sim: Simulation = Simulation()
    fieldName: str = "field_0"
    sim.loadSimField(simDir="./",
                     fieldName=fieldName)
    sim.saveFieldToVideo(fieldName=fieldName,
                         outputFilePath="./simRender.mp4",
                         fps=30)
