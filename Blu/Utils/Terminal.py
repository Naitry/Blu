from typing import List
import torch
import os
import shutil
import numpy as np

from PIL import Image

from Blu.Utils.TermColor import paintStr

BLU_pixel_chars: list[str] = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
BLU_color_spectrum: list[str] = ["red",
                                 "orange",
                                 "yellow",
                                 "green",
                                 "cyan",
                                 "blue",
                                 "indigo",
                                 "violet"]


def arrayToText(arr: np.ndarray,
                width: int,
                height: int) -> str:
    """
    Converts a numpy array into a detailed text representation using an expanded range of ASCII characters.

    Args:
        arr (np.ndarray): The array to convert, assumed to be in the range [0, 255].
        width (int): The target width of the text representation.
        height (int): The target height of the text representation.

    Returns:
        str: The detailed text representation of the array.
    """

    # Normalize the array only if necessary
    arr = arr / arr.max()

    # Calculate the adjusted width and height based on the aspect ratio of the characters
    char_aspect_ratio: float = 0.5
    adjusted_width: int = width
    adjusted_height: int = int(height * char_aspect_ratio)

    # Resize the array to the target dimensions
    print(arr.size)
    img: Image.Image = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((adjusted_width,
                      adjusted_height),
                     Image.NEAREST)
    arr = np.array(img)

    # Convert the resized array to a text representation
    lines: str = ""

    chars: list[str] = BLU_pixel_chars
    for row in arr:
        line = "".join((chars[int(pixel / 255 * (len(chars) - 1))]) for pixel in row)
        lines += line + "\n"

    return lines


def tensorToTextColored(tensor: torch.Tensor,
                        width: int,
                        height: int) -> str:
    """
    Converts a PyTorch tensor into a detailed, colorized text representation using an expanded range of ASCII characters.
    Args:
        tensor (torch.Tensor): The tensor to convert, assumed to be in the range [0, 255].
        width (int): The target width of the text representation.
        height (int): The target height of the text representation.
    Returns:
        str: The detailed, colorized text representation of the tensor.
    """
    # ensure the tensor is on CPU and in the correct format
    tensor: torch.Tensor = tensor.cpu().float()

    # normalize the tensor only if necessary
    if tensor.max() != 0:
        tensor = tensor / tensor.max()

    # Calculate the adjusted width and height based on the aspect ratio of the characters
    charAspectRatio: float = 0.5
    height = int(height * charAspectRatio)

    # Resize the tensor to the target dimensions
    # Reshape the tensor if necessary
    if tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)  # Remove the first dimension if it's 1
    elif tensor.dim() > 2:
        tensor = tensor.view(-1, tensor.size(-1))  # Flatten all dimensions except the last

    print(f"Reshaped tensor size: {tensor.size()}")

    # Resize the tensor to the target dimensions
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    tensor = torch.nn.functional.interpolate(tensor,
                                             size=(height,
                                                   width),
                                             mode='bicubic',
                                             align_corners=False)
    tensor = tensor.squeeze()  # Remove any extra dimensions

    print(f"Resized tensor size: {tensor.size()}")

    # Ensure the tensor is 2D
    if tensor.dim() > 2:
        tensor = tensor.view(height,
                             width)

    # Clip values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)

    print("C: ", tensor.shape)
    chars: List[str] = BLU_pixel_chars
    spectrum: List[str] = BLU_color_spectrum

    # Convert the resized tensor to a colorized text representation
    lines: str = ""
    count: int = 0
    print(len(tensor))
    print(len(tensor[0]))
    input()
    # Iterate through each row in the image
    for row in tensor:
        # Iterate through each pixel in the row
        for pixel in row:
            # Map the pixel intensity to a color in the spectrum
            colorIndex: int = int(pixel / 255 * (len(spectrum) - 1))
            # Map the color index to the name of the color
            colorName: str = spectrum[colorIndex]
            # Map the pixel intensity to a char
            char: str = chars[int(pixel / 255 * (len(chars) - 1))]
            # Use paintStr to apply the color
            coloredChar = paintStr(char, colorName)
            # Append the char to the line
            lines += coloredChar
        # Newline
        count += 1
        print(count)
        lines += "\n"
    print("FIELD CONVERTED")
    return lines


def getTerminalSize() -> tuple[int,
                               int]:
    """
    Get the size of the terminal.

    Returns:
        tuple[int, int]: A tuple containing the width (columns) and height (lines) of the terminal.
    """
    size: os.terminal_sizei = shutil.get_terminal_size(fallback=(80, 20))
    return (size.columns, size.lines)


def clearTerminal() -> None:
    """
    Clears the terminal
    """
    # Windows
    if os.name == 'nt':
        os.system('cls')
    # Linux and MacOS
    else:
        os.system('clear')
