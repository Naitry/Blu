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
    Converts a numpy array into a detailed, colorized text representation using an expanded range of ASCII characters.

    Args:
        tensor (torch.Tensor): The tensor to convert, assumed to be in the range [0, 255].
        width (int): The target width of the text representation.
        height (int): The target height of the text representation.

    Returns:
        str: The detailed, colorized text representation of the array.
    """
    chars = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    color_spectrum = ["red", "orange", "yellow", "green", "cyan", "blue", "indigo", "violet"]

    arr: np.ndarray = tensor.cpu().numpy()

    # Normalize the array only if necessary
    if tensor.max() != 0:
        normalized_arr = arr / arr.max()

    # Calculate the adjusted width and height based on the aspect ratio of the characters
    char_aspect_ratio = 0.5
    adjusted_width = width
    adjusted_height = int(height * char_aspect_ratio)

    # Resize the array to the target dimensions
    img = Image.fromarray((normalized_arr * 255).astype(np.uint8))
    img = img.resize((adjusted_width, adjusted_height), Image.NEAREST)
    resized_arr = np.array(img)

    # Convert the resized array to a colorized text representation
    lines = ""
    for row in resized_arr:
        for pixel in row:
            # Map the pixel intensity to a color in the spectrum
            color_index = int(pixel / 255 * (len(color_spectrum) - 1))
            color_name = color_spectrum[color_index]
            char = chars[int(pixel / 255 * (len(chars) - 1))]
            # Use paintStr to apply the color
            colored_char = paintStr(char, color_name)
            lines += colored_char
        lines += "\n"

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
