import os
import shutil
import numpy as np

from PIL import Image


def arrayToText(arr: np.ndarray, width: int, height: int) -> str:
    """
    Converts a numpy array into a text representation.

    Args:
        arr (np.ndarray): The array to convert.
        width (int): The target width of the text representation.
        height (int): The target height of the text representation.

    Returns:
        str: The text representation of the array.
    """
    chars = " .:-=+*#%@"
    if arr.max() > 0:
        normalized_arr = arr / arr.max()
    else:
        normalized_arr = arr

    # Calculate aspect ratio of a character in terminal
    char_aspect_ratio = 0.5  # This is an assumption; may need to adjust based on your terminal

    # Adjust width and height based on character aspect ratio
    adjusted_width = width
    adjusted_height = int(height * char_aspect_ratio)

    # Resize the array
    img = Image.fromarray((normalized_arr * 255).astype(np.uint8))
    img = img.resize((adjusted_width, adjusted_height), Image.NEAREST)
    resized_arr = np.array(img)

    # Convert array to text
    lines = ""
    for row in resized_arr:
        line = "".join(chars[min(pixel * len(chars) // 256, len(chars) - 1)] for pixel in row)
        lines += line + "\n"
    return lines


def getTerminalSize() -> tuple[int, int]:
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
