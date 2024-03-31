import os
import shutil
import numpy as np

from PIL import Image


def arrayToText(arr: np.ndarray,
                width: int,
                height: int) -> str:
    """
    Converts a numpy array into a text representation and prints it.

    Args:
        arr (np.ndarray): The array to convert.
        width (int): The target width of the text representation.
        height (int): The target height of the text representation.
    """
    # Mapping from value ranges to characters (from low to high values)
    chars = " .:-=+*#%@"
    if arr.max() > 0:  # Normalize only if the max is greater than 0
        normalized_arr = arr / arr.max()
    else:
        normalized_arr = arr

    # Resize the array to fit the terminal dimensions
    # Using PIL Image for resizing for simplicity
    img = Image.fromarray(np.uint8(normalized_arr * 255))
    img = img.resize((width, height), Image.NEAREST)
    resized_arr = np.asarray(img)
    lines: str = ""
    # Convert array values to characters
    for row in resized_arr:
        line: str = "".join([chars[int((len(chars) - 1) * pixel / 255)] for pixel in row])
        line += "\n"
        lines += line
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
    Clears the terminal screen.
    """
    # Windows
    if os.name == 'nt':
        os.system('cls')
    # Linux and MacOS
    else:
        os.system('clear')
