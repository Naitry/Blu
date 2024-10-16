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


def arrayToTextColored(arr: np.ndarray,
                       width: int,
                       height: int) -> str:
    """
    Converts a numpy array into a detailed, colorized text representation using an expanded range of ASCII characters.

    Args:
        arr (np.ndarray): The array to convert, assumed to be in the range [0, 255].
        width (int): The target width of the text representation.
        height (int): The target height of the text representation.

    Returns:
        str: The detailed, colorized text representation of the array.
    """
    # Normalize the array only if necessary
    if arr.max() != 0:
        arr = arr / arr.max()

    # Calculate the adjusted width and height based on the aspect ratio of the characters
    char_aspect_ratio: float = 0.5
    adjusted_width: int = width
    adjusted_height: int = int(height * char_aspect_ratio)

    print(arr.size)
    print(arr)
    # Resize the array to the target dimensions
    img: Image.Image = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((adjusted_width,
                      adjusted_height),
                     Image.NEAREST)
    arr = np.array(img)

    chars: list[str] = BLU_pixel_chars
    spectrum: list[str] = BLU_color_spectrum

    # convert the resized array to a colorized text representation
    lines: str = ""

    # iterate through each row in the image
    for row in arr:
        # iterate through each pixel in the row
        for pixel in row:
            # map the pixel intensity to a color in the spectrum
            colorIndex: int = int(pixel / 255 * (len(spectrum) - 1))
            # map the color index to the name of the color
            colorName: str = spectrum[colorIndex]
            # map the pixel intensity to a char
            char: str = chars[int(pixel / 255 * (len(chars) - 1))]
            # use paintStr to apply the color
            coloredChar = paintStr(char, colorName)
            # append the char to the the -
            lines += coloredChar
        # newline
        lines += "\n"

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
