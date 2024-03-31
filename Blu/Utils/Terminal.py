import os
import shutil


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
