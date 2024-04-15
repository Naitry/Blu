from time import sleep
from Blu.Utils.Terminal import getTerminalSize, clearTerminal


def test_TerminalPrelude() -> None:
    print("Terminal Test 1:")
    print("Clearing terminal ...")
    sleep(3)


def test_1Terminal() -> None:
    clearTerminal()
    print("Terminal Test 1 complete, Terminal cleared!")
    pass


def test_2Terminal() -> None:
    print("Terminal Test 2:")
    print(f"Current terminal size: {getTerminalSize()}")
    print("Terminal Test 2 Complete")
    pass
