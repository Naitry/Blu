from Blu.Color.ColorManager import hexToColor, getHexValue

BLU_TC_reset_color: str = "\033[0m"


def hexToAnsiForeGround(hexColor: str) -> str:
	"""Convert a hex color string to an ANSI foreground color string."""
	r, g, b = hexToColor(hexColor)
	return f"\033[38;2;{r};{g};{b}m"


def hexToAnsiBackGround(hexColor: str) -> str:
	"""Convert a hex color string to an ANSI background color string."""
	r, g, b = hexToColor(hexColor)
	return f"\033[48;2;{r};{g};{b}m"


def printC(text: str,
		   textColor: str,
		   bgColor: str = None) -> None:
	"""Print text in the specified color."""
	if bgColor:
		print(f"{hexToAnsiForeGround(getHexValue(textColor))}{hexToAnsiBackGround(getHexValue(bgColor))}{text}{BLU_TC_reset_color}")
	else:
		print(f"{hexToAnsiForeGround(getHexValue(textColor))}{text}{BLU_TC_reset_color}")


def paintStr(text: str,
			 textColor: str,
			 bgColor: str = None) -> str:
	"""Return a string that when printed will appear in the specified color."""
	if bgColor:
		return f"{hexToAnsiForeGround(getHexValue(textColor))}{hexToAnsiBackGround(getHexValue(bgColor))}{text}{BLU_TC_reset_color}"
	else:
		return f"{hexToAnsiForeGround(getHexValue(textColor))}{text}{BLU_TC_reset_color}"


def setColor(textColor: str,
			 bgColor: str = None) -> None:
	if bgColor:
		print(f"{hexToAnsiForeGround(getHexValue(textColor))}{hexToAnsiBackGround(getHexValue(bgColor))}",
			  end="")
	else:
		print(f"{hexToAnsiForeGround(getHexValue(textColor))}",
			  end="")


def resetColor() -> None:
	print(f"{BLU_TC_reset_color}")
