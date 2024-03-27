# Typing
from typing import Tuple, Dict

# Files imports
import csv
from importlib.resources import files
from importlib.abc import Traversable

# Global dictionary to store color data
BLU_CM_ColorHexMap: Dict[str, str] = {}

BLU_CM_ResourcePath: Traversable = files("data") / 'colornames.csv'

with BLU_CM_ResourcePath.open(mode='r',
							  encoding='utf-8') as file:
	csvReader = csv.reader(file)
	next(csvReader)
	for row in csvReader:
		if len(row) >= 2:
			name: str = row[0].lower().strip()
			hexValue: str = row[1].strip()
			goodName: str = row[2].lower().strip() if len(row) > 2 else ""
			key: str = goodName if goodName and goodName != "x" else name
			BLU_CM_ColorHexMap[key] = hexValue


def printFirstNColors(n: int) -> None:
	"""Print the first n entries in the color_hex_map."""
	for i, (key, value) in enumerate(BLU_CM_ColorHexMap.items()):
		if i >= n:
			break
		print(f"Key: {key}, Value: {value}")


def printSpecificColors(colorNames: Tuple[str, ...]) -> None:
	"""Print specific color entries given their names."""
	for color in colorNames:
		hexValue = BLU_CM_ColorHexMap.get(color.lower().strip())
		if hexValue:
			print(f"Key: {color}, Value: {hexValue}")
		else:
			print(f"Key: {color} not found in map.")


def getHexValue(colorName: str) -> str:
	"""Get the hex value of a color given its name."""
	return BLU_CM_ColorHexMap.get(colorName.lower().strip(),
								  "#EEEEEE")


def hexToColor(hex_string: str) -> tuple[int, ...]:
	"""Convert a hex color string to an RGB tuple."""
	hex_string = hex_string.lstrip('#')
	if len(hex_string) != 6:
		raise ValueError("Hex color string must be in the format '#RRGGBB'")
	return tuple(int(hex_string[i:i + 2],
					 16) for i in (0, 2, 4))
