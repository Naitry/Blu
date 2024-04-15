# Typing
from typing import Optional

# Imports and Paths
from importlib.abc import Traversable


def readMarkdownFile(filePath: str | Traversable) -> Optional[str]:
	"""
	Reads the content of a Markdown file and returns it as a single string.

	Args:
	  filePath (str): The path to the Markdown file.

	Returns:
	  Optional[str]: The content of the Markdown file as a single string, or None if the file could not be read.
	"""
	try:
		if type(filePath) is type(""):
			with open(file=filePath,
					  mode='r',
					  encoding='utf-8') as file:
				content: str = file.read()
			return content
		else:
			with filePath.open(mode='r',
							   encoding='utf-8') as file:
				content: str = file.read()
			return content
	except FileNotFoundError:
		print(f"File {filePath} not found.")
		return None
	except Exception as e:
		print(f"An error occurred: {e}")
	return None