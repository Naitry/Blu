# Typing
from typing import Optional, \
	List

# Imports and Paths
from pathlib import Path
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


def listFiles(directory: str,
			  fileExtension: str = None) -> List[Path]:
	"""
	Returns a list of files in the specified directory with the given file extension.

	Args:
	- directory (str): The path to the directory from which to list files.
	- file_extension (str): The file extension to filter by, including the leading dot (e.g., '.txt').

	Returns:
	- List[Path]: A list of Path objects representing the files with the specified extension in the directory.
	"""
	path = Path(directory)
	if fileExtension:
		return [file for file in path.iterdir() if file.is_file() and file.suffix == fileExtension]
	else:
		return [file for file in path.iterdir() if file.is_file()]


def listDirectories(directory: str) -> List[Path]:
	"""
	Returns a list of directories in the specified directory.

	Args:
	- directory (str): The path to the directory from which to list directories.

	Returns:
	- List[Path]: A list of Path objects representing the directories in the directory.
	"""
	path = Path(directory)
	return [d for d in path.iterdir() if d.is_dir()]
