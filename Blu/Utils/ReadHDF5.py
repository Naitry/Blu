import h5py
import numpy as np
import torch
from typing import Dict


def listDatasets(filePath: str):
    """List the names of all datasets in the HDF5 file."""
    def printName(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name)

    with h5py.File(filePath, 'r') as file:
        file.visititems(printName)


def readAndPrintDataset(filePath: str,
                        datasetName: str):
    """Read and print the contents of a specified dataset."""
    with h5py.File(filePath, 'r') as file:
        # Ensure the dataset exists
        if datasetName in file:
            data = np.array(file[datasetName])
            # mid_indices = tuple(map(lambda x: x // 2, data.shape))
            print(f"Contents of dataset '{datasetName}':\n{data}")
        else:
            print(f"Dataset '{datasetName}' not found in the file.")


def listTimesteps(filePath: str,
                  prefix: str) -> list:
    """
    List all timesteps for datasets with a given prefix in an HDF5 file.

    Args:
        filePath: The path to the HDF5 file.
        prefix: The prefix to filter datasets by (e.g., 'real' or 'imaginary').

    Returns:
        A list of timesteps (as integers) in chronological order.
    """
    data: Dict[int, torch.tensor] = {}

    def filterDatasets(name: str,
                       obj: h5py.Dataset):
        if isinstance(obj, h5py.Dataset) and name.startswith(prefix):
            # Extract timestep from the dataset name
            _, timestep_str = name.split('_')
            try:
                timestep = int(timestep_str)
                data[timestep] = torch.tensor(np.array(obj))
            except ValueError:
                # Handle cases where the conversion fails
                print(f"Warning: Found dataset with non-integer timestep: {name}")

    with h5py.File(filePath, 'r') as file:
        file.visititems(filterDatasets)

    return sorted(data.items())


# Example usage
timesteps = listTimesteps('/mnt/nfs/simulations/Run_24/field_0.hdf5', 'real')
print(timesteps)


# Example usage
# listDatasets('/mnt/nfs/simulations/Run_24/field_0.hdf5')

# Example usage: reading and printing the 'real_0' dataset
# readAndPrintDataset('/mnt/nfs/simulations/Run_24/field_0.hdf5', 'real_28500000')
