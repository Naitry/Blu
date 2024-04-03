import h5py
import numpy as np


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
            mid_indices = tuple(map(lambda x: x // 2, data.shape))
            print(f"Contents of dataset '{datasetName}':\n{data}")
        else:
            print(f"Dataset '{datasetName}' not found in the file.")


# Example usage
listDatasets('/mnt/nfs/simulations/Run_24/field_0.hdf5')

# Example usage: reading and printing the 'real_0' dataset
readAndPrintDataset('/mnt/nfs/simulations/Run_24/field_0.hdf5', 'real_28500000')
