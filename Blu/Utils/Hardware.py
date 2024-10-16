import torch


def getDeviceList() -> list[torch.device]:
    devices = []

    # Check for CPU
    devices.append('cpu')

    # Check for CUDA (GPU) devices
    if torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        for i in range(num_cuda_devices):
            devices.append(f'cuda:{i}')

    # Check for MPS (Apple Silicon) device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')

    return devices


def getDevice() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
