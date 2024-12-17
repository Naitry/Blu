import torch

# Default simulation size
BLU_PSITHON_defaultDimensions: int = 2
BLU_PSITHON_defaultRank: int = 1
BLU_PSITHON_defaultResolution: int = 1000

# Primary data type for the field
BLU_PSITHON_defaultDataType: torch.dtype = torch.cfloat

# Floating point data type which will be able to represent one of the complex components
BLU_PSITHON_defaultDataTypeComponent: torch.dtype = torch.float16
