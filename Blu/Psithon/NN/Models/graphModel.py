import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ComplexGCN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ComplexGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 500)  # Adjust the number of features
        self.conv2 = GCNConv(500, output_dim)  # Final layer matches output dimensions

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First convolution
        x = self.conv1(x, edge_index)
        x = torch.view_as_complex(x)  # Convert to complex if needed
        x = F.relu(x)

        # Second convolution
        x = self.conv2(x, edge_index)
        x = torch.view_as_complex(x)  # Convert to complex if needed

        return x  # Return the complex tensor of desired shape

# Example usage assuming proper data preparation
# model = ComplexGCN(input_dim=data.num_features, output_dim=1000*1000)
# out = model(data)
# out = out.view(1000, 1000)  # Reshape to the desired output format
