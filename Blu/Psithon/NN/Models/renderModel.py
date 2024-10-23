import torch
from torchviz import make_dot
from Blu.Psithon.models.model import UNet  # Ensure your model is accessible


# Assuming your U-Net takes 3 channels input and outputs 2 classes
model = UNet(n_channels=3, n_classes=2)
x = torch.randn(1, 3, 256, 256)  # Adjust the size according to your needs

# Generate the graph
vis = make_dot(model(x), params=dict(list(model.named_parameters()) + [('input', x)]))
vis.render('unet_visualization', format='png')  # This saves the visualization as 'unet_visualization.png'
