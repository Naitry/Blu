import torch

# 2d basic laplacian first order finite differences
def laplacian(field: torch.Tensor,
			  delta: float):
	# Compute the Laplacian of the field
	laplace = (-4 * field
			   + torch.roll(field,
							shifts=1,
							dims=0)
			   + torch.roll(field,
							shifts=-1,
							dims=0)
			   + torch.roll(field,
							shifts=1,
							dims=1)
			   + torch.roll(field,
							shifts=-1,
							dims=1))
	return laplace / (delta ** 2)


def laplacianFourthOrder(field: torch.Tensor,
						 delta: float) -> torch.Tensor:
	"""
	Compute the Laplacian of the field using a fourth-order central difference in n-dimensions.
	Applies Dirichlet boundary conditions (field = 0 at boundaries).

	:param: field: The input field as an n-dimensional torch tensor.
	:param: delta: The spacing between points in the field.
	:return: The Laplacian of the field as an n-dimensional torch tensor.
	"""
	originalShape = field.shape
	laplace = torch.zeros_like(field)

	for dim in range(field.dim()):
		# Apply boundary conditions for the current dimension
		field.index_fill_(dim,
						  torch.tensor([0, originalShape[dim] - 1],
									   device=field.device),
						  0)

		# Fourth-order central difference for the current dimension
		laplace.add_(-1 * torch.roll(field,
									 shifts=2,
									 dims=dim)
					 - 1 * torch.roll(field,
									  shifts=-2,
									  dims=dim)
					 + 16 * torch.roll(field,
									   shifts=1,
									   dims=dim)
					 + 16 * torch.roll(field,
									   shifts=-1,
									   dims=dim)
					 - 30 * field)

	laplace.divide_(12 * delta ** 2)

	# Reinforce boundary conditions after computing the Laplacian
	for dim in range(field.dim()):
		laplace.index_fill_(dim,
							torch.tensor([0, originalShape[dim] - 1],
										 device=field.device),
							0)

	return laplace
