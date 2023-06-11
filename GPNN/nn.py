import torch
import numpy as np

"""
Memory issue:

Overlapping patches create big matrices. 
7x7 patches on a 200x200 image results in a matrix of shape of around (7,7, 194, 194, 3). 
We want to compare to matrices of this shape so the resulting matrix is huge: ** $194^4$ =~ 1.5 billion elements**

Addressing the memory issue:

Our approach: $(x-y)^2 = x^2 + xy + y^2$

Their approach: ' Furthermore, GPNN exploits the fact that each operation uses only 1 row/column, and stores only small parts of the matrix at a time.'
"""
def squared_error(x, y): # Torch cdist alternative - less memory, longer runtime
     x_norm = (x**2).sum(-1).unsqueeze(-1)
     y_norm = (y**2).sum(-1).unsqueeze(-2)
     xy = torch.einsum('ijk,ilk->ijl', x, y)
     dist = x_norm + y_norm - 2.0 * xy
     return torch.clamp(dist, 0.0, torch.inf)

# Compute the distance between two images (patches), CPU implementation
def nearest_neighbors(list1, list2, alpha):

    num_images1, height1, width1, channels1 = list1.shape
    num_images2, height2, width2, channels2 = list2.shape

    mse_values = np.zeros((num_images1, num_images2))

    for i in range(num_images1):
        for j in range(num_images2):
            mse = np.mean((list1[i] - list2[j]) ** 2)
            mse_values[i, j] = mse

    mse_values = mse_values / ((np.min(mse_values, axis=1))[:, np.newaxis] + alpha)

    nn = np.argmin(mse_values, axis=1)

    return nn

# Same as above, in Torch
def nearest_neighbors_pytorch(list1, list2, alpha, device):

    list1 = torch.from_numpy(list1.astype(np.float16)).to(device)
    list2 = torch.from_numpy(list2.astype(np.float16)).to(device)

    num_images1, height1, width1, channels1 = list1.shape
    num_images2, height2, width2, channels2 = list2.shape

    list1_reshape = list1.view(num_images1, -1)
    list2_reshape = list2.view(num_images2, -1)

    list1_reshape = list1_reshape.unsqueeze(1)
    list2_reshape = list2_reshape.unsqueeze(0)

    # Normalized by 256, else the distance blows up to infinity
    distances = squared_error(list1_reshape/256, list2_reshape/256)

    mse_values = torch.mean(distances, dim=1)

    mse_values = mse_values / (torch.min(mse_values, dim=0, keepdim=True).values + alpha)

    nn = torch.argmin(mse_values, axis=1)

    return nn.cpu().numpy()
