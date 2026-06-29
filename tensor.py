import torch
import numpy as np

# make a numpy array
arr = np.array([1.0, 2.0, 3.0])

# convert to a tensor
tensor = torch.tensor(arr)

print("numpy:", arr)
print("tensor:", tensor)
print("tensor + 1:", tensor + 1)