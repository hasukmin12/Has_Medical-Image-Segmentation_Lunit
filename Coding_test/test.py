import torch

y = torch.tensor([['nan', 3],
                  [3, 5.5]])

print(y[0][0])
