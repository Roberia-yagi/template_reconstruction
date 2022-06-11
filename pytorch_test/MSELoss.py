import torch
import torch.nn as nn

a = torch.rand(3, 2)
b = a + 0.1
criterion = nn.MSELoss()
print(a)
print(b)
sum = torch.sum(torch.pow(a - b, 2))/6
print(criterion(a, b))
print(sum)