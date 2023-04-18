#%%
import torch

x = torch.tensor(3.0, requires_grad=True)
#%%
y1 = 5 * x
y2 = 3 * x
y1_no_grad = y1.detach()
y2_no_grad = y2.detach()
z = y1 * y2_no_grad
z.backward()
print(z)
print(x.grad)