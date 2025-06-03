import torch
from torch import nn

inputs = torch.tensor([3.0, 4.0, 5.0]).double(); inputs.requires_grad = True
targets = torch.tensor([3.3, 4.2, 5.1]).double(); targets.requires_grad = False
print("Inputs:", inputs)
print("Targets:", targets)

loss = nn.MSELoss()(inputs, targets)
loss.backward()
print("MSELoss:", loss.item(), inputs.grad)

inputs.grad.zero_()
loss = nn.CrossEntropyLoss()(inputs, targets)
loss.backward()
print("CrossEntropyLoss:", loss.item(), inputs.grad)

a = torch.tensor([1.3]).double(); a.requires_grad = True
b = torch.nn.GELU(approximate='tanh')(a)
b.backward()
print("GELU:", b.item(), a.grad.item())

c = torch.tensor([1.3]).double(); c.requires_grad = True
d = torch.nn.Sigmoid()(c)
d.backward()
print("Sigmoid:", d.item(), c.grad.item())
