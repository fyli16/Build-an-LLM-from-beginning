import torch 
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b 
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print("Gradient of loss with respect to w1:", grad_L_w1)
print("Gradient of loss with respect to b:", grad_L_b)
print('---')
loss.backward()
print("Gradient of loss with respect to w1 (using backward):", w1.grad)
print("Gradient of loss with respect to b (using backward):", b.grad)