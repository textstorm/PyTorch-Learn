
import torch

#Tensor
x = torch.Tensor(5, 3)
print x
x = torch.rand(5, 3)
print x
print x.size()

#operation
y = torch.rand(5, 3)
print x + y               #syntax 1
print torch.add(x, y)     #syntax 2
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print result              #synatax 3
y.add_(x)
print y                   #syntax 4

#numpy bridge
#when a changes, b also changes
a = torch.ones(5)
print a
b = a.numpy()
print b
a.add_(1)
print a
print b

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print a
print b

#cuda tensors
if torch.cuda.is_available():
  x = x.cuda()
  y = y.cuda()
  x + y

#autograd
#Variable
from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print x

y = x + 2
print y
print y.grad_fn

z = y * y * 3
out = z.mean()
print z, out

#Gradients
out.backward()
print x.grad

#more autograde
x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
  y = y * 2

print y

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print x.grad