
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

#neural network
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)

    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]     # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      out_features *= s
    return num_features

net = Net()
print net