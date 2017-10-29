
一、组件及内容
1.torch：  a Tensor library like NumPy, with strong GPU support
2.torch.autograd：
3.torch.nn：神经网络库
4.torch.optim：优化算法库

二、autograd
1.Variable
Variable是中心类，他对tensor进行包装，调用.backward()可以自动求导
2.Function
每一个变量都有.grad_fn属性，引用创建变量的函数(除了变量由用户创建，此时grad_fn is None)
