
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

一般使用方法是：首先创建tensor，然后用Variable包装该tensor，指定requires_grad=True，再进行函数计算，最后使用backward()自动求导。如果求导对象是scalar，则无需传参，如果求导对象有多个元素，则应传grad_output参数，指定输出对象。如果查看导数，使用变量的grad属性(.grad)查看

三、神经网络(nn)
