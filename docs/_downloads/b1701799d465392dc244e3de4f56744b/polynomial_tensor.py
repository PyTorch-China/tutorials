# -*- coding: utf-8 -*-
"""
PyTorch：张量(Tensors)
------------

一个三次多项式，通过最小化欧几里得距离的平方来训练预测从 :math:`-\pi` 到 :math:`\pi` 的 :math:`y=\sin(x)`。

该实现使用 PyTorch 张量手动计算前向传递、损失和反向传递。

PyTorch 张量基本上与 numpy 数组相同：它不了解深度学习、计算图或梯度，只是用于任意数值计算的通用n维数组。

numpy 数组和 PyTorch 张量之间最大的区别是， PyTorch 张量可以在 CPU 或 GPU 上运行。要在 GPU 上运行操作，只需将张量转换为 cuda 数据类型。
"""

import torch
import math


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
