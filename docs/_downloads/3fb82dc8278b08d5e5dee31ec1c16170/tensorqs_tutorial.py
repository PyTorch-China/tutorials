"""
`基础知识 <intro.html>`_ ||
`快速入门 <quickstart_tutorial.html>`_ ||
**张量** ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`构建神经网络 <buildmodel_tutorial.html>`_ ||
`自动微分 <autogradqs_tutorial.html>`_ ||
`优化模型参数 <optimization_tutorial.html>`_ ||
`保存和加载模型 <saveloadrun_tutorial.html>`_

张量
============

类似于数组和矩阵，张量也是一种特定的数据结构。在PyTorch中，我们使用张量对一个模型的参数、输入和输出进行编码。

张量的结构类似于 [Numpy](https://numpy.org/)中的ndarrays，而张量可以运行在GPU及其他相似的硬件加速器上。
事实上，为了减少数据的拷贝，张量和NumPy arrays在底层常常共享同一块内存(`bridge-to-np-label`{.interpreted-text role="ref"})。
在自动微分(automatic differentiation)的过程中也使用张量进行优化(在后续[Autograd](autogradqs_tutorial.html)章节可以看到更多有关内容)。
如果已经对ndarrays十分熟悉了，那对张量的API也可以运用自如。如果还不熟悉，下面的教程会帮助你上手。
"""

import torch
import numpy as np


######################################################################
# 初始化张量
# ~~~~~~~~~~~~~~~~~~~~~
#
# 我们可以通过多种方式创建一个张量，例如：
#
# **使用数据创建**
#
# 通过已定义的数据可以直接创建出来张量，创建时会自动推断数据类型。

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

######################################################################
# **使用NumPy array创建**
#
# 可以使用NumPy array创建张量(反之亦可`bridge-to-np-label`{.interpreted-text role="ref"})

np_array = np.array(data)
x_np = torch.from_numpy(np_array)


###############################################################
# **使用已有张量创建**
#
# 新的张量会保留原张量的属性(形状，数据类型)，除非创建时显示声明。

x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

# overrides the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")


######################################################################
# **通过随机或常量创建**
#
# `shape` 描述了张量的维度，在下面的方法调用时，通过它来声明创建张量的维度。

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


######################################################################
# --------------
#

######################################################################
# 张量的属性
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 张量的属性保存了其形状，数据类型，以及其存储设备类型。

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


######################################################################
# --------------
#

######################################################################
# 张量操作
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# 张量有超过100个操作方法，包括算数、线性代数、矩阵操作（转置、索引、切片）、采样等，都在[这里](https://pytorch.org/docs/stable/torch.html)有详细的描述。
#
# 每个操作都可以在GPU上运行（通常比在CPU上速度更快)。如果你在使用Colab，可以通过修改Runtime \> Change runtime type \> GPU来分配一个GPU。
#
# 默认情况下张量是在CPU上创建的，可以通过`.to`方法将张量显示的转移到GPU上（如果GPU在你的环境里可用的话）。需要注意的是，在不同设备间复制大型张量需要消耗大量内存，并且耗时较长。


# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")


######################################################################
# 尝试下列操作，如果你已经对NumPy API十分熟悉，上手张量API将会很简单。

###############################################################
# **类似numpy的索引和切片操作**

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

######################################################################
# **连接张量**
# 你可以使用 `torch.cat` 沿着给定的维度连接一系列张量。另一个张量连接操作符，
# 与 `torch.cat` 稍有不同，请参阅 `torch.stack` <https://pytorch.org/docs/stable/generated/torch.stack.html>`__。

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


######################################################################
# **运算操作**

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


######################################################################
# **单个元素的张量**
# 在聚合运算场景中，你可能会得到一个单元素的张量，可使用`item()`将其传唤为Python数值。

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


######################################################################
# **原地操作**
# 修改张量中的原值操作称为原地操作。它们以 `_` 后缀表示。例如：`x.copy_(y)`，`x.t_()`，会改变 `x`。

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

######################################################################
# .. 提示::
#      原地操作节省了一些内存，但在计算导数时可能会出现问题，因为会立即丢失历史记录。因此，不建议使用它们。


######################################################################
# --------------
#


######################################################################
# .. _bridge-to-np-label:
#
# 与NumPy转换
# ~~~~~~~~~~~~~~~~~
# 张量在使用CPU时，可与NumPy arrays共享内存空间，修改其中一个会同步映射到另一个上。


######################################################################
# 张量转为NumPy array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

######################################################################
# 对于张量的修改体现到了NumPy array上。

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


######################################################################
# NumPy array转为张量
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n = np.ones(5)
t = torch.from_numpy(n)

######################################################################
# NumPy array转为张量
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
