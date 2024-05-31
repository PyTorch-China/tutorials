"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
**Transforms** ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Transforms
===================

Data does not always come in its final processed form that is required for
training machine learning algorithms. We use **transforms** to perform some
manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters -``transform`` to modify the features and
``target_transform`` to modify the labels - that accept callables containing the transformation logic.
The `torchvision.transforms <https://pytorch.org/vision/stable/transforms.html>`_ module offers
several commonly-used transforms out of the box.

The FashionMNIST features are in PIL Image format, and the labels are integers.
For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors.
To make these transformations, we use ``ToTensor`` and ``Lambda``.


转换(transform)
==========

数据并不总是以训练机器学习算法所需的最终处理形式呈现。我们使用**transform**来对数据进行一些处理，使其适用于训练。

所有 TorchVision 数据集都有两个参数 - `transform` 用于修改特征，`target_transform` 用于修改标签 
- 它们接受包含转换逻辑的可调用对象。`torchvision.transforms <https://pytorch.org/vision/stable/transforms.html>`_ 模块提供了几种常用的转换。

FashionMNIST 的特征是以 PIL 图像格式呈现的，标签是整数。对于训练，我们需要将特征转换为归一化的张量，
将标签转换为编码的张量。为了进行这些转换，我们使用了 ``ToTensor`` 和 ``Lambda``。
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(
        10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

#################################################
# ToTensor()
# -------------------------------
#
# `ToTensor <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor>`_
# 将 PIL 图像或 NumPy ``ndarray`` 转换为 ``FloatTensor``，并将图像的像素强度值缩放到范围 [0., 1.]。

##############################################
# Lambda Transforms
# -------------------------------
#
# Lambda transforms apply any user-defined lambda function. Here, we define a function
# to turn the integer into a one-hot encoded tensor.
# It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls
# `scatter_ <https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html>`_ which assigns a
# ``value=1`` on the index as given by the label ``y``.
# Lambda transforms 应用任何用户定义的 lambda 函数。这里，我们定义一个函数将整数转换为独热编码的张量。
# 它首先创建一个大小为 10（我们数据集中标签的数量）的零张量，然后调用 `scatter_ <https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html>`_，
# 在由标签 ``y`` 指定的索引上赋值为 ``1``。

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

######################################################################
# --------------
#

#################################################################
# 延伸阅读
# ~~~~~~~~~~~~~~~~~
# - `torchvision.transforms API <https://pytorch.org/vision/stable/transforms.html>`_
