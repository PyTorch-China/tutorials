"""
PyTorch 创建神经网络
====================================
深度学习使用人工神经网络(模型),这是由许多互连单元层组成的计算系统。通过将数据传递到这些互连单元,
神经网络能够学习如何近似将输入转换为输出所需的计算。在PyTorch中,可以使用 ``torch.nn`` 包构建神经网络。

介绍
------------
PyTorch 提供了优雅设计的模块和类来帮助您创建和训练神经网络,包括 ``torch.nn``。
一个 ``nn.Module`` 中有层(layers)、以及一个返回 ``output`` 的 ``forward(input)`` 方法。

在本教程中,我们将使用 ``torch.nn`` 来定义一个用于 
`MNIST 数据集 <hhttps://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST>` 的神经网络。

环境设置
-----
在开始之前,如果还没有安装 ``torch``的话,我们需要先安装它。

::

   pip install torch


"""


######################################################################
# 具体步骤
# -----
#
# 1. 导入加载数据所需的所有必要库
# 2. 定义和初始化神经网络
# 3. 指定数据如何通过你的模型
# 4. [可选] 通过你的模型传递数据进行测试
#
# 1. 导入加载数据所需的必要库
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 对于本教程,我们将使用 ``torch`` 及其子模块 ``torch.nn`` 和 ``torch.nn.functional``。
#

import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################################
# 2. 定义和初始化神经网络
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 我们的网络将识别图像。我们将使用PyTorch内置的卷积过程。卷积将每个图像元素与其局部邻居相加,
# 并由一个小矩阵(内核)加权,该内核可帮助我们从输入图像中提取某些特征(如边缘检测、锐利度、模糊度等)。
#
# 定义模型的 ``Net`` 类有两个要求。第一是编写一个引用 ``nn.Module`` 的 __init__ 函数。
# 在这个函数中,你定义神经网络中的全连接层。
#
# 使用卷积,我们将定义我们的模型以接受1个输入图像通道,并输出与我们的目标相匹配的10个标签,表示0到9的数字。
# 这个算法由你自己创建,我们将遵循标准的MNIST算法。
#

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # 第一个2D卷积层,接受1个输入通道(图像),
      # 输出32个卷积特征,使用3x3的方形核
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      # 第二个2D卷积层,接受32个输入层,
      # 输出64个卷积特征,使用3x3的方形核
      self.conv2 = nn.Conv2d(32, 64, 3, 1)

      # 设计为确保相邻像素要么全为0,要么全为激活
      # 具有一定输入概率
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

      # 第一个全连接层
      self.fc1 = nn.Linear(9216, 128)
      # 第二个全连接层,输出我们的10个标签
      self.fc2 = nn.Linear(128, 10)

my_nn = Net()
print(my_nn)


######################################################################
# 我们已经完成了神经网络的定义,现在我们必须定义数据如何通过它。
#
# 3. 指定数据如何通过你的模型
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 当你使用PyTorch构建模型时,你只需要定义 ``forward`` 函数,它将数据传递到计算图(即我们的神经网络)中。
# 这将代表我们的前向算法。
#
# 你可以在 ``forward`` 函数中使用任何张量操作。
#

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    # x表示我们的数据
    def forward(self, x):
      # 将数据传递给conv1
      x = self.conv1(x)
      # 对x使用整流线性激活函数
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # 对x运行最大池化
      x = F.max_pool2d(x, 2)
      # 将数据传递给dropout1
      x = self.dropout1(x)
      # 展平x,start_dim=1
      x = torch.flatten(x, 1)
      # 将数据传递给 ``fc1``
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # 对x应用softmax
      output = F.log_softmax(x, dim=1)
      return output


######################################################################
# 4. [可选] 通过你的模型传递数据进行测试
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 为了确保我们得到期望的输出,让我们通过一些随机数据测试我们的模型。
#

# 等同于一个随机的28x28图像
random_data = torch.rand((1, 1, 28, 28))

my_nn = Net()
result = my_nn(random_data)
print (result)


######################################################################
# 这个结果张量中的每个数字都等同于随机张量所关联的标签的预测。
#
# 祝贺你!你已经成功地在PyTorch中定义了一个神经网络。
#
# 学习更多
# ----------
#
# 查看这些其他教程以继续学习:
#
# - `PyTorch 中 state_dict 是什么 <https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html>`__
# - `PyTorch 保存和加载模型用于推理 <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html>`__