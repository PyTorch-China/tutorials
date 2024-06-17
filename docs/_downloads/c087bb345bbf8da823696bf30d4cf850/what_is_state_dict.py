"""
PyTorch 中 state_dict 是什么
===============================
在 PyTorch 中,一个 ``torch.nn.Module`` 模型的可学习参数(即权重和偏置)包含在模型的参数中
(通过 ``model.parameters()`` 访问)。``state_dict`` 只是一个 Python 字典对象,它将每一层映射到其参数张量。

介绍
------------
如果使用 PyTorch 保存或加载模型,``state_dict`` 就是一个不可或缺的实体。
由于 ``state_dict`` 对象是 Python 字典,它们可以很容易地被保存、更新、修改和恢复,使 PyTorch 模型和优化器更好的做到了模块化。
请注意,只有具有可学习参数的层(卷积层、线性层等)和已注册的缓冲区(BatchNorm running_mean)在模型的 ``state_dict`` 中有条目。
优化器对象(``torch.optim``)也有一个 ``state_dict``,它包含了优化器状态的信息,以及使用的超参数。
在本教程中,我们将看到如何在一个简单的模型中 ``state_dict`` 是如何使用的。

环境设置
-----
在开始之前,如果还没有安装 ``torch``,我们需要先安装它。

.. code-block:: sh

   pip install torch

"""



######################################################################
# 具体步骤
# -----
#
# 1. 导入加载数据所需的所有必要库
# 2. 定义和初始化神经网络
# 3. 初始化优化器
# 4. 访问模型和优化器的 ``state_dict``
# 
# 1. Import necessary libraries for loading our data
# 1. 导入加载数据所需的必要库
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 对于本教程,我们将使用 ``torch`` 及其子模块 ``torch.nn`` 和 ``torch.optim``。
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


######################################################################
# 2. 定义并初始化神经网络
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 为了演示，我们将创建一个用于训练图像的神经网络。要了解更多信息，请参阅定义神经网络的教程。
#

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)


######################################################################
# 3. 初始化优化器
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 我们使用 SGD 优化器
# 

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 4. 访问模型和优化器的 ``state_dict``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now that we have constructed our model and optimizer, we can understand
# what is preserved in their respective ``state_dict`` properties.
# 
# 现在我们已经构建了模型和优化器,我们可以了解它们各自的 ``state_dict`` 属性中保存了什么。
#

print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print()

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


######################################################################
# 这些信息对于将来保存和加载模型和优化器很有用。
#
# 祝贺你!你已经成功使用了 PyTorch 中的 ``state_dict``。
#
# 学习更多
# ----------
#
# 查看这些其他教程,继续你的学习:
#
# - `在 PyTorch 中保存和加载模型用于推理 <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html>`__
# - `在 PyTorch 中保存和加载通用检查点 <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>`__