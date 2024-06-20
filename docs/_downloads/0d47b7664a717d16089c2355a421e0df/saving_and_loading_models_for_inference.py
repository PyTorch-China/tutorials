"""
PyTorch 保存和加载模型
==================================================
在PyTorch中保存和加载模型有两种方法。
第一种是保存和加载 ``state_dict``，第二种是保存和加载整个模型。

简介
------------
使用 ``torch.save()`` 函数保存模型的 ``state_dict`` 为后续恢复模型提供较大的灵活性。
保存模型的推荐使用此方法，因为只需要保存训练好的模型的学习参数。

当保存和加载整个模型时，你使用Python  `pickle <https://docs.python.org/3/library/pickle.html>`__ 模块保存整个模块。
使用这种方法语法最直观，代码量最少。但这种方法的缺点是序列化的数据与保存模型时使用的特定类和目录结构绑定在一起。
原因是pickle不保存模型类本身，而是保存包含该类的文件的路径，该路径在加载时使用。
因此，当在其他项目中使用或重构后，代码可能会出现各种异常导致程序中断。

在本教程中，我们将展示两种方式如何在PyTorch中保存和加载模型。

环境设置
-----
在开始之前，如果还没有安装 ``torch``，我们需要先安装它。

::

   pip install torch


"""


######################################################################
# 具体步骤
# -----
# 
# 1. 导入加载数据所需的所有必要库
# 2. 定义和初始化神经网络
# 3. 初始化优化器
# 4. 通过 ``state_dict`` 保存和加载模型
# 5. 保存和加载整个模型
# 
# 1. 导入加载数据所需的必要库
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 对于本教程，我们将使用 ``torch`` 及其子模块 ``torch.nn`` 和 ``torch.optim``。
# 

import torch
import torch.nn as nn
import torch.optim as optim


######################################################################
# 2. 定义和初始化神经网络
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
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
# 我们将使用 SGD 优化器。
# 

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 4. 通过 ``state_dict`` 保存和加载模型
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 让我们只使用 ``state_dict`` 来保存和加载我们的模型。
# 

# 路径
PATH = "state_dict_model.pt"

# 保存
torch.save(net.state_dict(), PATH)

# 加载
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()


######################################################################
# 在PyTorch中，通常使用 ``.pt`` 或 ``.pth`` 文件扩展名来保存模型。
# 
# 注意 ``load_state_dict()`` 函数接受一个字典对象，而不是保存对象的路径。
# 这意味着你必须先反序列化保存的state_dict，然后再传递给 ``load_state_dict()`` 函数。
# 不能使用 ``model.load_state_dict(PATH)`` 来加载。
# 
# 还要记住，在运行推理之前，你必须调用 ``model.eval()`` 将dropout和batch normalization layers设置为评估模式。
# 否则将导致推理结果不一致。
# 
# 5. 保存和加载整个模型
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 现在让我们尝试将整个模型进行保存和加载。
# 

# Specify a path
PATH = "entire_model.pt"

# Save
torch.save(net, PATH)

# Load
model = torch.load(PATH)
model.eval()


######################################################################
# 在这里，同样要记住在运行推理之前调用 ``model.eval()`` 将 dropout 和 batch normalization layers 设置为评估模式。
# 
# 祝贺你！你已经成功地在PyTorch中保存和加载了用于推理的模型。
# 
# 继续学习
# ----------
# 
# 查看这些其他教程以继续学习：
# 
# - `PyTorch中保存和加载通用检查点 <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>`__
# - `PyTorch中将多个模型保存在一个文件中 <https://pytorch.org/tutorials/recipes/recipes/saving_multiple_models_in_one_file.html>`__