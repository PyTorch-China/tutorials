"""
PyTorch 在一个文件中保存和加载多个模型
============================================================
保存和加载多个模型，可以帮助您重用之前训练过的模型。

简介
------------
当保存由多个 ``torch.nn.Modules`` 组成的模型时，例如生成对抗网络(GAN)、序列到序列模型或模型集合时，
您必须保存每个模型的state_dict和相应的优化器。
您还可以通过简单地将其附加到字典中来保存任何可能有助于恢复训练的其他项目。
要加载模型，首先初始化模型和优化器，然后使用 ``torch.load()`` 在本地加载字典。
从这里开始，您可以像期望的那样简单地查询字典来轻松访问保存的项目。
在本教程中，我们将演示如何使用PyTorch在一个文件中保存多个模型。

环境设置
-----
在开始之前，如果尚未安装 ``torch``，我们需要先安装它。

.. code-block:: sh

   pip install torch

"""



######################################################################
# 具体步骤
# -----
# 1. 导入加载数据所需的所有必要库
# 2. 定义和初始化神经网络
# 3. 初始化优化器
# 4. 保存多个模型
# 5. 加载多个模型
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
# 构建两个变量用于最终保存模型。
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

netA = Net()
netB = Net()


######################################################################
# 3. 初始化优化器
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 我们将使用 SGD 为我们创建的每个模型构建优化器。
# 

optimizerA = optim.SGD(netA.parameters(), lr=0.001, momentum=0.9)
optimizerB = optim.SGD(netB.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 4. 保存多个模型
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 收集所有相关信息并构建字典。
# 

# 指定保存路径
PATH = "model.pt"

torch.save({
            'modelA_state_dict': netA.state_dict(),
            'modelB_state_dict': netB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            }, PATH)


######################################################################
# 4. 加载多个模型
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 记住首先初始化模型和优化器，然后在本地加载字典。
# 

modelA = Net()
modelB = Net()
optimModelA = optim.SGD(modelA.parameters(), lr=0.001, momentum=0.9)
optimModelB = optim.SGD(modelB.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()


######################################################################
# 在运行推理之前，您必须调用 ``model.eval()`` 将 dropout 和 batch normalization 层设置为评估模式。
# 否则将导致推理结果不一致。
# 
# 如果您希望恢复训练，请调用 ``model.train()`` 以确保这些层处于训练模式。
# 
# 祝贺您！您已经成功地在PyTorch中保存和加载了多个模型。
#