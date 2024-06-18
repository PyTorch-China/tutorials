"""
PyTorch 保存和加载通用检查点
==================================================
保存和加载通用检查点模型用于推理或恢复训练可以帮助你从上次离开的地方继续。
当保存通用检查点时，你必须保存不仅仅是模型的 state_dict。
同时也很重要保存优化器的 state_dict,因为它包含了在模型训练过程中更新的缓冲区和参数。
根据你自己的算法,你可能还需要保存你离开时的 epoch、最新记录的训练损失、外部的 torch.nn.Embedding 层等等。

简介
------------
要保存多个检查点,你必须将它们组织在一个字典中,并使用 ``torch.save()`` 来序列化这个字典。
一个常见的 PyTorch 约定是使用 ``.tar`` 文件扩展名来保存这些检查点。
要加载这些项目,首先初始化模型和优化器,然后使用 ``torch.load()`` 在本地加载字典。
从这里开始,你可以通过简单地查询字典来轻松访问保存的项目,就像你期望的那样。

在这个示例中,我们将探索如何保存和加载多个检查点。

环境设置
-----
在开始之前,如果还没有安装 ``torch``,我们需要安装它。

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
# 4. 保存通用检查点
# 5. 加载通用检查点
#
# 1. 导入加载数据所需的必要库
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 对于这个示例,我们将使用 ``torch`` 及其子模块 ``torch.nn`` 和 ``torch.optim``。
#

import torch
import torch.nn as nn
import torch.optim as optim


######################################################################
# 2. 定义和初始化神经网络
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 为了示例,我们将创建一个用于训练图像的神经网络。
# 要了解更多信息,请参阅定义神经网络的示例。
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
# 4. 保存通用检查点
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 收集所有相关信息并构建字典。
#

# 附加信息
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)


######################################################################
# 5. 加载通用检查点
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 首先初始化模型和优化器,然后在本地加载字典。
#

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - 或者 -
model.train()


######################################################################
# You must call ``model.eval()`` to set dropout and batch normalization
# layers to evaluation mode before running inference. Failing to do this
# will yield inconsistent inference results.
# 
# If you wish to resuming training, call ``model.train()`` to ensure these
# layers are in training mode.
# 
# Congratulations! You have successfully saved and loaded a general
# checkpoint for inference and/or resuming training in PyTorch.
#

# 你必须调用model.eval()来将dropout和批归一化层设置为评估模式,然后才能运行推理。
# 如果不这样做,将会得到不一致的推理结果。
#
# 如果你希望恢复训练,调用 ``model.train()`` 以确保这些层处于训练模式。
#
# 祝贺你!你已经成功保存和加载了一个通用检查点。
#