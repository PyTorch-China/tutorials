"""
 PyTorch 中清零梯度
================================
在构建神经网络时，清零梯度是有益的。
因为默认情况下,每次调用 ``.backward()`` 时,梯度会累积在缓冲区中(即不会被覆盖)。

介绍
------------
在训练神经网络时,模型能够通过使用梯度下降来提高它们的精度。简而言之,梯度下降是通过调整模型中的权重和偏置来最小化损失(或误差)的过程。

``torch.Tensor`` 是PyTorch的中心类。当你创建一个张量时,如果将其属性 ``.requires_grad`` 设置为 ``True``,
该对象会跟踪对它的所有操作。这发生在后续的反向传播过程中。该张量的梯度将累积到 ``.grad`` 属性中。
所有梯度的累积(或求和)是在对损失张量调用 .backward() 时计算的。

在某些情况下,可能需要清零张量的梯度。例如:当你开始训练循环时,你应该清零梯度,以便正确执行此跟踪。
在本教程中,我们将学习如何使用PyTorch库清零梯度。我们将通过在PyTorch内置的 ``CIFAR10`` 数据集上训练神经网络来演示如何做到这一点。

环境设置
-----
由于我们将在本教程中训练数据,如果你在可运行的笔记本中,最好将运行时切换到GPU或TPU。
在开始之前,如果尚未安装 ``torch`` 和 ``torchvision``,我们需要安装它们。

.. code-block:: sh

   pip install torchvision


"""


######################################################################
# 具体步骤
# -----
#
# 步骤1到4设置了我们用于训练的数据和神经网络。清零梯度的过程发生在步骤5。如果你已经构建了数据和神经网络,可以跳过前四步,直接进入第5步。
#
# 1. 导入加载数据所需的所有必要库
# 2. 加载和标准化数据集
# 3. 构建神经网络
# 4. 定义损失函数
# 5. 在训练网络时清零梯度
#
# 1. 导入加载数据所需的必要库
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 对于本教程,我们只使用 ``torch`` 和 ``torchvision`` 来访问数据集。
#

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


######################################################################
# 2. 加载和标准化数据集
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# PyTorch提供了各种内置数据集(有关更多信息,请参阅加载数据教程)。
#

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


######################################################################
# 3. 构建神经网络
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 我们将使用卷积神经网络。要了解更多信息,请参阅定义神经网络教程。
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


######################################################################
# 4. 定义损失函数和优化器
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 让我们使用分类交叉熵损失和带动量的SGD。
#

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


######################################################################
# 5. 在训练网络时清零梯度
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 我们只需要遍历数据迭代器,并将输入馈送到网络中并优化。
#
# 注意,对于每个数据实体,我们都会清零梯度。这是为了确保在训练神经网络时,我们不会跟踪任何不必要的信息。
#

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入,data是一个包含[输入,标签]的列表
        inputs, labels = data

        # 清零参数梯度
        optimizer.zero_grad()

        # 前向 + 反向 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个小批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


######################################################################
# 你也可以使用 ``model.zero_grad()``。只要你的所有模型参数都在该优化器中,
# 使用 ``model.zero_grad()`` 和使用 ``optimizer.zero_grad()`` 是一样的。请根据具体情况决定使用哪一种方式。
#
# 祝贺你!你已经成功地在PyTorch中清零了梯度。
#
# 继续学习
# ----------
#
# 查看这些其他教程,继续你的学习之旅:
#
# - `在PyTorch中加载数据 <https://pytorch.org/tutorials/recipes/recipes/loading_data_recipe.html>`__
# - `在PyTorch中跨设备保存和加载模型 <https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html>`__