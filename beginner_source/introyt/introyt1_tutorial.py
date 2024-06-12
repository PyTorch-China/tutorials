"""
**简介** ||
`张量 <tensors_deeper_tutorial.html>`_ ||
`自动微分 <autogradyt_tutorial.html>`_ ||
`构建模型 <modelsyt_tutorial.html>`_ ||
`TensorBoard支持 <tensorboardyt_tutorial.html>`_ ||
`训练模型 <trainingyt.html>`_ ||
`模型理解 <captumyt.html>`_

PyTorch 简介
=======================

跟随下面的视频或在 `youtube <https://www.youtube.com/watch?v=IC0_FRiX-sw>`__ 上观看。

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/IC0_FRiX-sw" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

PyTorch 张量
---------------

从视频的 `03:50 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=230s>`__ 开始。

首先，我们将导入 pytorch。

"""

import torch

######################################################################
# 让我们看一些基本的张量操作。首先，创建张量的几种方式:
# 

z = torch.zeros(5, 3)
print(z)
print(z.dtype)


#########################################################################
# 上面，我们创建了一个 5x3 的零矩阵，并查询其数据类型，发现零是 32 位浮点数，这是 PyTorch 的默认设置。
# 
# 如果你想要整数呢?可以覆盖默认设置:
# 

i = torch.ones((5, 3), dtype=torch.int16)
print(i)


######################################################################
# 你可以看到，当我们改变默认设置时，在打印张量时会有所提示。
# 
# 通常情况下，会使用特定的种子初始化学习权重，以确保结果的可重复性:
# 

torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print('A random tensor:')
print(r1)

r2 = torch.rand(2, 2)
print('\nA different random tensor:')
print(r2) # 新的值

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print('\nShould match r1:')
print(r3) # 由于重新设置种子，所以与 r1 的值相同


#######################################################################
# PyTorch 张量执行算术运算很直观。形状相似的张量可以相加、相乘等。
# 与标量的运算会在整个张量上分布式进行:
#

ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # 每个元素都乘以 2
print(twos)

threes = ones + twos       # 形状相似，因此允许相加
print(threes)              # 张量按元素相加
print(threes.shape)        # 这与输入张量具有相同的维度

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# 取消注释这一行会导致运行时错误
# r3 = r1 + r2


######################################################################
# 这里是一些可用的数学运算示例:
# 

r = (torch.rand(2, 2) - 0.5) * 2 # 值在 -1 和 1 之间
print('A random matrix, r:')
print(r)

# 支持常见的数学运算:
print('\nAbsolute value of r:')
print(torch.abs(r))

# ...以及三角函数:
print('\nInverse sine of r:')
print(torch.asin(r))

# ...和线性代数运算，如行列式和奇异值分解
print('\nDeterminant of r:')
print(torch.det(r))
print('\nSingular value decomposition of r:')
print(torch.svd(r))

# ...以及统计和聚合运算:
print('\nAverage and standard deviation of r:')
print(torch.std_mean(r))
print('\nMaximum value of r:')
print(torch.max(r))


##########################################################################
# 关于 PyTorch 张量的强大功能还有很多需要了解，包括如何为 GPU 上的并行计算设置它们 - 我们将在另一个视频中深入探讨。
# 
# PyTorch 模型
# --------------
#
# 从视频的 `10:00 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=600s>`__ 开始。
#
# 让我们讨论一下如何在 PyTorch 中表示模型
#

import torch                     
import torch.nn as nn            # PyTorch 模型的父对象
import torch.nn.functional as F  # 用于激活函数


#########################################################################
# .. figure:: /_static/img/mnist.png
#    :alt: le-net-5 diagram
#
# *图: LeNet-5*
# 
# 上图是 LeNet-5 的示意图，它是最早的卷积神经网络之一，也是深度学习爆发式发展的驱动力之一。它被构建用于读取手写数字的小图像(MNIST 数据集)，并正确分类图像中表示的数字。
# 
# 它工作原理的简述为:
# 
# -  层 C1 是一个卷积层，它在输入图像中扫描它在训练期间学习到的特征。它输出一个特征激活图，
#    描述它在图像中看到每个学习到的特征的位置。这个"激活图"在层 S2 中被下采样。
# -  层 C3 是另一个卷积层，这次扫描 C1 的激活图以查找特征组合。它也输出一个激活图，
#    描述这些特征组合的空间位置，该激活图在层 S4 中被下采样。
# -  最后，最后的全连接层 F5、F6 和 OUTPUT 是一个分类器，它将最终的激活图分类为 10 个 bin 中的一个，
#    表示 10 个数字。
# 
# 我们如何在代码中表示这个简单的神经网络呢?
# 

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 个输入图像通道(黑白)，6 个输出通道，5x5 的正方形卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        # 一个仿射操作: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 在 (2, 2) 窗口上进行最大池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果尺寸是正方形，你只需指定一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除批次维度外的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


############################################################################
# 查看这段代码，你应该能够发现一些与上图结构相似的地方。
# 
# 这演示了典型 PyTorch 模型的结构: 
#
# -  它继承自 ``torch.nn.Module`` - 模块可以嵌套 - 事实上，即使 ``Conv2d`` 和 ``Linear`` 层类也继承自 ``torch.nn.Module``。
# -  一个模型将有一个 ``__init__()`` 函数，在这里它实例化其层，并加载任何它可能需要的数据组件(例如，一个 NLP 模型可能加载词汇表)。
# -  一个模型将有一个 ``forward()`` 函数。这是实际计算发生的地方:输入通过网络层和各种函数生成输出。
# -  除此之外，你可以像构建任何其他 Python 类一样构建你的模型类，添加任何你需要支持模型计算的属性和方法。
# 
# 让我们实例化这个对象并运行一个示例输入。
# 

net = LeNet()
print(net)                         # 对象打印了什么信息?

input = torch.rand(1, 1, 32, 32)   # 32x32 的黑白图像
print('\nImage batch shape:')
print(input.shape)

output = net(input)                # 不直接调用 forward()
print('\nRaw output:')
print(output)
print(output.shape)


##########################################################################
# 如上代码存在一些要点:
# 
# 首先，我们实例化 ``LeNet`` 类，并打印 ``net`` 对象。``torch.nn.Module`` 的子类将报告它创建的层及其形状和参数。
# 这可以提供一个模型的概览，如果你想了解它的处理过程。
# 
# 在下面，我们创建一个虚拟输入，表示一个 32x32 的单通道图像。通常情况下，你会加载一个图像切片并将其转换为这种形状的张量。
# 
# 你可能已经注意到我们的张量有一个额外的维度 - *批次维度*。PyTorch 模型假设它们正在处理数据*批次* 
# - 例如，包含 16 个图像切片的批次将具有形状 ``(16, 1, 32, 32)``。
# 由于我们只使用一个图像，我们创建了一个形状为 ``(1, 1, 32, 32)`` 的批次。
# 
# 我们通过像函数一样调用它来要求模型进行推理: ``net(input)``。这个调用的输出表示模型对输入表示特定数字的置信度。
# (由于这个模型实例还没有学习任何东西，我们不应该期望在输出中看到任何信号。)查看 ``output`` 的形状，
# 我们可以看到它也有一个批次维度，其大小应该始终与输入批次维度相匹配。如果我们传入了一个包含 16 个实例的输入批次，
# ``output`` 将具有 ``(16, 10)`` 的形状。
# 
# 数据集和数据加载器
# ------------------------
#
# 从视频的 `14:00 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=840s>`__ 开始。
#
# 下面，我们将演示如何使用 TorchVision 中的一个可下载的开放访问数据集，
# 如何转换图像以供你的模型使用，以及如何使用 DataLoader 将数据批次提供给你的模型。
#
# 我们需要做的第一件事是将传入的图像转换为 PyTorch 张量。
#

#%matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


##########################################################################
# 在这里，我们为输入指定了两种转换:
#
# -  ``transforms.ToTensor()`` 将 Pillow 加载的图像转换为 PyTorch 张量。
# -  ``transforms.Normalize()`` 调整张量的值，使其平均值为零，标准差为 1.0。
#    大多数激活函数在 x = 0 附近具有最强梯度，因此将我们的数据居中可以加快学习速度。
#    传递给转换的值是数据集中图像的 rgb 值的均值(第一个元组)和标准差(第二个元组)。
#    你可以通过运行以下几行代码自己计算这些值:
#          ```
#           from torch.utils.data import ConcatDataset
#           transform = transforms.Compose([transforms.ToTensor()])
#           trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
#
#           #将所有训练图像堆叠成形状为 (50000, 3, 32, 32) 的张量
#           x = torch.stack([sample[0] for sample in ConcatDataset([trainset])])
#
#           #获取每个通道的均值
#           mean = torch.mean(x, dim=(0,2,3)) #tensor([0.4914, 0.4822, 0.4465])
#           std = torch.std(x, dim=(0,2,3)) #tensor([0.2470, 0.2435, 0.2616])
#
#          ```
#
# 还有许多其他可用的转换，包括裁剪、居中、旋转和反射。
#
# 接下来，我们将创建 CIFAR10 数据集的一个实例。这是一组 32x32 的彩色图像切片，代表 10 类物体: 
# 6 种动物(鸟、猫、鹿、狗、青蛙、马)和 4 种车辆(飞机、汽车、船、卡车):
#

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


##########################################################################
# .. note::
#      当你运行上面的单元格时，它可能需要一些时间来下载数据集。
#
# 这是在 PyTorch 中创建数据集对象的一个示例。可下载的数据集(如上面的 CIFAR-10)是 
# ``torch.utils.data.Dataset`` 的子类。PyTorch 中的 ``Dataset`` 类包括 
# TorchVision、Torchtext 和 TorchAudio 中的可下载数据集，以及诸如 
# ``torchvision.datasets.ImageFolder`` 之类的实用程序数据集类，它将读取一个标记过的图像文件夹。
# 你也可以创建 ``Dataset`` 的自己的子类。
#
# 当我们实例化我们的数据集时，我们需要告诉它一些事情:
#
# -  我们希望数据存放的文件系统路径。
# -  我们是否使用这个集合进行训练;大多数数据集将被分为训练和测试子集。
# -  如果我们还没有下载数据集，我们是否希望下载它。
# -  我们想对数据应用哪些转换。
#
# 一旦你的数据集准备就绪，你就可以将它交给 ``DataLoader``:

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


##########################################################################
# ``Dataset`` 的子类包装了对数据的访问，并专门针对它正在服务的数据类型。
# ``DataLoader`` 对它正在服务的数据一无所知，但会根据你指定的参数将 ``Dataset`` 提供的输入张量组织成批次。
#
# 在上面的示例中，我们要求一个 ``DataLoader`` 从 ``trainset`` 中给我们批次大小为 4 的批次，
# 随机打乱它们的顺序(``shuffle=True``)，并告诉它启动两个工作进程从磁盘加载数据。
#
# 可视化你的 ``DataLoader`` 提供的批次是一个很好的做法:
#

import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 获取一些随机训练图像
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 显示图像
imshow(torchvision.utils.make_grid(images))
# 打印标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 运行上面的单元格应该会显示你一条四张图像的条带,以及每张图像的正确标签。
#
# 训练你的 PyTorch 模型
# ---------------------------
#
# 从视频的 `17:10 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=1030s>`__ 开始。
#
# 让我们把所有的部分放在一起,训练一个模型:
#

#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


#########################################################################
# 首先,我们需要训练和测试数据集。如果你还没有,运行下面的单元格来确保数据集已下载。(可能需要一分钟)
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
# 运行对 ``DataLoader`` 输出的检查:
#

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 显示图像
imshow(torchvision.utils.make_grid(images))
# 打印标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


##########################################################################
# 这是我们将要训练的模型。如果它看起来很熟悉,那是因为它是 LeNet 的一个变体 
# - 在本视频前面讨论过 - 适用于 3 色图像。


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


######################################################################
# 我们最后需要的是一个损失函数和一个优化器:
# 

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


##########################################################################
# 损失函数,如本视频前面所讨论的,是衡量模型预测与理想输出之间差距的指标。
# 交叉熵损失是像我们这样的分类模型的典型损失函数。
#
# **优化器**是驱动学习的关键。在这里,我们创建了一个实现 *随机梯度下降* 的优化器，
# 这是最直接的优化算法之一。除了算法的参数(如学习率 ``lr`` 和动量)之外，
# 我们还传入了 ``net.parameters()``，它是模型中所有学习权重的集合 - 这是优化器要调整的对象。
#
# 最后,所有这些都被组装到训练循环中。继续运行这个单元格,它可能需要几分钟才能执行:
#

for epoch in range(2):  # 在数据集上循环多次

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data

        # 将参数梯度归零
        optimizer.zero_grad()

        # 前向 + 反向 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每 2000 个小批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


########################################################################
# 在这里，我们只进行了 **2 个训练轮次** (第 1 行) - 也就是在训练数据集上进行了两次完整遍历。
# 每次遍历都有一个内部循环，**遍历训练数据** (第 4 行)，提供经过转换的输入图像批次及其正确标签。
#
# **将梯度归零** (第 9 行)是一个重要步骤。梯度会在一个批次上累积；如果我们不为每个批次重置它们，
# 它们将继续累积，从而提供错误的梯度值，使学习变得不可能。
#
# 在第 12 行,我们**要求模型对这个批次进行预测**。在下一行(13)中，我们计算损失 
# - ``outputs``(模型预测)与 ``labels``(正确输出)之间的差异。
#
# 在第 14 行，我们进行 ``backward()`` 传播，计算将指导学习的梯度。
#
# 在第 15 行，优化器执行一步学习 - 它使用 ``backward()`` 调用得到的梯度来调整学习权重，以减小损失。
#
# 循环的其余部分对轮次号、已完成的训练实例数以及训练循环中收集的损失进行了一些轻量级报告。
#
# **当你运行上面的单元格时**,你应该会看到类似这样的输出:
#
# .. code-block:: sh
#
#    [1,  2000] loss: 2.235
#    [1,  4000] loss: 1.940
#    [1,  6000] loss: 1.713
#    [1,  8000] loss: 1.573
#    [1, 10000] loss: 1.507
#    [1, 12000] loss: 1.442
#    [2,  2000] loss: 1.378
#    [2,  4000] loss: 1.364
#    [2,  6000] loss: 1.349
#    [2,  8000] loss: 1.319
#    [2, 10000] loss: 1.284
#    [2, 12000] loss: 1.267
#    Finished Training
#
# 注意损失值是单调下降的，表明我们的模型在继续提高其在训练数据集上的性能。
#
# 作为最后一步，我们应该检查模型是否真正做到了 *泛化* 学习，而不是简单地"记住"了数据集。这被称为 **过拟合**，
# 通常表明数据集太小(没有足够的样本进行泛化学习)，或者模型的学习参数比正确建模数据集所需的更多。
#
# 这就是为什么数据集被分为训练和测试子集的原因 - 为了测试模型的泛化能力,我们要求它对从未训练过的数据进行预测:
#

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


#########################################################################
# 如果你一直跟随下来,你应该会看到模型在这一点上的准确率大约为 50%。这并不是最先进的水平，
# 但比随机输出的 10% 准确率要好得多。这证明了模型确实发生了一些泛化学习。
