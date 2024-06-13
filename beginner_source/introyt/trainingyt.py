"""
`简介 <introyt1_tutorial.html>`_ ||
`张量 <tensors_deeper_tutorial.html>`_ ||
`自动微分 <autogradyt_tutorial.html>`_ ||
`构建模型 <modelsyt_tutorial.html>`_ ||
`TensorBoard支持 <tensorboardyt_tutorial.html>`_ ||
**训练模型** ||
`模型理解 <captumyt.html>`_

使用 PyTorch 训练模型
=====================

跟随下面的视频或在 `youtube <https://www.youtube.com/watch?v=jF43_wj_DCQ>`__ 上观看。

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/jF43_wj_DCQ" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

简介
------------

在过去的视频中,我们讨论并演示了:

- 使用 torch.nn 模块中的神经网络层和函数构建模型
- 自动梯度计算的机制,这是基于梯度的模型训练的核心
- 使用 TensorBoard 可视化训练进度和其他活动

在本视频中,我们将为您的库存添加一些新工具:

- 我们将熟悉数据集和数据加载器抽象,以及它们如何简化向模型训练循环提供数据的过程
- 我们将讨论特定的损失函数以及何时使用它们
- 我们将了解 PyTorch 优化器,它们实现了根据损失函数的结果调整模型权重的算法

最后,我们将把所有这些结合起来,看一个完整的 PyTorch 训练循环的实际运行。


数据集和数据加载器
----------------------
 
``Dataset`` 和 ``DataLoader`` 类封装了从存储中提取数据并以批次形式暴露给训练循环的过程。

``Dataset`` 负责访问和处理单个数据实例。
 
``DataLoader`` 从 ``Dataset`` 中提取数据实例(无论是自动提取还是使用您定义的采样器),将它们收集到批次中,并返回给您的训练循环进行消费。``DataLoader`` 可以与所有类型的数据集一起使用,无论它们包含什么类型的数据。
 
对于本教程,我们将使用 TorchVision 提供的 Fashion-MNIST 数据集。我们使用 ``torchvision.transforms.Normalize()`` 来零中心和标准化图像瓦片内容的分布,并下载训练和验证数据分割。

""" 

import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard 支持
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# 创建训练和验证数据集,如果需要则下载
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# 为我们的数据集创建数据加载器;训练时打乱,验证时不打乱
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# 类别标签
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# 报告分割大小
print('训练集有 {} 个实例'.format(len(training_set)))
print('验证集有 {} 个实例'.format(len(validation_set)))


######################################################################
# 像往常一样,让我们可视化数据作为健全性检查:
# 

import matplotlib.pyplot as plt
import numpy as np

# 内联图像显示的辅助函数
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # 反标准化
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(training_loader)
images, labels = next(dataiter)

# 从图像创建网格并显示它们
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print('  '.join(classes[labels[j]] for j in range(4)))


#########################################################################
# 模型
# ---------
# 
# 我们在本例中使用的模型是 LeNet-5 的变体 - 如果您观看了本系列的前几个视频,应该会很熟悉。
# 

import torch.nn as nn
import torch.nn.functional as F

# PyTorch 模型继承自 torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

model = GarmentClassifier()


##########################################################################
# 损失函数
# -------------
# 
# 对于本例,我们将使用交叉熵损失。为了演示目的,我们将创建虚拟输出和标签值的批次,将它们通过损失函数,并检查结果。
# 

loss_fn = torch.nn.CrossEntropyLoss()

# 注意:损失函数期望数据以批次形式,所以我们创建了 4 个批次
# 表示模型对给定输入的 10 个类别中每一个的置信度
dummy_outputs = torch.rand(4, 10)
# 表示正确的类别在测试的 10 个类别中
dummy_labels = torch.tensor([1, 5, 3, 7])
    
print(dummy_outputs)
print(dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print('此批次的总损失: {}'.format(loss.item()))


#################################################################################
# 优化器
# ---------
# 
# 对于本例,我们将使用带动量的简单随机梯度下降。
# 
# 尝试一些优化方案的变体会很有启发性:
# 
# - 学习率决定了优化器采取的步长大小。不同的学习率对您的训练结果有何影响,在准确性和收敛时间方面?
# - 动量在多个步骤中将优化器推向最强梯度的方向。改变这个值会对结果产生什么影响?
# - 尝试一些不同的优化算法,如平均 SGD、Adagrad 或 Adam。您的结果有何不同?
# 

# 在 torch.optim 包中指定优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


#######################################################################################
# 训练循环
# -----------------
# 
# 下面,我们有一个执行一个训练周期的函数。它
# 从 DataLoader 枚举数据,并在循环的每一次通过时执行以下操作:
# 
# - 从 DataLoader 获取一批训练数据
# - 将优化器的梯度归零
# - 执行推理 - 也就是从模型获取输入批次的预测
# - 计算该组预测与数据集上的标签之间的损失
# - 计算学习权重的反向梯度
# - 告诉优化器执行一个学习步骤 - 也就是根据我们选择的优化算法,基于该批次观察到的梯度来调整模型的学习权重
# - 它每 1000 个批次报告一次损失。
# - 最后,它报告最后 1000 个批次的平均每批次损失,以便与验证运行进行比较
# 

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    
    # 这里,我们使用 enumerate(training_loader) 而不是
    # iter(training_loader),以便我们可以跟踪批次索引并进行一些周期内报告
    for i, data in enumerate(training_loader):
        # 每个数据实例都是一个输入 + 标签对
        inputs, labels = data
        
        # 对于每个批次,将梯度归零!
        optimizer.zero_grad()
        
        # 对该批次进行预测
        outputs = model(inputs)
        
        # 计算损失及其梯度
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # 调整学习权重
        optimizer.step()
        
        # 收集数据并报告
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # 每批次损失
            print('  批次 {} 损失: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            
    return last_loss


##################################################################################
# 每周期活动
# ~~~~~~~~~~~~~~~~~~
# 
# 我们每个周期需要做的事情有:
#
# - 通过检查未用于训练的一组数据上的相对损失来执行验证,并报告这一点
# - 保存模型的副本
# 
# 在这里,我们将在 TensorBoard 中进行报告。这需要转到命令行启动 TensorBoard,并在另一个浏览器选项卡中打开它。
# 

# 在单独的单元格中初始化,以便我们可以轻松地将更多周期添加到同一运行中
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('周期 {}:'.format(epoch_number + 1))
    
    # 确保梯度跟踪已打开,并对数据进行一次传递
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    

    running_vloss = 0.0
    # 将模型设置为评估模式,禁用 dropout 并使用批量规范化的群体统计数据。
    model.eval()

    # 禁用梯度计算并减少内存消耗。
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    
    avg_vloss = running_vloss / (i + 1)
    print('损失 训练 {} 有效 {}'.format(avg_loss, avg_vloss))
    
    # 记录每批次平均的运行损失
    # 对于训练和验证
    writer.add_scalars('训练与验证损失',
                    { '训练' : avg_loss, '验证' : avg_vloss },
                    epoch_number + 1)
    writer.flush()
    
    # 跟踪最佳性能,并保存模型的状态
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
    
    epoch_number += 1


#########################################################################
# 要加载保存的模型版本:
#
# .. code:: python
#
#     saved_model = GarmentClassifier()
#     saved_model.load_state_dict(torch.load(PATH))
#
# 一旦加载了模型,它就可以用于您需要的任何事情 -
# 更多训练、推理或分析。
# 
# 请注意,如果您的模型有影响模型结构的构造函数参数,您需要提供它们并以与保存时相同的方式配置模型。
# 
# 其他资源
# ---------------
# 
# -  pytorch.org 上的数据工具文档,包括 Dataset 和 DataLoader
# -  关于使用固定内存进行 GPU 训练的说明
# -  TorchVision、TorchText 和 TorchAudio 中可用数据集的文档
# -  PyTorch 中可用损失
