"""
`基础知识 <intro.html>`_ ||
**快速入门** ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`构建神经网络 <buildmodel_tutorial.html>`_ ||
`自动微分 <autogradqs_tutorial.html>`_ ||
`优化模型参数 <optimization_tutorial.html>`_ ||
`保存和加载模型 <saveloadrun_tutorial.html>`_

快速入门
==========

本节将介绍机器学习任务中常用的API。想更深入了解各模块内容，可参考每节文末处的链接。

处理数据
-----------------

PyTorch 提供了两个用于 `处理数据的原语<https://pytorch.org/docs/stable/data.html>`:
`torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset`。`Dataset`存储样本及其对应的标签，而`DataLoader`则在`Dataset`外部封装一层，变为可迭代对象。

"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

######################################################################
# PyTorch 提供了特定领域的库，例如
# [TorchText](https://pytorch.org/text/stable/index.html)，
# [TorchVision](https://pytorch.org/vision/stable/index.html)，和
# [TorchAudio](https://pytorch.org/audio/stable/index.html)，
# 所有这些库都包含了对应数据集。在本教程中，我们将使用 TorchVision 数据集。

# `torchvision.datasets` 模块包含许多现实世界视觉数据`Dataset`，例如 CIFAR、COCO
# ([数据集列表](https://pytorch.org/vision/stable/datasets.html))。
# 在本教程中，我们使用 `FashionMNIST`数据集。每个TorchVision `Dataset`包括两个参数：`transform` 和 `target_transform`，分别用于修改样本数据和标签。

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

######################################################################
# 我们将 `Dataset` 作为参数传递给 `DataLoader`，在数据集上封装了一个可迭代对象，
# 支持自动批处理、采样、打乱和多进程数据加载。这里我们定义一个批处理大小为 64，
# 即 dataloader 每批将返回大小为 64 的特征数据和标签。

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

######################################################################
# 获取更多关于 `PyTorch数据加载<data_tutorial.html>`的信息。

######################################################################
# --------------
#

################################
# 创建模型
# ------------------
# 要在 PyTorch 中定义一个神经网络，我们需要创建一个继承自 `nn.Module<https://pytorch.org/docs/stable/generated/torch.nn.Module.html> 的类。
# 我们在 `__init__` 函数中定义网络的层，并在 `forward` 函数中指定数据如何经过网络。为了加速神经网络中的运算，
# 我们将其移到 GPU 或 MPS（如果可用）上。

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

######################################################################
# 获取更多关于 `PyTorch构建神经网络<buildmodel_tutorial.html>`的内容。


######################################################################
# --------------
#


#####################################################################
# 优化模型参数
# ----------------------------------------
# 练一个模型，我们需要一个`损失函数`<https://pytorch.org/docs/stable/nn.html#loss-functions>
# 和一个`优化器`https://pytorch.org/docs/stable/optim.html>。


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


#######################################################################
# 在单个训练循环中，模型对训练数据集（分批输入）进行预测，并通过反向传播预测误差来调整模型的参数。

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

##############################################################################
# 我们还需检查模型在测试数据集上的效果，以确保它在持续学习。


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

##############################################################################
# 通过多次迭代（*epochs*）进行训练。在每个迭代过程中，模型通过对参数的学习以提高预测准确性。
# 我们在每个 epoch 打印模型的准确率和损失；我们希望看到随着每个 epoch 训练，模型预测准确率不断提高，
# 损失逐渐减少。


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

######################################################################
# Read more about `Training your model <optimization_tutorial.html>`_.
# 获取更多关于 `训练模型 <optimization_tutorial.html>`的内容。

######################################################################
# --------------
#

######################################################################
# 保存模型
# -------------
# 保存模型的常见方法是将内部状态字典（包含模型参数）序列化。

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


######################################################################
# 加载模型
# ----------------------------
# 加载模型的过程包括重新创建模型结构并加载其内部状态字典。

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

#############################################################
# 这个模型现在可以用来进行预测了。

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


######################################################################
# 获取更多有关 `保存和加载模型 <saveloadrun_tutorial.html>`的内容。
