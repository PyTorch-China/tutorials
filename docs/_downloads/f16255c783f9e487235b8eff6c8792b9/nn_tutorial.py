# -*- coding: utf-8 -*-
"""
`torch.nn` 具体是什么?
============================

**Authors:** Jeremy Howard, `fast.ai <https://www.fast.ai>`_. Thanks to Rachel Thomas and Francisco Ingham.
"""

###############################################################################
# 我们建议将本教程作为笔记本（notebook）运行。请点击页面顶部的链接，下载笔记本（``.ipynb``）文件。
#
# PyTorch 提供了优雅设计的模块和类 
# `torch.nn <https://pytorch.org/docs/stable/nn.html>`_ ，
# `torch.optim <https://pytorch.org/docs/stable/optim.html>`_ ，
# `Dataset <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_ ，
# 以及 `DataLoader <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`_ ，
# 以帮助你创建和训练神经网络。
# 为了充分利用它们的功能，并通过自定义对应模块或类，来解决特定问题，需要理解它们的具体功能。为此，我们将首先在 MNIST 数据集上训练一个基本的神经网络，而不使用这些模型的任何特性；
# 我们最初只使用最基本的 PyTorch 张量功能。然后，我们将逐步添加``torch.nn``、``torch.optim``、``Dataset`` 
# 或 ``DataLoader``中的一个特性，展示每个部分的作用，以及如何使用它们让代码更简洁或更灵活。
#
# **本教程假定你已经安装了 PyTorch，并且熟悉张量操作的基础知识。**（如果你熟悉 Numpy 数组操作，你会发现这里使用的 PyTorch 张量操作几乎相同）。
#
# MNIST 数据集设置
# ----------------
#
# 我们将使用经典的 `MNIST <http://deeplearning.net/data/mnist/>`_ 数据集，
# 该数据集包含手绘数字（0到9之间）的黑白图像。
#
# 我们将使用 `pathlib <https://docs.python.org/3/library/pathlib.html>`_
# 来处理路径（Python 3 标准库的一部分），并使用 `requests <http://docs.python-requests.org/en/master/>`_ 下载数据集。
# 我们只会在使用模块时才导入它们，因此你可以清楚地看到每个步骤中正在使用的内容。

from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

###############################################################################
# 这个数据集是 numpy 数组格式的，并且使用 pickle 存储，
# 这是一个 Python 特有的用于序列化数据的格式。

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

###############################################################################
# 每张图像的尺寸为 28 x 28，并以长度为 784（=28x28）的展平行存储。让我们来看看其中一张；我们需要先将其重塑为二维。

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# ``pyplot.show()`` 在不使用 Colab 时使用
try:
    import google.colab
except ImportError:
    pyplot.show()
print(x_train.shape)

###############################################################################
# PyTorch 使用 ``torch.tensor`` 而不是 numpy 数组，所以我们需要转换我们的数据。

import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

###############################################################################
# 从零开始的神经网络（不使用 ``torch.nn``）
# --------------------------------------------
#
# 首先，我们只使用 PyTorch 张量操作创建一个模型。我们假设你已经熟悉神经网络的基础知识。（如果不熟悉，可以在 `course.fast.ai <https://course.fast.ai>`_ 学习。）
#
# PyTorch 提供方法来创建 随机 或 零 填充的张量，我们将使用这些方法为一个简单的线性模型创建权重和偏置。
# 这些只是常规的张量，有一个非常特别的附加功能：我们告诉 PyTorch 它们需要梯度。PyTorch 会记录在张量上完成的所有操作，以便在反向传播期间 *自动* 计算梯度！
#
# 对于权重，我们在初始化 **之后** 设置 ``requires_grad``，因为我们不希望初始化步骤包括在梯度中。（注意，PyTorch 中的尾随 ``_`` 表示操作是在原地执行。）
#
# .. 注意:: 我们在这里使用
#    `Xavier 初始化 <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
#    （通过乘以 ``1/sqrt(n)``）初始化权重。

import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

###############################################################################
# 由于 PyTorch 能够自动计算梯度，我们可以使用任何标准的 Python 函数（或可调用对象）作为模型！
# 让我们编写一个简单的矩阵乘法和广播加法，来创建一个简单的线性模型。我们还需要编写一个激活函数 `log_softmax`。
# PyTorch 提供了许多预先编写的损失函数、激活函数等，你仍可以使用普通的 Python 编写自己的函数。
# PyTorch 会为你的函数自动创建 GPU 或矢量化 CPU 代码。

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

######################################################################################
# 在上面的代码中，``@`` 表示矩阵乘法操作。在一个数据批次上调用我们的函数（在本例中为64张图像）。
# 这就是一次 *前向传递*。请注意，由于我们在开始时设置权重为随机数值，此时预测结果准确性较低。

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)

###############################################################################
# 如你所见，``preds`` 张量不仅包含张量值，还包含梯度函数。在稍后的反向传播过程中会用到它。
# 
# 让我们实现 negative log-likelihood 作为损失函数（同样，我们可以只使用标准的 Python）：


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

###############################################################################
# 让我们使用我们的随机模型来检查损失(loss)，这样我们就可以看到在之后进行反向传播后，预测结果准确率是否有所提升。

yb = y_train[0:bs]
print(loss_func(preds, yb))


###############################################################################
# 我们还要实现一个函数来计算我们模型的准确率。
# 对于每个预测结果，如果具有最大值的索引与目标值匹配，则预测是正确的。

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

###############################################################################
# 检查我们随机模型的准确率，这样我们就可以看到随着损失的改善，准确率是否有所提高。

print(accuracy(preds, yb))

###############################################################################
# 现在可以运行一个训练循环。对于每次迭代：
#
# - 选择一个大小为 ``bs`` 的批量数据
# - 使用模型进行预测
# - 计算损失
# - ``loss.backward()`` 更新模型的梯度，即更新 ``weights`` 和 ``bias``。
#
# 我们现在使用这些梯度来更新权重(weights)和偏置(bias)。我们在 ``torch.no_grad()`` 上下文管理器中执行此操作，
# 因为我们不希望这些操作记录为下一次梯度计算的一部分。你可以在 `这里 <https://pytorch.org/docs/stable/notes/autograd.html>`_ 
# 阅读有关 PyTorch 的 Autograd 如何记录操作的更多信息。
#
# 然后，我们将梯度设置为零，以便我们准备进行下一次循环。否则，我们的梯度将记录所有已发生的操作（即 ``loss.backward()`` *添加* 梯度到已有的梯度中，而不是替换它们）。
#
# .. 提示:: 您可以使用标准的 Python 调试器逐步执行 PyTorch 代码，从而可以检查每个步骤中的各种变量值(去除 ``set_trace()`` 的注释)。

from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

###############################################################################
# 我们已经从零开始创建并训练了一个最小的神经网络（使用逻辑回归，没有隐藏层）。
# 让我们检查一下损失和准确率，并将它们与之前得到的结果进行比较，预计损失会减少，准确率会提高。

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

###############################################################################
# 使用 ``torch.nn.functional``
# ------------------------------
#
# 现在我们将重构代码，使其与之前做的事情相同，只是我们将开始利用 PyTorch 的 ``nn`` 类，使其更简洁和灵活。
# 从这里开始的每一步，都让我们的代码变得更短、更易理解和更灵活。
#
# 第一步也是最简单的一步是通过用 ``torch.nn.functional``（通常按惯例导入为命名空间 ``F``）
# 中的激活和损失函数替换我们手写的激活和损失函数，从而使我们的代码更简短。该模块包含 ``torch.nn`` 库中的所有函数。
# 除了各种损失和激活函数，你还会看到一些创建神经网络的便捷函数，比如池化函数。
#（还有用于卷积、线性层等的函数，但正如我们将看到的，这些通常更适合使用库的其他模块来处理。）
#
# 如果你使用negative log likelihood loss 和 log softmax activation，那么 PyTorch 提供了一个结合了两者的单一函数 
# ``F.cross_entropy``。所以我们可以从模型中移除激活函数。

import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

###############################################################################
# 我们不再在 ``model`` 函数中调用 ``log_softmax``。查看下损失和准确率是否与之前结果一致：

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

###############################################################################
# 使用 ``nn.Module`` 重构
# -----------------------------
#
# 接下来，我们将使用 ``nn.Module`` 和 ``nn.Parameter``，以实现更清晰和简洁的训练循环。
# 我们将继承 ``nn.Module``（它本身是一个类，能够跟踪状态）。在这种情况下，我们想创建一个类来保存我们的权重、偏置和forward方法。
# 我们将会使用 ``nn.Module`` 的属性和方法（例如 ``.parameters()`` 和 ``.zero_grad()``）。

# .. 注意:: ``nn.Module``（大写 M）是 PyTorch 特有的概念，是使用PyTorch过程中大量使用的类。
# ``nn.Module`` 不要与 Python 概念的（小写 ``m``）`module <https://docs.python.org/3/tutorial/modules.html>`_ 混淆。

from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

###############################################################################
# 由于我们现在使用的是对象而不是仅仅使用函数，我们首先要创建模型对象：

model = Mnist_Logistic()

###############################################################################
# 现在我们可以像之前一样计算损失。请注意，``nn.Module`` 对象可以像函数一样使用（即它们是*可调用的*），
# PyTorch 会自动调用我们的 ``forward`` 方法。

print(loss_func(model(xb), yb))

###############################################################################
# 在之前的训练循环中，我们必须按名称更新每个参数的值，并手动将每个参数的梯度分别清零，如下所示：
#
# .. code-block:: python
#
#    with torch.no_grad():
#        weights -= weights.grad * lr
#        bias -= bias.grad * lr
#        weights.grad.zero_()
#        bias.grad.zero_()
#
#
# 现在我们可以利用 model.parameters() 和 model.zero_grad()（PyTorch 在 ``nn.Module`` 定义的方法）
# 来使这些步骤更简洁，防止忘记处理某些参数导致错误，尤其是当我们实现一个更复杂的模型时：
#
# .. code-block:: python
#
#    with torch.no_grad():
#        for p in model.parameters(): p -= p.grad * lr
#        model.zero_grad()
#
#
# 将训练循环包装在一个 ``fit`` 函数中，这样可以多次运行它。

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()

###############################################################################
# Let's double-check that our loss has gone down:
# 让我们查看下训练后，损失是否下降了：

print(loss_func(model(xb), yb))

###############################################################################
# 使用 ``nn.Linear`` 重构
# ----------------------------
#
# 我们继续重构代码。使用 PyTorch 类 `nn.Linear <https://pytorch.org/docs/stable/nn.html#linear-layers>`_ 来实现线性层，
# 不再手动定义和初始化 ``self.weights`` 和 ``self.bias``，以及计算 ``xb @ self.weights + self.bias``。
# PyTorch 具有多种预定义的层，可以大大简化我们的代码，并且提高执行速度。

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

###############################################################################
# 初始化模型对象，并计算损失数值

model = Mnist_Logistic()
print(loss_func(model(xb), yb))

###############################################################################
# 调用 ``fit`` 方法进行训练模型

fit()

# 查看训练结果
print(loss_func(model(xb), yb))

###############################################################################
# 使用 ``torch.optim`` 重构
# ------------------------------
#
# PyTorch ``torch.optim``包含多种优化算法 。我们可以使用优化器的 ``step`` 方法进行优化步骤，无需手动更新每个参数。
# 
# 之前的优化步骤：
#
# .. code-block:: python
#
#    with torch.no_grad():
#        for p in model.parameters(): p -= p.grad * lr
#        model.zero_grad()
#
# 重构为:
#
# .. code-block:: python
#
#    opt.step()
#    opt.zero_grad()
#
# (在下个训练循环开始前，我们需调用 ``optim.zero_grad()`` 方法，将参数的梯度重置为0。)

from torch import optim

###############################################################################
# 定义创建模型和优化器的方法如下:

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

###############################################################################
# 使用 Dataset 重构
# ------------------------------
#
# PyTorch 有一个抽象的 Dataset 类。Dataset 可以是任何具有 ``__len__`` 函数（由 Python 的标准 ``len`` 函数调用）
# 和 ``__getitem__`` 函数（作为索引方式）的对象。
# `教程 <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`_ 详细介绍了创建一个自定义 
# ``FacialLandmarkDataset`` 类作为 ``Dataset`` 子类的例子。
#
# PyTorch 的 `TensorDataset <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#TensorDataset>`_ 
# 是一个包装张量的 Dataset，为我们提供了一种迭代、索引和沿张量的第一个维度切片的方式，使我们在训练时更容易同时访问自变量和因变量。

from torch.utils.data import TensorDataset

###############################################################################
# 使用``TensorDataset`` 对 ``x_train`` 和 ``y_train`` 进行包装， 让我们更容易对数据进行遍历和切片操作。

train_ds = TensorDataset(x_train, y_train)

###############################################################################
# 之前我们需要单独处理 ``x``、``y`` 两组数值。
#
# .. code-block:: python
#
#    xb = x_train[start_i:end_i]
#    yb = y_train[start_i:end_i]
#
#
# 现在可以合并处理:
#
# .. code-block:: python
#
#    xb,yb = train_ds[i*bs : i*bs+bs]
#

model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

###############################################################################
# 使用 ``DataLoader`` 重构
# ------------------------------
#
# 你可以从任何 ``Dataset`` 创建一个 ``DataLoader``，而后由 ``DataLoader`` 负责对数据分批。
# 我们不必再去实现分批代码，如 ``train_ds[i*bs : i*bs+bs]``，``DataLoader`` 会自动为我们提供每批数据。

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

###############################################################################
# 之前我们编写分批代码如下：
#
# .. code-block:: python
#
#    for i in range((n-1)//bs + 1):
#        xb,yb = train_ds[i*bs : i*bs+bs]
#        pred = model(xb)
#
# 现在，我们的循环变得更加简洁，``(xb, yb)`` 自动从DataLoader中加载: 
#
# .. code-block:: python
#
#    for xb,yb in train_dl:
#        pred = model(xb)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

###############################################################################
# 通过使用 PyTorch 中 ``nn.Module``、``nn.Parameter``、``Dataset`` 和 ``DataLoader``，
# 我们实现的训练循代码量并且更容易理解。现在让我们尝试增加一些创建实际有效模型所需的基本功能。
#
# 添加验证集
# -----------------------
#
# 在第一部分中，我们只是实现了使用数据进行训练的逻辑。
# 实际应用中，还需要`验证集 <https://www.fast.ai/2017/11/13/validation-sets/>`_，以确定我们的模型是否存在过拟合问题。
#
# 打乱训练数据是 `十分必要的 <https://www.quora.com/Does-the-order-of-training-data-matter-when-training-neural-networks>`_，
# 以防止批次之间的相关性和过拟合。而验证数据集则无需进此操作，无论打乱与否，验证损失值是相同的，而且打乱操作需要消耗额外的时间，没有实际意义。
#
# 我们将为验证集使用的批量大小设为训练集的两倍。因为验证集不需要进行反向传播，因此需要的内存较少（不需要存储梯度）。
# 因此我我们可以配置较大单批数量，提高计算速度。

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

###############################################################################
# 我们在每个 epoch 结束时计算并打印损失值。
# 
# （请注意，我们在训练之前总是调用 ``model.train()``，在推断之前调用 ``model.eval()``，
# 因为 ``nn.BatchNorm2d`` 和 ``nn.Dropout`` 层会使用，来确保其结果正确。）

model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))

###############################################################################
# 创建 fit() 和 get_data()
# ----------------------------------
#
# 我们在计算训练集和验证集的损失类似的代码，抽取一个独立的函数 ``loss_batch``，用于计算一个批次的损失。
# 
# 训练集传入一个优化器，并使用它执行反向传播，对于验证集，则不传入优化器，不执行反向传播。


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

###############################################################################
# ``fit `` 在每个训练循环中计算训练和验证损失

import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

###############################################################################
# ``get_data`` 返回训练和验证数据集的DataLoader。


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

###############################################################################
# 现在，我们获取数据加载器和拟合模型的整个过程可以用 3 行代码来实现：

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# 你可以使用这三行基本代码来训练各种各样的模型。让我们看看是否可以用来训练一个卷积神经网络（CNN）。
#
# CNN
# -------------
#
# 现在我们将使用三个卷积层构建我们的神经网络。因为前面部分的函数都不假设任何关于模型形式的东西，
# 所以我们可以在不做任何修改的情况下使用它们来训练一个 CNN。
#
# 我们将使用 PyTorch 预定义的 `Conv2d <https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d>`_ 类作为我们的卷积层。
# 我们定义一个具有 3 个卷积层的 CNN。每个卷积层后面跟着一个 ReLU。最后，我们执行平均池化。
# （注意，``view`` 是 PyTorch 版的 Numpy ``reshape``）

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

lr = 0.1

###############################################################################
# `Momentum <https://cs231n.github.io/neural-networks-3/#sgd>`_ 是
# stochastic gradient descent 的一种变体，通过统计更新记录来提升训练速度

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# 使用 ``nn.Sequential``
# ------------------------
#
# 我们可以使用 ``torch.nn`` 中的 `Sequential <https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential>`_ 类
# 来帮助我们简化代码。`` Sequential`` 提供了一种更简单的编写神经网络的方式，其会按顺序运行定义中包含的每个模块。
#
# 我们可以创建一个 ``自定义层``，例如，PyTorch 没有的 view层：

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)

###############################################################################
# 使用 ``Sequential`` 创建模型十分简单：

model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# 包装 ``DataLoader``
# -----------------------------
#
# 我们编写的 CNN 十分简洁，但仅适用于MNIST，因为：
#  - 它假设输入是一个 28 * 28 长的向量
#  - 它假设最终的 CNN 网格大小是 4 * 4 (我们使用的平均池化核大小)

# 让我们去除这两个假设，使我们的模型适用于任何2D单通道图像。
# 首先，我们可以通过将数据预处理移到生成器中来删除 Lambda 层:

def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

###############################################################################
# 接下来，为了让我们定义我们想要的输出张量的大小，而非 *输入* 张量，我们可以用 `nn.AdaptiveAvgPool2d` 替换 `nn.AvgPool2d`。
# 从而使我们的模型可适用于任何大小的输入。

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

###############################################################################
# 让我们查看下结果:

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# 使用 GPU
# ---------------
#
# 在拥有 CUDA 的 GPU的环境中，你可以使用它来加速代码。首先检查你的GPU在PyTorch中是否正常工作:

print(torch.cuda.is_available())

###############################################################################
# 然后创建 device 对象：

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

###############################################################################
# 修改 ``preprocess`` 步骤，将数据移动至 GPU 上:


def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

###############################################################################
# 最后，将模型加载到 GPU 中。

model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

###############################################################################
# 运行速度会提升很多：

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# 总结
# -----------------
#
# 我们使用 PyTorch 编写了一个可以用于多种模型训练的实现，完整的训练代码
# `mnist_sample notebook <https://github.com/fastai/fastai_dev/blob/master/dev_nb/mnist_sample.ipynb>`__.
#
# 后续还可尝试增加其他功能，例如数据增强、超参数调优、监控训练、迁移学习等等。
# 这些功能在fastai库中都有提供，该库是使用本教程中所示的相同设计方法开发的，为希望进一步改进模型的从业人员提供下一步指导。
#
# 我们学习了如何使用
# ``torch.nn``，``torch.optim``，``Dataset``， and ``DataLoader``。现在让我们总结一下:
#
#  - ``torch.nn``:
#    + ``Module``: 创建一个类似于函数的可调用对象，其中包含了状态数据(如神经网络层权重)。它可以自动对包含的参数，进行梯度归零和更新权重等操作。
#    + ``Parameter``: 对张量进行包装，使 ``Module`` 对象在进行反向传播时，可更新权重参数(仅设置 `requires_grad=True` 参数时生效)。
#    + ``functional``: 包含多种激活函数、损失函数，以及无状态的卷积层和线性层等的实现。
#  - ``torch.optim``: 包含多种优化器，例如 ``SGD``，在反向传播过程中优化权重参数(``Parameter``)。
#  - ``Dataset``: 对 ``__len__`` 和 a ``__getitem__`` 方法的抽象接口定义，包含 ``TensorDataset`` 等 PyTorch 实现类。
#  - ``DataLoader``: 对 ``Dataset`` 进行封装，提供分批遍历数据集的能力。
