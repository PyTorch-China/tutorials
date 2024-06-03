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
# PyTorch 提供了创建 随机 或 零 填充张量的方法，我们将使用这些方法为一个简单的线性模型创建权重和偏置。
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
# PyTorch 提供了许多预先编写的损失函数、激活函数等，你仍可以使用普通的 Python 轻松编写自己的函数。
# PyTorch 会为你的函数自动创建 GPU 或矢量化 CPU 代码。

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

######################################################################################
# 在上面的代码中，``@`` 表示矩阵乘法操作。在一个数据批次上调用我们的函数（在本例中为64张图像）。
# 这就是一次 *前向传递*。请注意，由于我们从随机权重开始，此时我们的预测结果不会比随机来的好。

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)

###############################################################################
# 如你所见，``preds`` 张量不仅包含张量值，还包含梯度函数。我们稍后会用它来进行反向传播。
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
# 对于每个预测，如果具有最大值的索引与目标值匹配，则预测是正确的。

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

###############################################################################
# 检查我们随机模型的准确率，这样我们就可以看到随着损失的改善，准确率是否有所提高。

print(accuracy(preds, yb))

###############################################################################
# We can now run a training loop.  For each iteration, we will:
#
# - select a mini-batch of data (of size ``bs``)
# - use the model to make predictions
# - calculate the loss
# - ``loss.backward()`` updates the gradients of the model, in this case, ``weights``
#   and ``bias``.
#
# We now use these gradients to update the weights and bias.  We do this
# within the ``torch.no_grad()`` context manager, because we do not want these
# actions to be recorded for our next calculation of the gradient.  You can read
# more about how PyTorch's Autograd records operations
# `here <https://pytorch.org/docs/stable/notes/autograd.html>`_.
#
# We then set the
# gradients to zero, so that we are ready for the next loop.
# Otherwise, our gradients would record a running tally of all the operations
# that had happened (i.e. ``loss.backward()`` *adds* the gradients to whatever is
# already stored, rather than replacing them).
#
# .. tip:: You can use the standard python debugger to step through PyTorch
#    code, allowing you to check the various variable values at each step.
#    Uncomment ``set_trace()`` below to try it out.
#
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
# That's it: we've created and trained a minimal neural network (in this case, a
# logistic regression, since we have no hidden layers) entirely from scratch!
#
# Let's check the loss and accuracy and compare those to what we got
# earlier. We expect that the loss will have decreased and accuracy to
# have increased, and they have.

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

###############################################################################
# Using ``torch.nn.functional``
# ------------------------------
#
# We will now refactor our code, so that it does the same thing as before, only
# we'll start taking advantage of PyTorch's ``nn`` classes to make it more concise
# and flexible. At each step from here, we should be making our code one or more
# of: shorter, more understandable, and/or more flexible.
#
# The first and easiest step is to make our code shorter by replacing our
# hand-written activation and loss functions with those from ``torch.nn.functional``
# (which is generally imported into the namespace ``F`` by convention). This module
# contains all the functions in the ``torch.nn`` library (whereas other parts of the
# library contain classes). As well as a wide range of loss and activation
# functions, you'll also find here some convenient functions for creating neural
# nets, such as pooling functions. (There are also functions for doing convolutions,
# linear layers, etc, but as we'll see, these are usually better handled using
# other parts of the library.)
#
# If you're using negative log likelihood loss and log softmax activation,
# then Pytorch provides a single function ``F.cross_entropy`` that combines
# the two. So we can even remove the activation function from our model.

import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

###############################################################################
# Note that we no longer call ``log_softmax`` in the ``model`` function. Let's
# confirm that our loss and accuracy are the same as before:

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

###############################################################################
# Refactor using ``nn.Module``
# -----------------------------
# Next up, we'll use ``nn.Module`` and ``nn.Parameter``, for a clearer and more
# concise training loop. We subclass ``nn.Module`` (which itself is a class and
# able to keep track of state).  In this case, we want to create a class that
# holds our weights, bias, and method for the forward step.  ``nn.Module`` has a
# number of attributes and methods (such as ``.parameters()`` and ``.zero_grad()``)
# which we will be using.
#
# .. note:: ``nn.Module`` (uppercase M) is a PyTorch specific concept, and is a
#    class we'll be using a lot. ``nn.Module`` is not to be confused with the Python
#    concept of a (lowercase ``m``) `module <https://docs.python.org/3/tutorial/modules.html>`_,
#    which is a file of Python code that can be imported.

from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

###############################################################################
# Since we're now using an object instead of just using a function, we
# first have to instantiate our model:

model = Mnist_Logistic()

###############################################################################
# Now we can calculate the loss in the same way as before. Note that
# ``nn.Module`` objects are used as if they are functions (i.e they are
# *callable*), but behind the scenes Pytorch will call our ``forward``
# method automatically.

print(loss_func(model(xb), yb))

###############################################################################
# Previously for our training loop we had to update the values for each parameter
# by name, and manually zero out the grads for each parameter separately, like this:
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
# Now we can take advantage of model.parameters() and model.zero_grad() (which
# are both defined by PyTorch for ``nn.Module``) to make those steps more concise
# and less prone to the error of forgetting some of our parameters, particularly
# if we had a more complicated model:
#
# .. code-block:: python
#
#    with torch.no_grad():
#        for p in model.parameters(): p -= p.grad * lr
#        model.zero_grad()
#
#
# We'll wrap our little training loop in a ``fit`` function so we can run it
# again later.

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

print(loss_func(model(xb), yb))

###############################################################################
# Refactor using ``nn.Linear``
# ----------------------------
#
# We continue to refactor our code.  Instead of manually defining and
# initializing ``self.weights`` and ``self.bias``, and calculating ``xb  @
# self.weights + self.bias``, we will instead use the Pytorch class
# `nn.Linear <https://pytorch.org/docs/stable/nn.html#linear-layers>`_ for a
# linear layer, which does all that for us. Pytorch has many types of
# predefined layers that can greatly simplify our code, and often makes it
# faster too.

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

###############################################################################
# We instantiate our model and calculate the loss in the same way as before:

model = Mnist_Logistic()
print(loss_func(model(xb), yb))

###############################################################################
# We are still able to use our same ``fit`` method as before.

fit()

print(loss_func(model(xb), yb))

###############################################################################
# Refactor using ``torch.optim``
# ------------------------------
#
# Pytorch also has a package with various optimization algorithms, ``torch.optim``.
# We can use the ``step`` method from our optimizer to take a forward step, instead
# of manually updating each parameter.
#
# This will let us replace our previous manually coded optimization step:
#
# .. code-block:: python
#
#    with torch.no_grad():
#        for p in model.parameters(): p -= p.grad * lr
#        model.zero_grad()
#
# and instead use just:
#
# .. code-block:: python
#
#    opt.step()
#    opt.zero_grad()
#
# (``optim.zero_grad()`` resets the gradient to 0 and we need to call it before
# computing the gradient for the next minibatch.)

from torch import optim

###############################################################################
# We'll define a little function to create our model and optimizer so we
# can reuse it in the future.

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
# Refactor using Dataset
# ------------------------------
#
# PyTorch has an abstract Dataset class.  A Dataset can be anything that has
# a ``__len__`` function (called by Python's standard ``len`` function) and
# a ``__getitem__`` function as a way of indexing into it.
# `This tutorial <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`_
# walks through a nice example of creating a custom ``FacialLandmarkDataset`` class
# as a subclass of ``Dataset``.
#
# PyTorch's `TensorDataset <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#TensorDataset>`_
# is a Dataset wrapping tensors. By defining a length and way of indexing,
# this also gives us a way to iterate, index, and slice along the first
# dimension of a tensor. This will make it easier to access both the
# independent and dependent variables in the same line as we train.

from torch.utils.data import TensorDataset

###############################################################################
# Both ``x_train`` and ``y_train`` can be combined in a single ``TensorDataset``,
# which will be easier to iterate over and slice.

train_ds = TensorDataset(x_train, y_train)

###############################################################################
# Previously, we had to iterate through minibatches of ``x`` and ``y`` values separately:
#
# .. code-block:: python
#
#    xb = x_train[start_i:end_i]
#    yb = y_train[start_i:end_i]
#
#
# Now, we can do these two steps together:
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
# Refactor using ``DataLoader``
# ------------------------------
#
# PyTorch's ``DataLoader`` is responsible for managing batches. You can
# create a ``DataLoader`` from any ``Dataset``. ``DataLoader`` makes it easier
# to iterate over batches. Rather than having to use ``train_ds[i*bs : i*bs+bs]``,
# the ``DataLoader`` gives us each minibatch automatically.

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

###############################################################################
# Previously, our loop iterated over batches ``(xb, yb)`` like this:
#
# .. code-block:: python
#
#    for i in range((n-1)//bs + 1):
#        xb,yb = train_ds[i*bs : i*bs+bs]
#        pred = model(xb)
#
# Now, our loop is much cleaner, as ``(xb, yb)`` are loaded automatically from the data loader:
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
# Thanks to PyTorch's ``nn.Module``, ``nn.Parameter``, ``Dataset``, and ``DataLoader``,
# our training loop is now dramatically smaller and easier to understand. Let's
# now try to add the basic features necessary to create effective models in practice.
#
# Add validation
# -----------------------
#
# In section 1, we were just trying to get a reasonable training loop set up for
# use on our training data.  In reality, you **always** should also have
# a `validation set <https://www.fast.ai/2017/11/13/validation-sets/>`_, in order
# to identify if you are overfitting.
#
# Shuffling the training data is
# `important <https://www.quora.com/Does-the-order-of-training-data-matter-when-training-neural-networks>`_
# to prevent correlation between batches and overfitting. On the other hand, the
# validation loss will be identical whether we shuffle the validation set or not.
# Since shuffling takes extra time, it makes no sense to shuffle the validation data.
#
# We'll use a batch size for the validation set that is twice as large as
# that for the training set. This is because the validation set does not
# need backpropagation and thus takes less memory (it doesn't need to
# store the gradients). We take advantage of this to use a larger batch
# size and compute the loss more quickly.

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

###############################################################################
# We will calculate and print the validation loss at the end of each epoch.
#
# (Note that we always call ``model.train()`` before training, and ``model.eval()``
# before inference, because these are used by layers such as ``nn.BatchNorm2d``
# and ``nn.Dropout`` to ensure appropriate behavior for these different phases.)

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
# Create fit() and get_data()
# ----------------------------------
#
# We'll now do a little refactoring of our own. Since we go through a similar
# process twice of calculating the loss for both the training set and the
# validation set, let's make that into its own function, ``loss_batch``, which
# computes the loss for one batch.
#
# We pass an optimizer in for the training set, and use it to perform
# backprop.  For the validation set, we don't pass an optimizer, so the
# method doesn't perform backprop.


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

###############################################################################
# ``fit`` runs the necessary operations to train our model and compute the
# training and validation losses for each epoch.

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
# ``get_data`` returns dataloaders for the training and validation sets.


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

###############################################################################
# Now, our whole process of obtaining the data loaders and fitting the
# model can be run in 3 lines of code:

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# You can use these basic 3 lines of code to train a wide variety of models.
# Let's see if we can use them to train a convolutional neural network (CNN)!
#
# Switch to CNN
# -------------
#
# We are now going to build our neural network with three convolutional layers.
# Because none of the functions in the previous section assume anything about
# the model form, we'll be able to use them to train a CNN without any modification.
#
# We will use PyTorch's predefined
# `Conv2d <https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d>`_ class
# as our convolutional layer. We define a CNN with 3 convolutional layers.
# Each convolution is followed by a ReLU.  At the end, we perform an
# average pooling.  (Note that ``view`` is PyTorch's version of Numpy's
# ``reshape``)

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
# `Momentum <https://cs231n.github.io/neural-networks-3/#sgd>`_ is a variation on
# stochastic gradient descent that takes previous updates into account as well
# and generally leads to faster training.

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# Using ``nn.Sequential``
# ------------------------
#
# ``torch.nn`` has another handy class we can use to simplify our code:
# `Sequential <https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential>`_ .
# A ``Sequential`` object runs each of the modules contained within it, in a
# sequential manner. This is a simpler way of writing our neural network.
#
# To take advantage of this, we need to be able to easily define a
# **custom layer** from a given function.  For instance, PyTorch doesn't
# have a `view` layer, and we need to create one for our network. ``Lambda``
# will create a layer that we can then use when defining a network with
# ``Sequential``.

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)

###############################################################################
# The model created with ``Sequential`` is simple:

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
# Wrapping ``DataLoader``
# -----------------------------
#
# Our CNN is fairly concise, but it only works with MNIST, because:
#  - It assumes the input is a 28\*28 long vector
#  - It assumes that the final CNN grid size is 4\*4 (since that's the average pooling kernel size we used)
#
# Let's get rid of these two assumptions, so our model works with any 2d
# single channel image. First, we can remove the initial Lambda layer by
# moving the data preprocessing into a generator:

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
# Next, we can replace ``nn.AvgPool2d`` with ``nn.AdaptiveAvgPool2d``, which
# allows us to define the size of the *output* tensor we want, rather than
# the *input* tensor we have. As a result, our model will work with any
# size input.

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
# Let's try it out:

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# Using your GPU
# ---------------
#
# If you're lucky enough to have access to a CUDA-capable GPU (you can
# rent one for about $0.50/hour from most cloud providers) you can
# use it to speed up your code. First check that your GPU is working in
# Pytorch:

print(torch.cuda.is_available())

###############################################################################
# And then create a device object for it:

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

###############################################################################
# Let's update ``preprocess`` to move batches to the GPU:


def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

###############################################################################
# Finally, we can move our model to the GPU.

model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

###############################################################################
# You should find it runs faster now:

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

###############################################################################
# Closing thoughts
# -----------------
#
# We now have a general data pipeline and training loop which you can use for
# training many types of models using Pytorch. To see how simple training a model
# can now be, take a look at the `mnist_sample notebook <https://github.com/fastai/fastai_dev/blob/master/dev_nb/mnist_sample.ipynb>`__.
#
# Of course, there are many things you'll want to add, such as data augmentation,
# hyperparameter tuning, monitoring training, transfer learning, and so forth.
# These features are available in the fastai library, which has been developed
# using the same design approach shown in this tutorial, providing a natural
# next step for practitioners looking to take their models further.
#
# We promised at the start of this tutorial we'd explain through example each of
# ``torch.nn``, ``torch.optim``, ``Dataset``, and ``DataLoader``. So let's summarize
# what we've seen:
#
#  - ``torch.nn``:
#
#    + ``Module``: creates a callable which behaves like a function, but can also
#      contain state(such as neural net layer weights). It knows what ``Parameter`` (s) it
#      contains and can zero all their gradients, loop through them for weight updates, etc.
#    + ``Parameter``: a wrapper for a tensor that tells a ``Module`` that it has weights
#      that need updating during backprop. Only tensors with the `requires_grad` attribute set are updated
#    + ``functional``: a module(usually imported into the ``F`` namespace by convention)
#      which contains activation functions, loss functions, etc, as well as non-stateful
#      versions of layers such as convolutional and linear layers.
#  - ``torch.optim``: Contains optimizers such as ``SGD``, which update the weights
#    of ``Parameter`` during the backward step
#  - ``Dataset``: An abstract interface of objects with a ``__len__`` and a ``__getitem__``,
#    including classes provided with Pytorch such as ``TensorDataset``
#  - ``DataLoader``: Takes any ``Dataset`` and creates an iterator which returns batches of data.
