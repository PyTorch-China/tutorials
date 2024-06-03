"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
**Optimization** ||
`Save & Load Model <saveloadrun_tutorial.html>`_

优化模型参数
===========================

现在我们有了模型和数据，是时候通过在数据上优化模型参数来训练、验证和测试我们的模型了。训练模型是一个迭代过程；在每次迭代中，模型会对输出进行猜测，
计算其猜测的误差（*损失-loss*），收集误差相对于其参数的导数（如我们在`前一节 <autograd_tutorial.html>`_中所见），并使用梯度下降法**优化**这些参数。
有关此过程的更详细讲解，请查看 3Blue1Brown 的这个视频`反向传播 <https://www.youtube.com/watch?v=tIeHLnjs5U8>`__。

前置代码
-----------------
我们加载前几节中的`数据集和数据加载器 <data_tutorial.html>`_和`构建模型 <buildmodel_tutorial.html>`_的代码。
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()


##############################################
# 超参数
# -----------------
#
# 超参数是可调参数，它们可以让您控制模型的优化过程。不同的超参数值会影响模型的训练和收敛速度
# （`阅读更多 <https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html>`__ 关于超参数调整的内容）。
#
# 我们为训练定义以下超参数：
#  - **Epoch数量** - 迭代整个数据集的次数
#  - **批量大小** - 在更新参数之前，通过网络传播的数据样本数量
#  - **学习率** - 在每个批次/epoch中更新模型参数的幅度。较小的值会导致学习速度缓慢，而较大的值可能会导致训练过程中出现不可预测的行为。

epochs = 5
batch_size = 64
learning_rate = 1e-3

#####################################
# 优化循环
# -----------------
#
# 一旦设置好超参数，我们就可以用优化循环来训练和优化我们的模型。优化循环的每次迭代称为一个**epoch**。
#
# 每个epoch由两个主要部分组成：
#  - **训练循环** - 迭代训练数据集并尝试收敛到最佳参数。
#  - **验证/测试循环** - 迭代测试数据集以检查模型性能是否有提高。
#
# 让我们简要了解训练循环中使用的一些概念。跳到前面查看优化循环的 :ref:`full-impl-label` 。
#
# 损失函数(Loss Function)
# ~~~~~~~~~~~~~~~~~
#
# 当面对一些训练数据时，我们未训练的网络可能不会给出正确的答案。**损失函数(Loss Function)** 衡量获得的结果与目标值的差异程度，
# 这是我们在训练过程中希望最小化的。要计算损失，我们使用给定数据样本的输入进行预测，并将其与真实的数据标签值进行比较。
#
# 常见的损失函数包括用于回归任务的`nn.MSELoss <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`_（均方误差），
# 以及用于分类的`nn.NLLLoss <https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss>`_（负对数似然）。
# `nn.CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss>`_结合了``nn.LogSoftmax``和``nn.NLLLoss``。
#
# 我们将模型的输出logits传递给``nn.CrossEntropyLoss``，它将标准化logits并计算预测误差。

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()



#####################################
# 优化器
# ~~~~~~~~~~~~~~~~~
#
# 优化是调整模型参数以减少每次训练步骤中的模型误差的过程。**优化算法**定义了这个过程如何进行（在这个例子中我们使用随机梯度下降法）。
# 所有优化逻辑都封装在``optimizer``对象中。在这里，我们使用SGD优化器；此外，PyTorch中还有许多`不同的优化器 <https://pytorch.org/docs/stable/optim.html>`_，
# 如ADAM和RMSProp，它们对不同类型的模型和数据效果更好。
#
# 我们通过注册需要训练的模型参数并传入学习率超参数来初始化优化器。

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#####################################
# 在训练循环中，优化分为三个步骤：
#  * 调用``optimizer.zero_grad()``来重置模型参数的梯度。默认情况下，梯度会累加；为防止重复计算，我们在每次迭代时显式将其归零。
#  * 调用``loss.backward()``反向传播预测损失。PyTorch会将损失相对于每个参数的梯度存储下来。
#  * 一旦我们有了梯度，就调用``optimizer.step()``通过反向传播中收集的梯度来调整参数。


########################################
# .. _full-impl-label:
#
# 完整实现
# -----------------------
# 我们定义了``train_loop``来循环执行优化代码，并定义了``test_loop``来评估模型在测试数据上的性能。

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


########################################
# 我们初始化损失函数和优化器，并将它们传递给``train_loop``和``test_loop``。
# 您可以尝试增加epoch的数量以观察模型性能的提升。

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")



#################################################################
# 延伸阅读
# -----------------------
# - `Loss Functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# - `torch.optim <https://pytorch.org/docs/stable/optim.html>`_
# - `Warmstart Training a Model <https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html>`_
#
