"""
`基础知识 <intro.html>`_ ||
`快速入门 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
**数据集与数据加载器** ||
`Transforms <transforms_tutorial.html>`_ ||
`构建神经网络 <buildmodel_tutorial.html>`_ ||
`自动微分 <autogradqs_tutorial.html>`_ ||
`优化模型参数 <optimization_tutorial.html>`_ ||
`保存和加载模型 <saveloadrun_tutorial.html>`_

数据集与数据加载器
======================

"""

#################################################################
# 处理数据样本的代码可能会变得混乱且难以维护。理想情况下，我们希望数据集代码与模型训练代码解耦，
# 以提高可读性和模块化。PyTorch 提供了两个数据处理的基本工具：`torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset`，
# 它们允许您使用预加载的数据集以及您自己的数据。`Dataset` 存储样本及其对应的标签，
# 而 `DataLoader` 则为 `Dataset` 包装了一个可迭代对象，以便于访问样本。
#
# PyTorch 域库提供了许多预加载的数据集（例如 FashionMNIST），
# 这些数据集是 `torch.utils.data.Dataset` 的子类，并实现了特定于该数据的函数。
# 它们可以用于模型的原型设计和基准测试。您可以在以下链接找到这些数据集：
# `图像数据集 <https://pytorch.org/vision/stable/datasets.html`、
# `文本数据集 <https://pytorch.org/text/stable/datasets.html>` 和
# `音频数据集 <https://pytorch.org/audio/stable/datasets.html>`。

############################################################
# 加载数据集
# -------------------
#
# 下面是一个从 TorchVision 加载 `Fashion-MNIST <https://research.zalando.com/project/fashion_mnist/fashion_mnist/>`_ 数据集的示例。
# Fashion-MNIST 是 Zalando 的商品图片数据集，包括 60,000 个训练样本和 10,000 个测试样本。每个样本包含一个 28×28 的灰度图像和一个来自 10 个类别之一的标签。
#
# 我们使用以下参数加载 [FashionMNIST 数据集](https://pytorch.org/vision/stable/datasets.html#fashion-mnist)：
#
# - ``root`` 是存储训练/测试数据的路径，
# - ``train`` 指定是训练集还是测试集，
# - ``download=True`` 表示如果数据在 ``root`` 路径中不可用，则从互联网下载数据，
# - ``transform`` 和 ``target_transform`` 指定特征和标签的转换。


from torch.utils.data import DataLoader
from torchvision.io import read_image
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


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


#################################################################
# 迭代和可视化数据集
# -------------------------------------
#
# 我们可以像列表一样手动索引 ``Datasets``：``training_data[index]``。
# 使用 `matplotlib` 来可视化训练数据中的一些样本。

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#################################################################
# ..
#  .. figure:: /_static/img/basics/fashion_mnist.png
#    :alt: fashion_mnist


######################################################################
# --------------
#

#################################################################
# 创建自定义数据集
# --------------
#
# 自定义数据集类必须实现三个函数：`__init__`、`__len__` 和 `__getitem__`。请看这个实现示例；FashionMNIST 图像存储在目录 `img_dir` 中，它们的标签单独存储在 CSV 文件 ``annotations_file`` 中。
#
# 具体代码实现如下：


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


#################################################################
# ``__init__``
# ^^^^^^^^^^^^^^^^^^^^
#
# __init__ 函数在实例化数据集对象时运行一次。我们初始化包含图像的目录、注释文件和两种转换（在下一部分中将更详细地介绍）。
#
# labels.csv 文件的内容如下: ::
#
#     tshirt1.jpg, 0
#     tshirt2.jpg, 0
#     ......
#     ankleboot999.jpg, 9


def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform


#################################################################
# ``__len__``
# ^^^^^^^^^^^^^^^^^^^^
#
# __len__ 函数返回数据集中的样本数量。
#
# Example:


def __len__(self):
    return len(self.img_labels)


#################################################################
# ``__getitem__``
# ^^^^^^^^^^^^^^^^^^^^
#
# __getitem__ 函数加载并返回数据集中给定索引 ``idx`` 的样本。根据索引，它确定图像在磁盘上的位置，
# 使用 ``read_image`` 将其转换为张量，从 ``self.img_labels`` 中的 CSV 数据中检索相应的标签，
# 对它们调用转换函数（如果适用），并以元组形式返回张量图像和相应的标签。

def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label


######################################################################
# --------------
#


#################################################################
# 使用数据加载器为训练准备数据
# -------------------------------------------------
# ``Dataset`` 一次检索我们数据集的一个样本的特征和标签。在训练模型时，我们通常希望以“小批量”的方式传递样本，在每个周期重新随机排列数据以减少模型过拟合，并使用 Python 的 ``multiprocessing`` 加速数据检索。
#
# ``DataLoader`` 是一个可迭代对象，它通过简单的 API 为我们抽象了这些复杂性。


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

###########################
# 通过 DataLoader 进行迭代
# -------------------------------
#
# We have loaded that dataset into the ``DataLoader`` and can iterate through the dataset as needed.
# Each iteration below returns a batch of ``train_features`` and ``train_labels`` (containing ``batch_size=64`` features and labels respectively).
# Because we specified ``shuffle=True``, after we iterate over all batches the data is shuffled (for finer-grained control over
# the data loading order, take a look at `Samplers <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`_).

# 我们已经将数据集加载到 ``DataLoader`` 中，并可以根据需要对数据集进行迭代。
# 下面的每次迭代都会返回一个批次的 ``train_features`` 和 ``train_labels``
# （分别包含 ``batch_size=64`` 个特征和标签）。因为我们指定了 ``shuffle=True``，
# 所以在迭代完所有批次后数据会被重新洗牌（如果想对数据加载顺序进行更精细的控制，
# 请查看 `Samplers <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`_）。

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

######################################################################
# --------------
#

#################################################################
# 延伸阅读
# ----------------
# - `torch.utils.data API <https://pytorch.org/docs/stable/data.html>`_
