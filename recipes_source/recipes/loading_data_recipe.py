"""
PyTorch 加载数据
=======================
PyTorch 提供了广泛的神经网络构建模块,并拥有简单、直观且稳定的 API。PyTorch包含用于准备和加载常见数据集的工具包,为训练模型提供数据。


简介
----
PyTorch 数据加载工具的核心类为 `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__ 。
它表示数据集上的一个 Python 可迭代对象。PyTorch 提供了内置的高质量数据集，
可通过 `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__ 使用。使用这些数据集可通过：

* `torchvision <https://pytorch.org/vision/stable/datasets.html>`__
* `torchaudio <https://pytorch.org/audio/stable/datasets.html>`__
* `torchtext <https://pytorch.org/text/stable/datasets.html>`__

未来会持续新增。
通过使用 ``torchaudio.datasets.YESNO`` 中的 ``yesno`` 数据集，我们将演示如何有效地将数据从 PyTorch ``Dataset`` 加载到 PyTorch ``DataLoader`` 中。
"""


# 安装
# -----
# 在开始之前,我们需要安装 ``torchaudio`` 以访问该数据集。

# pip install torchaudio

#######################################################
# 如果在Google Colab中运行,请取消注释以下行:

# !pip install torchaudio

#############################
# 使用步骤
# -----
#
# 1. 导入加载数据所需的所有必要库
# 2. 访问数据集中的数据
# 3. 加载数据
# 4. 遍历数据
# 5. [可选] 可视化数据
#
#
# 1. 导入加载数据所需的必要库
# ---------------------------------------------------------------
#
# 对于本例,我们将使用 ``torch`` 和 ``torchaudio``。根据使用的内置数据集,您还可以安装并导入 
# ``torchvision`` 或 ``torchtext``。
#

import torch
import torchaudio


######################################################################
# 2. 访问数据集中的数据
# ---------------------------------------------------------------
#
# ``torchaudio`` 中的 ``yesno`` 数据集包含一个人说希伯来语"是"或"否"的60个录音,
# 每个录音长度为8个单词(`更多信息 <https://www.openslr.org/1/>`__)。
#
# ``torchaudio.datasets.YESNO`` 创建了一个 ``yesno`` 数据集。

torchaudio.datasets.YESNO(
     root='./',
     url='http://www.openslr.org/resources/1/waves_yesno.tar.gz',
     folder_in_archive='waves_yesno',
     download=True)

###########################################################################
# 数据集中的每个条目都是一个元组,形式为:(波形,采样率,标签)。
#
# 您必须为 ``yesno`` 数据集设置一个 ``root``目录,用于存放训练和测试数据集。其他参数是可选的,显示了它们的默认值。
# 以下是其他参数的一些有用信息:

# * ``download``: 如果为True,则从互联网下载数据集并将其放在root目录中。如果数据集已下载,则不会重新下载。
#
# 让我们访问 ``yesno`` 数据:
#

# ``yesno`` 中的一个数据点是一个元组(波形,采样率,标签),其中标签是一个整数列表,1表示yes,0表示no。
yesno_data = torchaudio.datasets.YESNO('./', download=True)

# 选择数据点编号3,查看 ``yesno_data`` 的示例:
n = 3
waveform, sample_rate, labels = yesno_data[n]
print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))


######################################################################
# 在实践中使用这些数据时,最好将数据划分为"训练"数据集和"测试"数据集。这可确保您有未使用的数据来测试模型的性能。
#
# 3. 加载数据
# ---------------------------------------------------------------
#
# 现在我们可以访问数据集,我们必须通过 ``torch.utils.data.DataLoader`` 传递它。
# ``DataLoader`` 将数据集和采样器组合在一起,返回数据集上的一个可迭代对象。
#

data_loader = torch.utils.data.DataLoader(yesno_data,
                                          batch_size=1,
                                          shuffle=True)


######################################################################
# 4. 遍历数据
# ---------------------------------------------------------------
#
# 我们的数据现在可以使用 ``data_loader`` 进行迭代。在开始训练模型时,这将是必需的!
# 您会注意到,现在 ``data_loader`` 对象中的每个数据条目都转换为一个张量,其中包含表示波形、采样率和标签的张量。
#

for data in data_loader:
  print("Data: ", data)
  print("Waveform: {}\nSample rate: {}\nLabels: {}".format(data[0], data[1], data[2]))
  break


######################################################################
# 5. [可选] 可视化数据
# ---------------------------------------------------------------
#
# 您可以选择可视化数据,以进一步了解 ``DataLoader`` 的输出。
#


import matplotlib.pyplot as plt

print(data[0][0].numpy())

plt.figure()
plt.plot(waveform.t().numpy())


######################################################################
# 祝贺您!您已成功在PyTorch中加载数据。
#
# 学习更多
# ----------
#
# 查看这些其他教程,继续您的学习:
#
# - `定义神经网络 <https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html>`__
# - `PyTorch中的state_dict 是什么 <https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html>`__