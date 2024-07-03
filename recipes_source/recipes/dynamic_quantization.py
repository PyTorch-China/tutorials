"""
动态量化
====================

在这个示例中,您将看到如何利用动态量化来加速 LSTM 风格的循环神经网络的推理。这可以减小模型权重的大小,并加快模型执行速度。

介绍
-------------

在设计神经网络时,可以做出多种权衡。在模型开发和训练期间,您可以改变循环神经网络中的层数和参数数量,在模型大小和/或模型延迟或吞吐量与精度之间进行权衡。由于您需要重复模型训练过程,因此这种改变需要大量的时间和计算资源。量化为您提供了一种在已知模型上在性能和模型精度之间进行权衡的方式,而无需重新训练模型。

您可以在单个会话中尝试一下,您肯定会显著减小模型大小,并可能在不会损失太多精度的情况下获得显著的延迟减少。

什么是动态量化?
-----------------------------

量化网络意味着将其转换为使用较低精度的整数表示形式来表示权重和/或激活。这可以减小模型大小,并允许在 CPU 或 GPU 上使用更高吞吐量的数学运算。

从浮点数转换为整数值时,您实际上是将浮点数乘以某个比例因子,然后将结果舍入为整数。不同的量化方法在确定该比例因子的方式上有所不同。

这里介绍的动态量化的关键思想是,我们将根据运行时观察到的数据范围动态确定激活的比例因子。这可确保比例因子被"调整"为尽可能保留每个观察到的数据集的信号。

另一方面,模型参数在模型转换期间是已知的,它们会提前转换并以 INT8 形式存储。

量化模型中的算术运算使用矢量化的 INT8 指令完成。累加通常使用 INT16 或 INT32 来避免溢出。如果下一层是量化的,则将此较高精度值缩放回 INT8;如果是输出,则将其转换为 FP32。

动态量化相对来说没有太多需要调整的参数,因此非常适合作为将 LSTM 模型转换为部署的标准部分添加到生产管道中。

.. note::
   本示例中采用的方法的局限性

   本示例提供了对 PyTorch 中动态量化功能的快速介绍,以及使用它的工作流程。我们的重点是解释用于转换模型的特定函数。为了简洁和清晰,我们做出了一些重大简化,包括:

1. 您将从一个最小的 LSTM 网络开始
2. 您只需用随机隐藏状态初始化网络
3. 您将使用随机输入来测试网络
4. 您不会在本教程中训练网络
5. 您将看到,与我们开始时的浮点网络相比,量化后的网络更小且运行速度更快
6. 您将看到,量化网络产生的输出张量值与 FP32 网络输出的值在同一数量级,但我们并未在这里展示该技术在经过训练的 LSTM 上能够保留较高模型精度的情况

您将了解如何进行动态量化,并能够看到内存使用和延迟时间的潜在减小。关于该技术在经过训练的 LSTM 上能够保留较高模型精度的演示,将留待更高级的教程。如果您想直接进入更严格的处理,请继续学习 `高级动态量化教程 <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>`__。

步骤
-------------

本示例包含 5 个步骤。

1. 设置 - 在这里,您定义一个非常简单的 LSTM,导入模块,并建立一些随机输入张量。

2. 执行量化 - 在这里,您实例化一个浮点模型,然后创建其量化版本。

3. 查看模型大小 - 在这里,您显示模型大小变小了。

4. 查看延迟 - 在这里,您运行两个模型并比较模型运行时间(延迟)。

5. 查看精度 - 在这里,您运行两个模型并比较输出。

1: 设置
~~~~~~~~~~~~~~~
这是一段直接的代码,用于为本示例的其余部分做准备。

我们在这里导入的唯一模块是 torch.quantization,它包含了 PyTorch 的量化算子和转换函数。我们还定义了一个非常简单的 LSTM 模型,并设置了一些输入。
"""

# 导入本示例中使用的模块
import copy
import os
import time

import torch
import torch.nn as nn
import torch.quantization


# 为演示目的定义一个非常简单的 LSTM
# 在这种情况下,我们只是包装了 ``nn.LSTM``、一层,没有预处理或后处理
# 受到以下教程的启发:
# `序列模型和长短期记忆网络教程 <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html>`_, 作者 Robert Guthrie
# 和 `动态量化教程 <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>`__。
class lstm_for_demonstration(nn.Module):
    """基本的长短期记忆风格模型,只是包装了 ``nn.LSTM``
    不应用于除演示之外的任何其他用途。
    """

    def __init__(self, in_dim, out_dim, depth):
        super(lstm_for_demonstration, self).__init__()
        self.lstm = nn.LSTM(in_dim, out_dim, depth)

    def forward(self, inputs, hidden):
        out, hidden = self.lstm(inputs, hidden)
        return out, hidden


torch.manual_seed(29592)  # 设置种子以获得可重复结果

# 形状参数
model_dimension = 8
sequence_length = 20
batch_size = 1
lstm_depth = 1

# 随机输入数据
inputs = torch.randn(sequence_length, batch_size, model_dimension)
# hidden 实际上是初始隐藏状态和初始细胞状态的元组
hidden = (
    torch.randn(lstm_depth, batch_size, model_dimension),
    torch.randn(lstm_depth, batch_size, model_dimension),
)


######################################################################
# 2: 执行量化
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 现在我们来执行有趣的部分。首先,我们创建一个名为 ``float_lstm`` 的模型实例,然后我们将对其进行量化。我们将使用 `torch.quantization.quantize_dynamic <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`__ 函数,它接受模型、我们希望量化的子模块列表(如果存在)以及目标数据类型。此函数返回原始模型的量化版本,作为一个新模块。
#
# 就这么简单。
#

# 这是我们的浮点实例
float_lstm = lstm_for_demonstration(model_dimension, model_dimension, lstm_depth)

# 这是执行量化的调用
quantized_lstm = torch.quantization.quantize_dynamic(
    float_lstm, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# 显示所做的更改
print("这是该模块的浮点版本:")
print(float_lstm)
print("")
print("现在是量化版本:")
print(quantized_lstm)


######################################################################
# 3. 查看模型大小
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 我们已经量化了模型。这给我们带来了什么好处?好处之一是我们用 INT8 值(和一些记录的比例因子)替换了 FP32 模型参数。这意味着存储和移动数据的大小减小了约 75%。使用默认值时,下面显示的减小量将小于 75%,但如果您将模型大小增加到更大值(例如将 model_dimension 设置为 80),随着存储的模型大小越来越多地由参数值主导,减小量将趋近于 4 倍。
#


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("模型: ", label, " \t", "大小 (KB):", size / 1e3)
    os.remove("temp.p")
    return size


# 比较大小
f = print_size_of_model(float_lstm, "fp32")
q = print_size_of_model(quantized_lstm, "int8")
print("{0:.2f} 倍更小".format(f / q))


######################################################################
# 4. 查看延迟
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 第二个好处是量化模型通常会运行得更快。这是由于多种效果的组合,至少包括:
#
# 1. 减少了移动参数数据所花费的时间
# 2. INT8 操作更快
#
# 如您所见,这个超级简单的网络的量化版本运行速度更快。对于更复杂的网络通常也是如此,但正如他们所说,"您的里程可能会有所不同",这取决于许多因素,包括模型的结构和您运行的硬件。
#

# 比较性能
print("浮点 FP32")

#####################################################################
# .. code-block:: python
#
#    %timeit float_lstm.forward(inputs, hidden)

print("量化 INT8")

######################################################################
# .. code-block:: python
#
#    %timeit quantized_lstm.forward(inputs,hidden)


######################################################################
# 5: 查看精度
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 我们不会在这里仔细查看精度,因为我们使用的是随机初始化的网络,而不是经过正确训练的网络。但是,我认为值得快速展示一下量化网络确实产生了与原始网络"同一数量级"的输出张量值。
#
# 有关更详细的分析,请参阅本示例结尾处引用的更高级教程。
#

# 运行浮点模型
out1, hidden1 = float_lstm(inputs, hidden)
mag1 = torch.mean(abs(out1)).item()
print("FP32 模型中输出张量值的绝对值均值为 {0:.5f} ".format(mag1))

# 运行量化模型
out2, hidden2 = quantized_lstm(inputs, hidden)
mag2 = torch.mean(abs(out2)).item()
print("INT8 模型中输出张量值的绝对值均值为 {0:.5f}".format(mag2))

# 比较它们
mag3 = torch.mean(abs(out1 - out2)).item()
print(
    "输出张量之间差值的绝对值均值为 {0:.5f}，或占 {1:.2f} 百分比".format(
        mag3, mag3 / mag1 * 100
    )
)


######################################################################
# 了解更多
# ------------
# 我们已经解释了什么是动态量化,它带来了什么好处,您已经使用 ``torch.quantization.quantize_dynamic()`` 函数快速量化了一个简单的 LSTM 模型。
#
# 这是对该材料的快速和高级处理;要了解更多详细信息,请继续学习 `(beta) 动态量化 LSTM 词语言模型教程 <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>`_。
#
#
# 其他资源
# --------------------
#
# * `量化 API 文档 <https://pytorch.org/docs/stable/quantization.html>`_
# * `(beta) 动态量化 BERT <https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html>`_
# * `(beta) 动态量化 LSTM 词语言模型 <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>`_
# * `PyTorch 量化介绍 <https://pytorch.org/blog/introduction-to-quantization-on-pytorch/>`_
#
