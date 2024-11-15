"""
`简介 <introyt1_tutorial.html>`_ ||
`张量 <tensors_deeper_tutorial.html>`_ ||
`自动微分 <autogradyt_tutorial.html>`_ ||
**构建模型** ||
`TensorBoard支持 <tensorboardyt_tutorial.html>`_ ||
`训练模型 <trainingyt.html>`_ ||
`模型理解 <captumyt.html>`_

使用 PyTorch 构建模型
============================

跟随下面的视频或在 `youtube <https://www.youtube.com/watch?v=OSqIP-mOWOI>`__ 上观看.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/OSqIP-mOWOI" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

``torch.nn.Module`` 和 ``torch.nn.Parameter``
----------------------------------------------

在这个视频中,我们将讨论 PyTorch 提供的一些用于构建深度学习网络的工具。

除了 ``Parameter`` 之外,我们在本视频中讨论的所有类都是 ``torch.nn.Module`` 的子类。
这是 PyTorch 的基类,旨在封装特定于 PyTorch 模型及其组件的行为。

``torch.nn.Module`` 的一个重要行为是注册参数。如果特定的 ``Module`` 子类具有学习权重,
这些权重将表示为 ``torch.nn.Parameter`` 的实例。``Parameter`` 类是 ``torch.Tensor`` 的子类,
具有特殊行为,即当它们被分配为 ``Module`` 的属性时,它们将被添加到该模块的参数列表中。
可以通过 ``Module`` 类上的 ``parameters()`` 方法访问这些参数。

作为一个简单的例子,这里有一个非常简单的模型,包含两个线性层和一个激活函数。
我们将创建它的一个实例,并要求它报告其参数:

"""

import torch

class TinyModel(torch.nn.Module):
    
    def __init__(self):
        super(TinyModel, self).__init__()
        
        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)


#########################################################################
# 这显示了 PyTorch 模型的基本结构:有一个 ``__init__()`` 方法定义模型的层和其他组件,
# 还有一个 ``forward()`` 方法执行计算。注意我们可以打印模型或任何子模块,以了解其结构。
# 
# 常见层类型
# ------------------
# 
# 线性层
# ~~~~~~~~~~~~~
# 
# 最基本的神经网络层类型是 *线性* 或 *全连接* 层。这是一种每个输入都会影响该层每个输出的层,
# 其影响程度由层的权重指定。如果一个模型有 *m* 个输入和 *n* 个输出,
# 权重将是一个 *m* x *n* 矩阵。例如:
# 

lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)


#########################################################################
# 如果你将 ``x`` 与线性层的权重相乘,并加上偏置,你会发现得到的是输出向量 ``y``。
# 
# 另一个需要注意的重要特性是:当我们用 ``lin.weight`` 检查层的权重时,
# 它将自己报告为一个 ``Parameter``(这是 ``Tensor`` 的子类),
# 并让我们知道它正在使用 autograd 跟踪梯度。这是 ``Parameter`` 与 ``Tensor`` 不同的默认行为。
# 
# 线性层在深度学习模型中被广泛使用。你会经常在分类器模型的末端看到它们,其中最后一层将有 *n* 个输出,
# 其中 *n* 是分类器所处理的类别数。
# 
# 卷积层
# ~~~~~~~~~~~~~~~~~~~~
# 
# *卷积* 层被设计用于处理具有高度空间相关性的数据。它们在计算机视觉领域非常常用,
# 用于检测组成更高级特征的紧密特征组。它们也出现在其他上下文中 - 例如,在 NLP 应用程序中,
# 一个单词的直接上下文(即序列中附近的其他单词)可能会影响句子的含义。
# 
# 我们在之前的视频中看到了 LeNet5 中的卷积层:
# 

import torch.functional as F


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 输入图像通道(黑白), 6 输出通道, 5x5 平方卷积核
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # 一个仿射操作: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 来自图像维度
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # 在 (2, 2) 窗口上进行最大池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果尺寸是正方形,你只需指定一个数字
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


##########################################################################
# 让我们分解一下这个模型中卷积层的工作原理。从 ``conv1`` 开始:
# 
# -  LeNet5 旨在接受 1x32x32 的黑白图像。**卷积层构造函数的第一个参数是输入通道数。**这里是 1。
#    如果我们构建这个模型来查看 3 色彩通道,它将是 3。
# -  卷积层就像一个扫描图像的窗口,寻找它能识别的模式。这些模式被称为 *特征*,
#    卷积层的一个参数是我们希望它学习的特征数量。**构造函数的第二个参数是输出特征的数量。** 
#    这里,我们要求我们的层学习 6 个特征。
# -  就在上面,我将卷积层比作一个窗口 - 但是窗口有多大? **第三个参数是窗口或内核大小。** 
#    这里,数字 "5" 意味着我们选择了一个 5x5 的内核。
#    (如果你希望内核的高度与宽度不同,你可以为此参数指定一个元组 - 例如 ``(3, 5)`` 来获得一个 3x5 的卷积核。)
# 
# 卷积层的输出是一个 *激活映射* - 输入张量中特征存在的空间表示。``conv1`` 将给我们一个 6x28x28 的输出张量;
# 6 是特征数,28 是映射的高度和宽度。(28 来自于当在 32 像素行上扫描 5 像素窗口时,只有 28 个有效位置的事实。)
# 
# 然后,我们将卷积的输出通过 ReLU 激活函数(稍后将讨论激活函数),然后通过最大池化层。
# 最大池化层将激活映射中彼此靠近的特征组合在一起。它通过减小张量来实现这一点,
# 将输出中每个 2x2 组的单元格合并为一个单元格,并将该单元格的值分配为组成它的 4 个单元格中的最大值。
# 这给了我们一个较低分辨率的激活映射,尺寸为 6x14x14。
# 
# 我们的下一个卷积层 ``conv2`` 期望 6 个输入通道(对应于第一层寻找的 6 个特征),有 16 个输出通道,
# 并且内核大小为 3x3。它输出一个 16x12x12 的激活映射,再次通过最大池化层减小到 16x6x6。
# 在将此输出传递给线性层之前,它被重新整形为一个 16 * 6 * 6 = 576 元素向量,供下一层使用。
# 
# 有针对 1D、2D 和 3D 张量的卷积层。卷积层构造函数还有许多其他可选参数,
# 包括步长长度(例如,只扫描每第二个或每第三个位置)、填充(因此你可以扫描到输入的边缘)等等。
# 更多信息请参见 `文档 <https://pytorch.org/docs/stable/nn.html#convolution-layers>`__。
# 
# 循环层
# ~~~~~~~~~~~~~~~~
# 
# *循环神经网络* (或 *RNN*)用于序列数据 - 从科学仪器的时间序列测量到自然语言句子再到 DNA 核苷酸。
# RNN 通过维护一个 *隐藏状态* 来实现这一点,该隐藏状态充当一种记忆,记录到目前为止它在序列中看到的内容。
# 
# RNN 层的内部结构 - 或其变体 LSTM(长短期记忆)和 GRU(门控循环单元) - 相当复杂,超出了本视频的范围,
# 但我们将向你展示基于 LSTM 的词性标注器(一种分类器,用于告诉你一个单词是名词、动词等)的样子:
# 

class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # LSTM 接受词嵌入作为输入,并输出维度为 hidden_dim 的隐藏状态
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # 将从隐藏状态空间映射到标记空间的线性层
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


########################################################################
# 构造函数有四个参数:
#
# - ``vocab_size`` 是输入词汇表中单词的数量。每个单词是一个 ``vocab_size`` 维的一热向量(或单位向量)。
#
# - ``tagset_size`` 是输出标签集的大小。
#
# - ``embedding_dim`` 是词汇的*嵌入*空间的大小。嵌入将词汇映射到一个低维空间,在该空间中,意义相似的单词彼此接近。
#
# - ``hidden_dim`` 是 LSTM 的记忆大小。
#
# 输入将是一个句子,单词表示为一热向量的索引。嵌入层将把这些映射到一个 ``embedding_dim`` 维的空间。
# LSTM 接收这个嵌入序列并对其进行迭代,产生一个长度为 ``hidden_dim`` 的输出向量。最后的线性层充当分类器;
# 将最后一层的输出通过 ``log_softmax()`` 转换为一组归一化的估计概率,表示给定单词映射到给定标签的概率。
#
# 如果你想看看这个网络的实际运行情况,可以查看 pytorch.org 上的 `序列模型和 
# LSTM 网络 <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html>`__ 教程。
#
# 转换器
# ~~~~~~~~~~~~
#
# *转换器*是多用途网络,在 NLP 领域取得了最先进的成果,例如 BERT 模型。讨论转换器架构超出了本视频的范围,
# 但是 PyTorch 有一个 ``Transformer`` 类,允许你定义转换器模型的整体参数 - 注意力头的数量、
# 编码器和解码器层的数量、dropout 和激活函数等。(你甚至可以用正确的参数从这个单一类构建 BERT 模型!)
#  ``torch.nn.Transformer`` 类还包含封装单个组件(``TransformerEncoder``、``TransformerDecoder``)
# 和子组件(``TransformerEncoderLayer``、``TransformerDecoderLayer``)的类。
# 详情请查看 `文档 <https://pytorch.org/docs/stable/nn.html#transformer-layers>`__ 中关于转换器类的内容,
# 以及 pytorch.org 上相关的 `教程 <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`__。
#
# 其他层和函数
# --------------------------
#
# 数据操作层
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# 还有其他类型的层执行模型中的重要功能,但它们自身不参与学习过程。
#
# **最大池化**(及其孪生层最小池化)通过组合单元并将输入单元的最大值分配给输出单元来减小张量(我们之前看到过这一点)。
# 例如:
#

my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))


#########################################################################
# 如果你仔细观察上面的值,你会发现最大池化输出中的每个值都是 6x6 输入的每个象限的最大值。
#
# **归一化层**在将一层的输出馈送到另一层之前,重新居中并归一化输出。居中和缩放中间张量有许多有益的效果,
# 例如让你可以使用更高的学习率而不会出现梯度爆炸/消失。
#

my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())



##########################################################################
# 运行上面的单元格,我们为输入张量添加了一个大的缩放因子和偏移量;你应该会看到输入张量的 ``mean()`` 在 15 左右。
# 经过归一化层处理后,你可以看到值变小了,并且围绕着 0 分布 - 事实上,平均值应该非常小(> 1e-8)。
#
# 这是有益的,因为许多激活函数(下面将讨论)在 0 附近具有最强梯度,但有时对于将它们推离 0 很远的输入会遇到梯度消失或爆炸的问题。
# 将数据保持在最陡梯度区域周围将倾向于意味着更快、更好的学习和更高的可行学习率。
#
# **Dropout 层**是一种鼓励模型*稀疏表示*的工具 - 也就是说,推动它在推理时使用较少的数据。
#
# Dropout 层的工作原理是在*训练期间*随机设置输入张量的一部分 - dropout 层在推理时总是关闭的。
# 这迫使模型针对这种掩码或减少的数据集进行学习。例如:
#

my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))


##########################################################################
# 上面,你可以看到 dropout 对示例张量的影响。你可以使用可选的 ``p`` 参数设置单个权重丢弃的概率;
# 如果不设置,默认为 0.5。
#
# 激活函数
# ~~~~~~~~~~~~~~~~~~~~
#
# 激活函数使深度学习成为可能。神经网络实际上是一个程序 - 有许多参数 - 用于*模拟一个数学函数*。
# 如果我们只是重复地将张量与层权重相乘,我们只能模拟*线性函数*;此外,有多层也没有意义,
# 因为整个网络可以简化为单个矩阵乘法。在层之间插入*非线性*激活函数使得深度学习模型能够模拟任何函数,
# 而不仅仅是线性函数。
#
# ``torch.nn.Module`` 有封装所有主要激活函数的对象,包括 ReLU 及其许多变体、Tanh、Hardtanh、sigmoid 等。
# 它还包括其他函数,如 Softmax,这些函数在模型的输出阶段最有用。
#
# 损失函数
# ~~~~~~~~~~~~~~
#
# 损失函数告诉我们模型的预测与正确答案相差多远。PyTorch 包含各种损失函数,包括常见的 MSE(均方误差 = L2 范数)、
# 交叉熵损失和负对数似然损失(对于分类器很有用)等。