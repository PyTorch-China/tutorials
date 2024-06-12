"""
`简介 <introyt1_tutorial.html>`_ ||
**张量** ||
`自动微分 <autogradyt_tutorial.html>`_ ||
`构建模型 <modelsyt_tutorial.html>`_ ||
`TensorBoard支持 <tensorboardyt_tutorial.html>`_ ||
`训练模型 <trainingyt.html>`_ ||
`模型理解 <captumyt.html>`_

PyTorch Tensors 介绍
===============================

跟随下面的视频或在 `youtube <https://www.youtube.com/watch?v=r7QDUPb2dCM>`__ 上观看。

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/r7QDUPb2dCM" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

张量是PyTorch中的中心数据抽象。这个交互式笔记本提供了对 ``torch.Tensor`` 类的深入介绍。

首先,让我们导入PyTorch模块。我们还将添加Python的数学模块，以便于一些示例。

"""

import torch
import math


#########################################################################
# 创建张量
# ----------------
# 
# 创建张量最简单的方法是使用 ``torch.empty()`` 调用:
# 

x = torch.empty(3, 4)
print(type(x))
print(x)


##########################################################################
# 让我们解释下刚才发生的事情:
#
# - 我们使用附加到 ``torch`` 模块的众多工厂方法之一创建了一个张量。
# - 该张量是二维的,有3行4列。
# - 返回对象的类型是 ``torch.Tensor``，这是 ``torch.FloatTensor`` 的别名；
#   默认情况下，PyTorch张量用32位浮点数填充。(更多关于数据类型的内容见下文。)
# - 当打印你的张量时，你可能会看到一些随机的值。``torch.empty()`` 调用为张量分配内存，
#   但不会用任何值初始化它 - 所以你看到的是分配时内存中的任何值。
#
# 关于张量及其维数和术语的简要说明:
#
# - 你有时会看到一维张量被称为 *向量*。
# - 同样,二维张量通常被称为 *矩阵*。
# - 任何超过两个维度的张量通常都被称为张量。
#
# 大多数情况下,你会希望用一些值初始化你的张量。常见的情况是全零、全一或随机值，
# ``torch`` 模块为所有这些情况提供了工厂方法:
#

zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)


#########################################################################
# 工厂方法都做了你期望的事情 - 我们有一个全零张量、一个全一张量和一个随机值在0到1之间的张量。
#
# 随机张量和种子
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 说到随机张量,你是否注意到在它之前立即调用了 ``torch.manual_seed()``?
# 用随机值初始化张量(如模型的学习权重)是很常见的，但在某些情况下 - 特别是在研究环境中 - 
# 你可能希望对结果的可重复性有一些保证。手动设置随机数生成器的种子就是这样做的方法。让我们仔细看看:
#

torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)


############################################################################
# 你应该看到上面 ``random1`` 和 ``random3`` 包含相同的值,``random2`` 和 ``random4`` 也是如此。
# 手动设置RNG的种子会重置它,因此相同的随机数计算在大多数设置下应该提供相同的结果。
#
# 有关更多信息,请参阅PyTorch关于可重复性的 
# `文档 <https://pytorch.org/docs/stable/notes/randomness.html>`__。
#
# 张量形状
# ~~~~~~~~~~~~~
#
# 当你在两个或多个张量上执行操作时,它们通常需要具有相同的 *形状* - 也就是说，
# 具有相同的维数和每个维度中的相同数量的单元。为此,我们有 ``torch.*_like()`` 方法:
#

x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)


#########################################################################
# 上面代码单元中的第一个新事物是在张量上使用 ``.shape`` 属性。
# 这个属性包含了每个维度张量的范围的列表 - 在我们的例子中，``x`` 是一个三维张量，形状为 2 x 2 x 3。
#
# 在下面,我们调用 ``.empty_like()``，``.zeros_like()``，``.ones_like()`` 和 ``.rand_like()`` 方法。
# 使用 ``.shape`` 属性，我们可以验证每个这些方法都返回一个具有相同维数和范围的张量。
#
# 创建张量的最后一种方式是直接从PyTorch集合中指定其数据:
#

some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)


######################################################################
# 使用 ``torch.tensor()`` 是在你已经有Python元组或列表数据的情况下创建张量的最直接方式。
# 如上所示，嵌套集合会生成多维张量。
#
# .. note::
#      ``torch.tensor()`` 创建数据的副本。
#
# 张量数据类型
# ~~~~~~~~~~~~~~~~~
#
# 设置张量的数据类型有两种方式:
#

a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)


##########################################################################
# 设置张量底层数据类型的最简单方式是在创建时使用可选参数。在上面单元格的第一行中，
# 我们将 ``dtype=torch.int16`` 设置为张量 ``a``。当我们打印 ``a`` 时，
# 我们可以看到它是由 ``1`` 而不是 ``1.`` 填充的 - Python的一个微妙提示，这是一个整数类型而不是浮点数。
#
# 你可能还注意到，打印 ``a`` 时，与我们将 ``dtype`` 保留为默认值(32位浮点数)时不同，
# 打印张量时也指定了其 ``dtype``。
#
# 你可能还注意到,我们从指定张量形状为一系列整数参数，转为将这些参数分组到一个元组中。
# 这不是绝对必要的 - PyTorch会将一系列初始的、未标记的整数参数视为张量形状 - 但是当添加可选参数时，
# 它可以使你的意图更加可读。
#
# 设置数据类型的另一种方式是使用 ``.to()`` 方法。在上面的单元格中，
# 我们以通常的方式创建了一个随机浮点张量 ``b``。接下来,我们通过将 ``b`` 转换为32位整数来创建 ``c``。
# 注意 ``c`` 包含与 ``b`` 相同的值,但被截断为整数。
#
# 可用的数据类型包括:
#
# - ``torch.bool``
# - ``torch.int8``
# - ``torch.uint8``
# - ``torch.int16``
# - ``torch.int32``
# - ``torch.int64``
# - ``torch.half``
# - ``torch.float``
# - ``torch.double``
# - ``torch.bfloat``
#
# 使用PyTorch张量进行数学和逻辑运算
# ---------------------------------
#
# 现在你知道了一些创建张量的方法，那你能对它们做什么呢?
#
# 让我们首先看基本算术运算，以及张量如何与简单的标量交互:
#

ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)


##########################################################################
# 如你所见，张量和标量之间的加法、减法、乘法、除法和指数运算都是在张量的每个元素上分布式进行的。
# 由于这种操作的输出将是一个张量，你可以像通常的运算符优先级规则一样将它们链接在一起，
# 就像我们在创建 ``threes`` 的那一行中所做的那样。
#
# 两个张量之间的类似运算也像你直觉上期望的那样:
#

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)


##########################################################################
# 这里需要注意的是，前面代码单元中的所有张量都具有相同的形状。如果我们尝试在不同形状的张量上执行二元运算会怎样?
# 
# .. note::
#    下面的单元格会抛出一个运行时错误，这是有意的。
#
#    .. code-block:: sh
#
#       a = torch.rand(2, 3)
#       b = torch.rand(3, 2)
#
#       print(a * b)
#


##########################################################################
# 一般情况下，你不能以这种方式对不同形状的张量进行操作，即使在上面的单元格中，张量具有相同数量的元素。
# 
# 简要介绍:张量广播
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# .. note::
#      如果你熟悉NumPy ndarrays中的广播语义,你会发现这里应用的是相同的规则。
# 
# 同形规则的例外是 *张量广播*。这里有一个例子:
# 

rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)


#########################################################################
# 这里的技巧是什么?我们是如何将 2 x 4 张量与 1 x 4 张量相乘的?
#
# 广播是一种在具有相似形状的张量之间执行操作的方式。在上面的例子中，一行四列的张量与两行四列张量的 *两行* 相乘。
#
# 这是深度学习中一个重要的操作。常见的例子是将一批输入张量的学习权重张量相乘，分别对批次中的每个实例应用该操作，
# 并返回一个形状相同的张量 - 就像我们上面的(2,4) * (1,4)示例一样，返回了一个形状为(2,4)的张量。
#
# 广播的规则是:
#
# - 每个张量必须至少有一个维度 - 不允许空张量。
#
# - 比较两个张量的维度大小，*从最后一个到第一个:*
#
#    - 每个维度必须相等，*或*
#
#    - 其中一个维度必须为1，*或*
#
#    - 该维度在其中一个张量中不存在
#
# 当然，相同形状的张量是"可广播"的，正如你之前看到的那样。
#
# 这里有一些符合上述规则并允许广播的情况示例:
#

a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 第3和第2维与a相同，第1维不存在
print(b)

c = a * torch.rand(   3, 1) # 第3维为1，第2维与a相同
print(c)

d = a * torch.rand(   1, 2) # 第3维与a相同，第2维为1
print(d)


#############################################################################
# 仔细观察上面每个张量的值:
#
# - 创建 `b` 的乘法运算是在 `a` 的每一层上广播的。
# - 对于 `c`，该运算在 `a` 的每一层和每一行上都进行了广播 - 每一列3个元素都是相同的。
# - 对于 `d`，我们颠倒了一下 - 现在每一行在层与列之间都是相同的。
#
# 有关广播的更多信息,请参阅PyTorch关于此的
# `文档 <https://pytorch.org/docs/stable/notes/broadcasting.html>`__。
#
# 这里有一些尝试广播但会失败的例子:
#
# .. note::
#    下面的单元格会抛出一个运行时错误，这是有意的。
#
#    .. code-block:: python
#
#       a =     torch.ones(4, 3, 2)
#
#       b = a * torch.rand(4, 3)    # 维度必须从最后到第一个匹配
#
#       c = a * torch.rand(   2, 3) # 第3和第2维都不同
#
#       d = a * torch.rand((0, ))   # 不能与空张量进行广播
#

###########################################################################
# 更多张量数学运算
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# PyTorch 张量有超过三百种可以执行的操作。
# 
# 这里是一些主要操作类别的示例:
# 

# 常用方法
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# 三角函数及其反函数
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# 位运算
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# 比较操作
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  
print(torch.eq(d, e)) # 返回布尔类型张量

# 归约操作:
print('\n归约操作:')
print(torch.max(d))        # 返回单元素张量
print(torch.max(d).item()) # 从返回的张量中提取值
print(torch.mean(d))       # 平均值
print(torch.std(d))        # 标准差
print(torch.prod(d))       # 所有数字的乘积
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # 过滤唯一元素

# 向量和线性代数运算
v1 = torch.tensor([1., 0., 0.])         # x 单位向量
v2 = torch.tensor([0., 1., 0.])         # y 单位向量
m1 = torch.rand(2, 2)                   # 随机矩阵
m2 = torch.tensor([[3., 0.], [0., 3.]]) # 三倍单位矩阵

print('\n向量和矩阵:')
print(torch.cross(v2, v1)) # z 单位向量的负值 (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)
print(m3)                  # m1 的三倍
print(torch.svd(m3))       # 奇异值分解


##################################################################################
# 有关更多详细信息和完整的数学函数清单,请查看
# `文档 <https://pytorch.org/docs/stable/torch.html#math-operations>`__。
#
# 本地修改张量
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 大多数张量的二元运算将返回第三个新张量。当我们说 `c = a * b` (其中 `a` 和 `b` 是张量)时,
# 新张量 `c` 将占用与其他张量不同的内存区域。
#
# 但是,有时您可能希望就地修改张量 - 例如，如果您正在执行元素wise计算,可以丢弃中间值。
# 为此，大多数数学函数都有一个带有附加下划线 (`_`) 的版本，它将就地修改张量。
#
# 例如:
#

a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # 此操作在内存中创建新张量
print(a)              # a 未更改

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # 注意下划线
print(b)              # b 被修改

#######################################################################
# 对于算术运算,有一些函数的行为类似:


a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)


##########################################################################
# 注意,这些就地算术函数是 `torch.Tensor` 对象上的方法，
# 而不是像许多其他函数(例如 `torch.sin()`)那样附加到 `torch` 模块上。
# 正如你从 `a.add_(b)` 中看到的，*被调用的张量是就地改变的那个*。
#
# 还有另一种选择，可以将计算结果放在一个已经分配的张量中。我们到目前为止看到的许多方法和函数
#  - 包括创建方法! - 都有一个 `out` 参数，让你指定一个张量来接收输出。
# 如果 `out` 张量的形状和 `dtype` 正确，这可以在不分配新内存的情况下发生:
#

a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)                # c 的内容已经改变

assert c is d           # 测试 c 和 d 是同一个对象,而不只是包含相等的值
assert id(c) == old_id  # 确保我们的新 c 是旧 c 的同一个对象

torch.rand(2, 2, out=c) # 对于创建也可以!
print(c)                # c 又一次改变
assert id(c) == old_id  # 仍然是同一个对象!

##########################################################################
# 复制张量
# ---------------
#
# 与 Python 中的任何对象一样，将张量赋值给变量会使该变量成为张量的 *标签*，而不会复制它。例如:
#

a = torch.ones(2, 2)
b = a

a[0][1] = 561  # 我们改变 a...
print(b)       # ...b 也被改变了

######################################################################
# 但是,如果你想要一个单独的数据副本来处理呢? 这时就可以使用 `clone()` 方法:
#


a = torch.ones(2, 2)
b = a.clone()

assert b is not a      # 内存中的不同对象...
print(torch.eq(a, b))  # ...但仍然具有相同的内容!

a[0][1] = 561          # a 改变了...
print(b)               # ...但 b 仍然是全 1


#########################################################################
# **使用 `clone()` 时,有一个重要的事情需要注意。**
# 如果你的源张量启用了自动求导,那么克隆张量也会启用自动求导。
# **这将在关于自动求导的视频中更深入地介绍**， 但如果你想了解细节的简单版本,请继续阅读。
#
# *在许多情况下,这正是你所需要的。*例如,如果你的模型在其 `forward()` 方法中有多个计算路径，
# 并且 *原始张量和它的克隆* 都会影响模型的输出，那么为了启用模型学习，你希望两个张量都启用自动求导。
# 如果你的源张量启用了自动求导(通常如果它是一组学习权重或源自涉及权重的计算)，那么你就会得到所需的结果。
#
# 另一方面，如果你正在进行一个计算。其中 *原始张量和它的克隆* 都不需要跟踪梯度，那么只要源张量关闭了自动求导，你就可以继续了。
#
# *还有第三种情况:* 假设你在模型的 `forward()` 函数中执行一个计算，默认情况下所有内容的梯度都打开，
# 但你想在中间提取一些值来生成一些指标。在这种情况下，你 *不希望* 克隆的源张量副本跟踪梯度
#  - 关闭自动求导的历史记录跟踪可以提高性能。为此，你可以在源张量上使用 `.detach()` 方法:
#

a = torch.rand(2, 2, requires_grad=True) # 打开自动求导
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)


#########################################################################
# 此处发生了什么?
#
# -  我们创建了 ``a`` 并将 ``requires_grad=True`` 打开。**我们还没有介绍这个可选参数，
#    但将在关于自动求导的单元中介绍。**
# -  当我们打印 ``a`` 时,它告诉我们属性 ``requires_grad=True`` - 这意味着自动求导和计算历史跟踪已打开。
# -  我们克隆 ``a`` 并将其标记为 ``b``。当我们打印 ``b`` 时，我们可以看到它正在跟踪其计算历史 - 它继承了 ``a`` 的自动求导设置，
#    并添加到了计算历史中。
# -  我们克隆 ``a`` 到 ``c``,但首先调用 ``detach()``。
# -  打印 `c`，我们看不到任何计算历史，也没有 `requires_grad=True`。
#
# ``detach()`` 方法*将张量与其计算历史分离。*它说,"无论接下来发生什么，都像自动求导关闭时那样进行。
# "它这样做*并不会改变 ``a``* - 你可以看到,当我们在最后再次打印 `a` 时，它保留了其 ``requires_grad=True`` 属性。
#
# 移动到 GPU
# -------------
#
# PyTorch 的主要优势之一是在 CUDA 兼容的 Nvidia GPU 上有强大的加速能力。
# ("CUDA"代表*Compute Unified Device Architecture*,这是 Nvidia 的并行计算平台。)
# 到目前为止，我们所做的一切都是在 CPU 上。我们如何移动到更快的硬件上呢?
#
# 首先,我们应该使用 `is_available()` 方法检查是否有 GPU 可用。
#
# .. note::
#      如果你没有安装 CUDA 兼容的 GPU 和 CUDA 驱动程序，本节中的可执行单元格将不会执行任何 GPU 相关的代码。
#

if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')


##########################################################################
# Once we’ve determined that one or more GPUs is available, we need to put
# our data someplace where the GPU can see it. Your CPU does computation
# on data in your computer’s RAM. Your GPU has dedicated memory attached
# to it. Whenever you want to perform a computation on a device, you must
# move *all* the data needed for that computation to memory accessible by
# that device. (Colloquially, “moving the data to memory accessible by the
# GPU” is shorted to, “moving the data to the GPU”.)
# 
# There are multiple ways to get your data onto your target device. You
# may do it at creation time:
# 

if torch.cuda.is_available():
    gpu_rand = torch.rand(2, 2, device='cuda')
    print(gpu_rand)
else:
    print('Sorry, CPU only.')


##########################################################################
# By default, new tensors are created on the CPU, so we have to specify
# when we want to create our tensor on the GPU with the optional
# ``device`` argument. You can see when we print the new tensor, PyTorch
# informs us which device it’s on (if it’s not on CPU).
# 
# You can query the number of GPUs with ``torch.cuda.device_count()``. If
# you have more than one GPU, you can specify them by index:
# ``device='cuda:0'``, ``device='cuda:1'``, etc.
# 
# As a coding practice, specifying our devices everywhere with string
# constants is pretty fragile. In an ideal world, your code would perform
# robustly whether you’re on CPU or GPU hardware. You can do this by
# creating a device handle that can be passed to your tensors instead of a
# string:
# 

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)


#########################################################################
# If you have an existing tensor living on one device, you can move it to
# another with the ``to()`` method. The following line of code creates a
# tensor on CPU, and moves it to whichever device handle you acquired in
# the previous cell.
# 

y = torch.rand(2, 2)
y = y.to(my_device)


##########################################################################
# It is important to know that in order to do computation involving two or
# more tensors, *all of the tensors must be on the same device*. The
# following code will throw a runtime error, regardless of whether you
# have a GPU device available:
# 
# .. code-block:: python
# 
#    x = torch.rand(2, 2)
#    y = torch.rand(2, 2, device='gpu')
#    z = x + y  # exception will be thrown
# 


###########################################################################
# Manipulating Tensor Shapes
# --------------------------
# 
# Sometimes, you’ll need to change the shape of your tensor. Below, we’ll
# look at a few common cases, and how to handle them.
# 
# Changing the Number of Dimensions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# One case where you might need to change the number of dimensions is
# passing a single instance of input to your model. PyTorch models
# generally expect *batches* of input.
# 
# For example, imagine having a model that works on 3 x 226 x 226 images -
# a 226-pixel square with 3 color channels. When you load and transform
# it, you’ll get a tensor of shape ``(3, 226, 226)``. Your model, though,
# is expecting input of shape ``(N, 3, 226, 226)``, where ``N`` is the
# number of images in the batch. So how do you make a batch of one?
# 

a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)


##########################################################################
# The ``unsqueeze()`` method adds a dimension of extent 1.
# ``unsqueeze(0)`` adds it as a new zeroth dimension - now you have a
# batch of one!
# 
# So if that’s *un*\ squeezing? What do we mean by squeezing? We’re taking
# advantage of the fact that any dimension of extent 1 *does not* change
# the number of elements in the tensor.
# 

c = torch.rand(1, 1, 1, 1, 1)
print(c)


##########################################################################
# Continuing the example above, let’s say the model’s output is a
# 20-element vector for each input. You would then expect the output to
# have shape ``(N, 20)``, where ``N`` is the number of instances in the
# input batch. That means that for our single-input batch, we’ll get an
# output of shape ``(1, 20)``.
# 
# What if you want to do some *non-batched* computation with that output -
# something that’s just expecting a 20-element vector?
# 

a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)


#########################################################################
# You can see from the shapes that our 2-dimensional tensor is now
# 1-dimensional, and if you look closely at the output of the cell above
# you’ll see that printing ``a`` shows an “extra” set of square brackets
# ``[]`` due to having an extra dimension.
# 
# You may only ``squeeze()`` dimensions of extent 1. See above where we
# try to squeeze a dimension of size 2 in ``c``, and get back the same
# shape we started with. Calls to ``squeeze()`` and ``unsqueeze()`` can
# only act on dimensions of extent 1 because to do otherwise would change
# the number of elements in the tensor.
# 
# Another place you might use ``unsqueeze()`` is to ease broadcasting.
# Recall the example above where we had the following code:
# 
# .. code-block:: python
# 
#    a = torch.ones(4, 3, 2)
# 
#    c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
#    print(c)
# 
# The net effect of that was to broadcast the operation over dimensions 0
# and 2, causing the random, 3 x 1 tensor to be multiplied element-wise by
# every 3-element column in ``a``.
# 
# What if the random vector had just been 3-element vector? We’d lose the
# ability to do the broadcast, because the final dimensions would not
# match up according to the broadcasting rules. ``unsqueeze()`` comes to
# the rescue:
# 

a = torch.ones(4, 3, 2)
b = torch.rand(   3)     # trying to multiply a * b will give a runtime error
c = b.unsqueeze(1)       # change to a 2-dimensional tensor, adding new dim at the end
print(c.shape)
print(a * c)             # broadcasting works again!


######################################################################
# The ``squeeze()`` and ``unsqueeze()`` methods also have in-place
# versions, ``squeeze_()`` and ``unsqueeze_()``:
# 

batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)


##########################################################################
# Sometimes you’ll want to change the shape of a tensor more radically,
# while still preserving the number of elements and their contents. One
# case where this happens is at the interface between a convolutional
# layer of a model and a linear layer of the model - this is common in
# image classification models. A convolution kernel will yield an output
# tensor of shape *features x width x height,* but the following linear
# layer expects a 1-dimensional input. ``reshape()`` will do this for you,
# provided that the dimensions you request yield the same number of
# elements as the input tensor has:
# 

output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)


###############################################################################
# .. note::
#      The ``(6 * 20 * 20,)`` argument in the final line of the cell
#      above is because PyTorch expects a **tuple** when specifying a
#      tensor shape - but when the shape is the first argument of a method, it
#      lets us cheat and just use a series of integers. Here, we had to add the
#      parentheses and comma to convince the method that this is really a
#      one-element tuple.
# 
# When it can, ``reshape()`` will return a *view* on the tensor to be
# changed - that is, a separate tensor object looking at the same
# underlying region of memory. *This is important:* That means any change
# made to the source tensor will be reflected in the view on that tensor,
# unless you ``clone()`` it.
# 
# There *are* conditions, beyond the scope of this introduction, where
# ``reshape()`` has to return a tensor carrying a copy of the data. For
# more information, see the
# `docs <https://pytorch.org/docs/stable/torch.html#torch.reshape>`__.
# 


#######################################################################
# NumPy Bridge
# ------------
# 
# In the section above on broadcasting, it was mentioned that PyTorch’s
# broadcast semantics are compatible with NumPy’s - but the kinship
# between PyTorch and NumPy goes even deeper than that.
# 
# If you have existing ML or scientific code with data stored in NumPy
# ndarrays, you may wish to express that same data as PyTorch tensors,
# whether to take advantage of PyTorch’s GPU acceleration, or its
# efficient abstractions for building ML models. It’s easy to switch
# between ndarrays and PyTorch tensors:
# 

import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)


##########################################################################
# PyTorch creates a tensor of the same shape and containing the same data
# as the NumPy array, going so far as to keep NumPy’s default 64-bit float
# data type.
# 
# The conversion can just as easily go the other way:
# 

pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)


##########################################################################
# It is important to know that these converted objects are using *the same
# underlying memory* as their source objects, meaning that changes to one
# are reflected in the other:
# 

numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
