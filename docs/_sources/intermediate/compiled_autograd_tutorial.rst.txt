编译的 Autograd：捕获更大的反向图用于 ``torch.compile``
==========================================================================
**作者:** `Simon Fan <https://github.com/xmfan>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` 你将学到什么
       :class-card: card-prerequisites

       * 编译的 autograd 如何与 ``torch.compile`` 交互
       * 如何使用编译的 autograd API
       * 如何使用 ``TORCH_LOGS`` 检查日志

    .. grid-item-card:: :octicon:`list-unordered;1em;` 前提条件
       :class-card: card-prerequisites

       * PyTorch 2.4
       * 完成 `Introduction to torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_
       * 阅读 `Get Started with PyTorch 2.x <https://pytorch.org/get-started/pytorch-2.0/>`_ 中的 TorchDynamo 和 AOTAutograd 部分

概述
--------
编译的 Autograd 是 PyTorch 2.4 中引入的 ``torch.compile`` 扩展，允许捕获更大的反向图。

虽然 ``torch.compile`` 确实捕获了反向图，但它是 **部分** 捕获的。AOTAutograd 组件提前捕获反向图，但有一定的限制：

* 前向图中的图中断会导致反向图中的图中断
* `反向钩子 <https://pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution>`_ 未被捕获

编译的 Autograd 通过直接与 autograd 引擎集成，解决了这些限制，允许它在运行时捕获完整的反向图。具有这些特征的模型应尝试编译的 Autograd，并可能观察到更好的性能。

然而，编译的 Autograd 也引入了自身的限制：

* 在反向开始时增加了缓存查找的运行时开销
* 由于更大的捕获，更容易在 dynamo 中重新编译和图中断

.. note:: 编译的 Autograd 正在积极开发中，尚不兼容所有现有的 PyTorch 功能。有关特定功能的最新状态，请参阅 `Compiled Autograd Landing Page <https://docs.google.com/document/d/11VucFBEewzqgkABIjebZIzMvrXr3BtcY1aGKpX61pJY>`_。

设置
-----
在本教程中，我们将基于这个简单的神经网络模型进行示例。它接受一个 10 维的输入向量，通过一个线性层处理，并输出另一个 10 维的向量。

.. code:: python

   import torch

   class Model(torch.nn.Module):
      def __init__(self):
         super().__init__()
         self.linear = torch.nn.Linear(10, 10)

      def forward(self, x):
         return self.linear(x)

基本用法
------------
在调用 ``torch.compile`` API 之前，请确保将 ``torch._dynamo.config.compiled_autograd`` 设置为 ``True``：

.. code:: python

   model = Model()
   x = torch.randn(10)

   torch._dynamo.config.compiled_autograd = True
   @torch.compile
   def train(model, x):
      loss = model(x).sum()
      loss.backward()

   train(model, x)

在上面的代码中，我们创建了一个 ``Model`` 类的实例，并使用 ``torch.randn(10)`` 生成一个随机的 10 维张量 ``x``。
我们定义了训练循环函数 ``train`` 并用 @torch.compile 装饰它以优化其执行。
当调用 ``train(model, x)`` 时：

* 由于此调用被 ``@torch.compile`` 装饰，Python 解释器调用 Dynamo。
* Dynamo 拦截 Python 字节码，模拟其执行并将操作记录到图中。
* ``AOTDispatcher`` 禁用钩子并调用 autograd 引擎来计算 ``model.linear.weight`` 和 ``model.linear.bias`` 的梯度，并将操作记录到图中。使用 ``torch.autograd.Function``，AOTDispatcher 重写了 ``train`` 的前向和反向实现。
* Inductor 生成一个函数，对应于 AOTDispatcher 前向和反向的优化实现。
* Dynamo 设置优化函数，以便由 Python 解释器接下来评估。
* Python 解释器执行优化函数，该函数执行 ``loss = model(x).sum()``。
* Python 解释器执行 ``loss.backward()``，调用 autograd 引擎，由于我们设置了 ``torch._dynamo.config.compiled_autograd = True``，它会路由到编译的 Autograd 引擎。
* 编译的 Autograd 计算 ``model.linear.weight`` 和 ``model.linear.bias`` 的梯度，并将操作记录到图中，包括它遇到的任何钩子。在此过程中，它将记录 AOTDispatcher 先前重写的反向。然后，编译的 Autograd 生成一个新函数，该函数对应于 ``loss.backward()`` 的完全跟踪实现，并在推理模式下使用 ``torch.compile`` 执行它。
* 相同的步骤递归应用于编译的 Autograd 图，但这次 AOTDispatcher 不需要对图进行分区。

检查编译的 autograd 日志
-------------------------------------
使用 ``TORCH_LOGS`` 环境变量运行脚本：

* 仅打印编译的 autograd 图，使用 ``TORCH_LOGS="compiled_autograd" python example.py``
* 打印包含更多张量元数据和重新编译原因的图，以性能为代价，使用 ``TORCH_LOGS="compiled_autograd_verbose" python example.py``

重新运行上面的代码片段，编译的 autograd 图现在应该记录到 ``stderr``。某些图节点的名称将以 ``aot0_`` 为前缀，
这些对应于先前在 AOTAutograd 反向图 0 中提前编译的节点，例如，``aot0_view_2`` 对应于 id=0 的 AOT 反向图中的 ``view_2``。

在下图中，红色框封装了由 ``torch.compile`` 捕获的 AOT 反向图，而没有编译的 Autograd。

.. image:: ../_static/img/compiled_autograd/entire_verbose_log.png

.. note:: This is the graph on which we will call ``torch.compile``, **NOT** the optimized graph. Compiled Autograd essentially generates some unoptimized Python code to represent the entire C++ autograd execution.

编译前向和后向传递使用不同的标志
-------------------------------------------------------------
您可以为两次编译使用不同的编译器配置，例如，即使前向传递中有图断裂，后向传递也可以是完整图。

.. code:: python

   def train(model, x):
       model = torch.compile(model)
       loss = model(x).sum()
       torch._dynamo.config.compiled_autograd = True
       torch.compile(lambda: loss.backward(), fullgraph=True)()

或者您可以使用上下文管理器，它将应用于其范围内的所有自动梯度调用。

.. code:: python

   def train(model, x):
      model = torch.compile(model)
      loss = model(x).sum()
      with torch._dynamo.compiled_autograd.enable(torch.compile(fullgraph=True)):
         loss.backward()


编译的自动梯度解决了AOTAutograd的某些限制
--------------------------------------------------------------
1. 前向传递中的图断裂不再必然导致后向传递中的图断裂：

.. code:: python

   @torch.compile(backend="aot_eager")
   def fn(x):
      # 1st graph
      temp = x + 10
      torch._dynamo.graph_break()
      # 2nd graph
      temp = temp + 10
      torch._dynamo.graph_break()
      # 3rd graph
      return temp.sum()

   x = torch.randn(10, 10, requires_grad=True)
   torch._dynamo.utils.counters.clear()
   loss = fn(x)

   # 1. base torch.compile 
   loss.backward(retain_graph=True)
   assert(torch._dynamo.utils.counters["stats"]["unique_graphs"] == 3)
   torch._dynamo.utils.counters.clear()

   # 2. torch.compile with compiled autograd
   with torch._dynamo.compiled_autograd.enable(torch.compile(backend="aot_eager")):
      loss.backward()

   # single graph for the backward
   assert(torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1)

在第一个 torch.compile 情况下，由于编译函数 fn 中的2次图断裂，生成了3个后向图。 而在第二个使用编译的自动梯度的 torch.compile 情况下，尽管有图断裂，仍然跟踪了完整的后向图。

.. note:: 在跟踪由编译的自动梯度捕获的后向钩子时，Dynamo仍然可能会图断裂。

2. 现在可以捕获后向hooks

.. code:: python

   @torch.compile(backend="aot_eager")
   def fn(x):
      return x.sum()

   x = torch.randn(10, 10, requires_grad=True)
   x.register_hook(lambda grad: grad+10)
   loss = fn(x)

   with torch._dynamo.compiled_autograd.enable(torch.compile(backend="aot_eager")):
      loss.backward()

图中应该有一个 ``call_hook`` 节点，dynamo 稍后会将其内联到以下内容中：


.. image:: ../_static/img/compiled_autograd/call_hook_node.png

编译自动梯度的常见重新编译原因
--------------------------------------------------
1. 由于损失值的自动梯度结构发生变化：

.. code:: python

   torch._dynamo.config.compiled_autograd = True
   x = torch.randn(10, requires_grad=True)
   for op in [torch.add, torch.sub, torch.mul, torch.div]:
      loss = op(x, x).sum()
      torch.compile(lambda: loss.backward(), backend="eager")()

在上面的例子中，我们在每次迭代中调用不同的操作符，导致 ``loss`` 每次跟踪不同的自动梯度历史。您应该会看到一些重新编译消息： **由于新的自动梯度节点导致缓存未命中**。

.. image:: ../_static/img/compiled_autograd/recompile_due_to_node.png

2. 由于张量形状发生变化：

.. code:: python

   torch._dynamo.config.compiled_autograd = True
   for i in [10, 100, 10]:
      x = torch.randn(i, i, requires_grad=True)
      loss = x.sum()
      torch.compile(lambda: loss.backward(), backend="eager")()

在上面的例子中， ``x`` 的形状发生变化，编译的自动梯度将在第一次变化后将 ``x`` 标记为动态形状张量。您应该会看到重新编译消息： **由于形状变化导致缓存未命中**。


.. image:: ../_static/img/compiled_autograd/recompile_due_to_dynamic.png

结论
----------
在本教程中，我们介绍了 ``torch.compile`` 与编译的自动梯度的高级生态系统，编译的自动梯度的基础知识以及一些常见的重新编译原因。请关注 `dev-discuss <https://dev-discuss.pytorch.org/>_` 进行深入讨论。