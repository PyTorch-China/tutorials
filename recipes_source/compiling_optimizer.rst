(beta) 使用 torch.compile 编译优化器
==========================================================================================

**作者:** `Michael Lazos <https://github.com/mlazos>`_

优化器是训练任何深度学习模型的关键算法。由于它负责更新每个模型参数,因此对于大型模型,它往往会成为训练性能的瓶颈。
在本教程中,我们将在优化器使用 ``torch.compile`` ,提升在 GPU 上的性能。

.. note::

   本教程需要 PyTorch 2.2.0 或更高版本。

模型设置
~~~~~~~~~~~~~~~~~~~~~
对于本例,我们将使用一个简单的线性层序列。由于我们只是对优化器进行基准测试,所选择的模型并不重要,
因为优化器的性能与函数参数数量有关。

根据您使用的机器不同,结果可能会有所不同。

.. code-block:: python

   import torch
   
   model = torch.nn.Sequential(
       *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
   )
   input = torch.rand(1024, device="cuda")
   output = model(input)
   output.sum().backward()

设置和运行优化器基准测试
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
在本例中,我们将使用 Adam 优化器,并创建一个辅助函数来将 step() 包装在 ``torch.compile()`` 中。

.. note::
   
   ``torch.compile`` 仅支持device_capability >= 7.0 的 CUDA 设备

.. code-block:: python

   # 如果我们在不支持 torch.compile 的设备上,则干净地退出
   if torch.cuda.get_device_capability() < (7, 0):
       print("Exiting because torch.compile is not supported on this device.")
       import sys
       sys.exit(0)


   opt = torch.optim.Adam(model.parameters(), lr=0.01)


   @torch.compile(fullgraph=False)
   def fn():
       opt.step()
   
   
   # 让我们定义一个有用的基准测试函数:
   import torch.utils.benchmark as benchmark
   
   
   def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
       t0 = benchmark.Timer(
           stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
       )
       return t0.blocked_autorange().mean * 1e6


   # 预热运行以编译函数
   for _ in range(5):
       fn()
   
   eager_runtime = benchmark_torch_function_in_microseconds(opt.step)
   compiled_runtime = benchmark_torch_function_in_microseconds(fn)
   
   assert eager_runtime > compiled_runtime
   
   print(f"eager runtime: {eager_runtime}us")
   print(f"compiled runtime: {compiled_runtime}us")

示例结果:

* Eager runtime: 747.2437149845064us
* Compiled runtime: 392.07384741178us

另请参阅
~~~~~~~~~

* 有关深入的技术概述,请参阅
`使用 PT2 编译优化器 <https://dev-discuss.pytorch.org/t/compiling-the-optimizer-with-pt2/1669>`__
