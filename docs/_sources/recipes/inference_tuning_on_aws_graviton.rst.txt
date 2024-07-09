

(Beta) PyTorch在AWS Graviton处理器上的推理性能优化
======================================================================

**作者**: `Sunita Nadampalli <https://github.com/snadampal>`_

`AWS Graviton <https://aws.amazon.com/ec2/graviton/>`_ 是一系列由AWS设计的基于ARM的处理器。AWS Graviton3处理器针对机器学习(ML)工作负载进行了优化,包括支持 ``bfloat16``、可扩展向量扩展(SVE)以及比Graviton2高两倍的单指令多数据(SIMD)带宽。

PyTorch为机器学习算子(如卷积、矩阵乘法、relu等)提供了原生参考ATen内核。这些算子可以通过来自基本线性代数(BLAS)库的特定于平台的内核实现进行加速。在AWS Graviton CPU上,MKLDNN与Arm Compute Library (`ACL <https://github.com/ARM-software/ComputeLibrary>`_) 和 `OpenBLAS <https://github.com/OpenMathLib/OpenBLAS>`_ 库为一部分算子提供了优化实现。从PyTorch 2.0版本开始,这两个库都集成到了PyTorch中。

在本教程中,我们将介绍如何通过 ``bfloat16`` 内核和正确的后端选择,在AWS Graviton3 CPU (`AWS c7g实例 <https://aws.amazon.com/ec2/instance-types/c7g/>`_) 上实现线性层神经网络的最佳推理性能。

内容
--------
1. 基本用法
2. 使用Bfloat16快速数学内核加速推理
3. 对于较小的批次维度,使用OpenBLAS提高推理性能
4. 使用Linux透明大页优化内存分配开销
5. 总结

.. note::
   要成功运行本教程并重现下面显示的加速数字,您需要来自Graviton3系列(``c7g/r7g/m7g``)的硬件实例。对于本教程,我们使用了 `c7g.xl (4vcpu)实例 <https://aws.amazon.com/ec2/instance-types/c7g/>`_ 。

基本用法
---------------

从PyTorch 2.0版本开始,PyTorch原生支持AWS Graviton3优化。
更多详细信息请参阅此 `博客 <https://pytorch.org/blog/optimized-pytorch-w-graviton/>`_。

1. 运行以下命令安装PyTorch:

   .. code-block::

      python3 -m pip install torch

2. 我们将从导入所需的依赖项并定义将在其上运行的设备开始:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.profiler import profile, record_function, ProfilerActivity

    # AWS Graviton3 cpu
    device = ("cpu")
    print(f"Using {device} device")


3. 鉴于线性层是许多神经网络(包括Transformer)的核心,我们在此演示中使用线性层。我们通过子类化 ``nn.Module`` 并在 ``__init__`` 中初始化层来定义我们的神经网络。我们使用典型的大型语言模型参数构建网络,以匹配真实世界场景:

.. code-block:: python

  class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 11008),
            nn.ReLU(),
            nn.Linear(11008, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

4. 让我们创建一个 ``MyNeuralNetwork`` 的实例,并将其移动到设备上:

.. code-block:: python

    model = MyNeuralNetwork().to(device)
    print(model)

接下来,让我们通过将它们传递给 ``nn.Softmax`` 模块的实例来获取预测概率:

.. code-block:: python

    X = torch.rand(1, 64, 64, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

输出:

.. code-block::

    Predicted class: tensor([2])

我们已验证了网络功能。接下来,我们将分析性能。让我们检查两种不同的情况:小批次维度和大批次维度。

**情况1:** 较大的批次维度,例如256:

.. code-block:: python

    # 首先进行预热,并循环多次以获得足够的执行时间

    X = torch.rand(256, 64, 64, device=device)

    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #Warmup
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


使用默认PyTorch配置时的分析器输出如下:

.. table::
   :widths: auto

   ======================  ============   ===========  =============  ===========  ============  ============
                  Name      Self CPU %      Self CPU    CPU total %    CPU total   CPU time avg    # of Calls
   ======================  ============   ===========  =============  ===========  ============  ============
           aten::addmm        97.61%         15.813s        98.61%       15.977s      53.255ms           300
       aten::clamp_min         1.09%       177.032ms         1.09%     177.032ms     885.160us           200
            aten::copy         1.00%       162.054ms         1.00%     162.054ms     540.180us           300
     mymodel_inference         0.22%        35.738ms       100.00%       16.201s       16.201s             1
          aten::linear         0.02%         2.955ms        98.66%       15.985s      53.282ms           300
               aten::t         0.01%         2.421ms         0.03%       5.043ms      16.810us           300
            aten::relu         0.01%         2.356ms         1.11%     179.388ms     896.940us           200
   ======================  ============   ===========  =============  ===========  ============  ============

**Self CPU time total:** 16.201s


使用 ``bfloat16`` Fast Math Kernels加速推理
----------------------------------------------------------

AWS Graviton3处理器支持 `bfloat16 MMLA指令 <https://developer.arm.com/documentation/ddi0596/2020-12/SVE-Instructions/BFMMLA--BFloat16-floating-point-matrix-multiply-accumulate->`_。Arm Compute Library (`ACL <https://github.com/ARM-software/ComputeLibrary>`_) 为AWS Graviton处理器提供了优化的 ``bfloat16`` 通用矩阵乘法(GEMM)内核,并从PyTorch 2.0版本开始通过MKLDNN后端集成到PyTorch中。可以使用快速数学GEMM内核优化推理性能。默认情况下不启用快速数学模式,因为这些内核以 ``bfloat16`` 精度而不是 ``float`` 执行GEMM,因此会导致模型推理精度略有下降。但是,精度下降在 ``torchbench`` 测试套件中为 ``bfloat16`` 后端定义的 ``余弦相似度`` 阈值范围内,因此对大多数应用程序来说是可以接受的。要启用快速数学GEMM内核,请设置以下环境变量:

.. code-block:: bash

    $ export DNNL_DEFAULT_FPMATH_MODE=BF16


当您运行上述推理脚本时,应该会看到启用MKLDNN快速数学模式后的分析器输出:

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  ============  ============
                  Name      Self CPU %     Self CPU    CPU total %     CPU total   CPU time avg    # of Calls
   ======================  ============  ============  ============  ============  ============  ============
           aten::addmm        95.61%        6.943s        97.10%        7.052s      23.507ms           300
       aten::clamp_min         2.31%     167.653ms         2.31%     167.653ms     838.265us           200
            aten::copy         1.48%     107.593ms         1.48%     107.593ms     358.643us           300
     mymodel_inference         0.43%      31.167ms       100.00%        7.262s        7.262s             1
          aten::linear         0.04%       2.911ms        97.21%        7.060s      23.533ms           300
               aten::t         0.03%       2.414ms         0.07%       4.892ms      16.307us           300
            aten::relu         0.03%       2.281ms         2.34%     169.934ms     849.670us           200
   ======================  ============  ============  ============  ============  ============  ============

**Self CPU time total:** 7.262s


这比默认配置快约 ``2倍 (7.262s vs 16.201s)``。接下来,让我们看看较小批次维度的情况。

**场景 2:** 较小的批量维度，例如 32:

.. code-block:: python

    X = torch.rand(32, 64, 64, device=device)
    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #预热
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


使用 PyTorch 默认配置运行上述脚本时，您应该会看到以下 profiler 输出:

.. table::
   :widths: auto

   ======================  =============  ============  ============  ============  ============  ============
                     名称    自身 CPU %      自身 CPU   CPU 总计 %     CPU 总计   CPU 平均时间    调用次数
   ======================  =============  ============  ============  ============  ============  ============
           aten::addmm        95.51%         5.821s        97.04%        5.914s      19.713ms           300
       aten::clamp_min         2.33%      142.244ms         2.33%     142.244ms     711.220us           200
            aten::copy         1.51%       92.322ms         1.51%      92.322ms     307.740us           300
     mymodel_inference         0.45%       27.713ms       100.00%        6.094s        6.094s             1
          aten::linear         0.04%        2.495ms        97.16%        5.921s      19.736ms           300
               aten::t         0.03%        2.131ms         0.07%       4.441ms      14.803us           300
            aten::relu         0.03%        1.942ms         2.37%     144.186ms     720.930us           200
   ======================  =============  ============  ============  ============  ============  ============

**自身 CPU 总计:** 6.094s


以下是启用 MKLDNN 快速数学模式时的 profiler 输出:

.. code-block:: bash

   $ export DNNL_DEFAULT_FPMATH_MODE=BF16

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  ============   =============
                   名称     自身 CPU %      自身 CPU    CPU 总计 %   CPU 总计    CPU 平均时间    调用次数
   ======================  ============  ============  ============  ============  ============   =============
           aten::addmm        93.31%        3.848s        95.66%        3.944s      13.148ms           300
       aten::clamp_min         3.43%     141.309ms         3.43%     141.309ms     706.545us           200
            aten::copy         2.33%      95.916ms         2.33%      95.916ms     319.720us           300
     mymodel_inference         0.67%      27.431ms       100.00%        4.123s        4.123s             1
          aten::linear         0.06%       2.471ms        95.83%        3.951s      13.170ms           300
               aten::t         0.05%       2.027ms         0.10%       4.243ms      14.143us           300
            aten::relu         0.05%       1.928ms         3.47%     143.237ms     716.185us           200
   ======================  ============  ============  ============  ============  ============   =============

**自身 CPU 总计:** 4.123s

MKLDNN 快速数学模式为较小的批量维度提供了大约 **1.47x (4.123s vs 6.094s)** 的性能提升。
尽管性能提升明显,但整体仍有提升空间。因为来自 oneDNN 和 ACL 后端的运行时开销(权重重排和内核启动时间)
超过了 ACL GEMM 内核对较小批量计算的计算优势。


使用 OpenBLAS 提高较小批量维度的推理性能
----------------------------------------

可以通过将较小的形状从 MKLDNN 卸载到 OpenBLAS 后端来提高较小批量维度的推理性能。我们正在努力为未来版本实现自动化的后端选择,并具有健壮的启发式算法。在实现启发式算法之前,可以通过增加 MKLDNN 后端选择的阈值将较小的形状卸载到 OpenBLAS。在以下示例中,我们使用 ``64`` 作为阈值,因此批量维度为 ``32`` 的输入不会分派到 MKLDNN。相反,它会被分派到 OpenBLAS。

.. code-block:: bash

   $ export TORCH_MKLDNN_MATMUL_MIN_DIM=64

以下是使用 OpenBLAS 后端时的 profiler 输出:

.. table::
   :widths: auto

   ======================  ============  ============  ============  =============  ============  =============
                     名称    自身 CPU %      自身 CPU   CPU 总计 %     CPU 总计   CPU 平均时间    调用次数
   ======================  ============  ============  ============  =============  ============  =============
           aten::addmm        96.25%        1.958s        97.51%        1.984s        6.612ms           300
       aten::clamp_min         1.28%      26.124ms         1.28%      26.124ms      130.620us           200
            aten::copy         1.23%      24.951ms         1.23%      24.951ms       83.170us           300
     mymodel_inference         0.86%      17.423ms       100.00%        2.034s         2.034s             1
          aten::linear         0.08%       1.691ms        97.74%        1.988s        6.628ms           300
               aten::t         0.07%       1.520ms         0.14%       2.945ms        9.817us           300
            aten::relu         0.06%       1.258ms         1.35%      27.382ms      136.910us           200
   ======================  ============  ============  ============  =============  ============  =============

**自身 CPU 总计:** 2.034s


如您所见,切换到 OpenBLAS 将性能提高了一倍 **(2.034s vs 4.123s)** 与默认的 MKLDNN 后端配置相比。
对于更小的批量维度,例如批量维度为 10,这一点更加显著:

.. code-block:: python

    X = torch.rand(10, 64, 64, device=device)
    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #预热
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))


以下是启用 MKLDNN 快速数学模式时的 profiler 输出:

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  =============  =============
                     名称    自身 CPU %      自身 CPU   CPU 总计 %     CPU 总计   CPU 平均时间    调用次数
   ======================  ============  ============  ============  ============  =============  =============
           aten::addmm        87.81%        3.613s        91.90%        3.781s      12.604ms           300
       aten::clamp_min         7.18%     295.437ms         7.18%     295.437ms       1.477ms           200
            aten::copy         4.07%     167.516ms         4.07%     167.516ms     558.387us           300
     mymodel_inference         0.67%      27.708ms       100.00%        4.115s        4.115s             1
          aten::linear         0.06%       2.499ms        92.06%        3.788s      12.627ms           300
               aten::t         0.05%       1.982ms         0.11%       4.385ms      14.617us           300
            aten::relu         0.05%       1.932ms         7.23%     297.369ms       1.487ms           200
   ======================  ============  ============  ============  ============  =============  =============

**自身 CPU 总计:** 4.115s


以下是使用 OpenBLAS 后端时的 profiler 输出:

.. code-block:: bash

   $ export TORCH_MKLDNN_MATMUL_MIN_DIM=64

.. table::
   :widths: auto

   ======================  =============  ============  ============  ============  =============  ============
                   名称     自身 CPU %      自身 CPU     CPU 总计 %   CPU 总计    CPU 平均时间    调用次数
   ======================  =============  ============  ============  ============  =============  ============
           aten::addmm        92.66%        1.179s        95.23%        1.211s         4.038ms           300
       aten::clamp_min         2.83%      36.060ms         2.83%      36.060ms       180.300us           200
            aten::copy         2.52%      32.013ms         2.52%      32.013ms       106.710us           300
     mymodel_inference         1.38%      17.521ms       100.00%        1.272s          1.272s             1
          aten::linear         0.14%       1.750ms        95.60%        1.216s         4.054ms           300
               aten::t         0.12%       1.475ms         0.24%       3.033ms        10.110us           300
            aten::relu         0.10%       1.285ms         2.94%      37.345ms       186.725us           200
   ======================  =============  ============  ============  ============  =============  ============

**自身 CPU 总计:** 1.272s

这里我们观察到通过适当调整后端阈值,**性能提高了3.2倍(1.272s vs 4.115s)**。

使用 Linux Transparent Huge Pages (THP) 优化内存分配开销
------------------------------------------------------

我们还观察到,对于这些较大的网络,张量内存分配占推理延迟的很大一部分。这可以通过从PyTorch C10内存分配器
启用 THP 来优化。目前,该功能默认未启用,因为它会略微增加内存占用。设置以下环境变量以启用它:

.. code-block:: bash

    $ export THP_MEM_ALLOC_ENABLE=1

对于批量维度为 256 且启用 MKLDNN Fast Math 模式:

.. code-block:: python

    X = torch.rand(256, 64, 64, device=device)
    with torch.set_grad_enabled(False):
        for _ in range(50):
            model(X) #预热
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("mymodel_inference"):
                for _ in range(100):
                    model(X)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

启用THP内存分配后,profiler的输出如下:

.. table::
   :widths: auto

   ======================  ============  ============  ============  ============  ==============  ============
                     名称    自身CPU%      自身CPU       CPU总%        CPU总        CPU平均时间     调用次数
   ======================  ============  ============  ============  ============  ==============  ============
           aten::addmm        91.31%        6.115s        94.39%        6.321s      21.069ms           300
       aten::clamp_min         4.82%     322.568ms         4.82%     322.568ms       1.613ms           200
            aten::copy         3.06%     204.602ms         3.06%     204.602ms     682.007us           300
     mymodel_inference         0.61%      40.777ms       100.00%        6.697s        6.697s             1
          aten::linear         0.05%       3.082ms        94.51%        6.329s      21.097ms           300
            aten::relu         0.04%       2.547ms         4.85%     325.115ms       1.626ms           200
   ======================  ============  ============  ============  ============  ==============  ============

**自身CPU总时间:** 6.697s

这比上面测量的已优化的 MKLDNN Fast Math 模式又提高了 **1.08倍或8%(6.697s vs 7.262s)**。

结论
------------

在本教程中,我们介绍了在AWS Graviton3实例上的PyTorch推理,包括基本用法、使用快速数学内核的加速、
比较不同批量维度下不同后端的性能,以及如何使用Linux透明大页面优化张量内存分配延迟。
对于较大的张量形状,建议使用MKLDNN后端和Bfloat16快速数学模式以及THP内存分配;对于较小的张量形状,
建议使用OpenBLAS后端。希望您能尝试一下!
