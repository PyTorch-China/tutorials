"""
性能调优指南
*************************
**作者**: `Szymon Migacz <https://github.com/szmigacz>`_

性能调优指南是一组优化和最佳实践,可以加速PyTorch中深度学习模型的训练和推理。
提出的技术通常只需要更改几行代码,就可以应用于各个领域的广泛深度学习模型。

一般优化
---------------------
"""

###############################################################################
# 启用异步数据加载和数据增强
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
# 支持在单独的工作子进程中异步加载数据和进行数据增强。
# ``DataLoader`` 的默认设置是 ``num_workers=0``，
# 这意味着数据加载是同步的,并在主进程中完成。
# 因此,主训练进程必须等待数据可用才能继续执行。
#
# 设置 ``num_workers > 0`` 可启用异步数据加载,并实现训练和数据加载之间的重叠。
# ``num_workers`` 应根据工作负载、CPU、GPU 和训练数据的位置进行调整。
#
# ``DataLoader`` 接受 ``pin_memory`` 参数,默认为 ``False``。
# 在使用 GPU 时,最好设置 ``pin_memory=True``,这会指示 ``DataLoader`` 使用锁页内存,
# 并启用从主机到 GPU 的更快和异步内存复制。

###############################################################################
# 对于验证或推理,禁用梯度计算
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch 会保存涉及需要梯度的张量的所有操作的中间缓冲区。
# 通常在验证或推理时不需要梯度。
# `torch.no_grad() <https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad>`_
# 上下文管理器可应用于禁用指定代码块内的梯度计算,这可加快执行速度并减少所需内存量。
# `torch.no_grad() <https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad>`_
# 也可以用作函数装饰器。

###############################################################################
# 对于直接后跟批量归一化的卷积,禁用偏置
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `torch.nn.Conv2d() <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_
# 具有 ``bias`` 参数,默认为 ``True``(对于
# `Conv1d <https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d>`_
# 和
# `Conv3d <https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d>`_
# 也是如此)。
#
# 如果 ``nn.Conv2d`` 层直接后跟 ``nn.BatchNorm2d`` 层,
# 则卷积中的偏置是不需要的,请改用
# ``nn.Conv2d(..., bias=False, ....)``。不需要偏置,因为在第一步中 ``BatchNorm`` 会减去均值,
# 这实际上会抵消偏置的效果。
#
# 只要 ``BatchNorm``(或其他归一化层)在与卷积偏置相同的维度上进行归一化,
# 这也适用于1d和3d卷积。
#
# `torchvision <https://github.com/pytorch/vision>`_
# 中可用的模型已经实现了这种优化。

###############################################################################
# 使用 parameter.grad = None 而不是 model.zero_grad() 或 optimizer.zero_grad()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 不要调用:
model.zero_grad()
# 或
optimizer.zero_grad()

###############################################################################
# 而是使用以下方法清零梯度:

for param in model.parameters():
    param.grad = None

###############################################################################
# 第二段代码不会清零每个参数的内存,
# 而且在后续的反向传播过程中使用赋值而不是累加来存储梯度,这减少了内存操作的数量。
#
# 将梯度设置为 ``None`` 与将其设置为零有略微不同的数值行为,
# 更多详细信息请参阅
# `文档 <https://pytorch.org/docs/master/optim.html#torch.optim.Optimizer.zero_grad>`_。
#
# 或者,从 PyTorch 1.7 开始,调用 ``model`` 或
# ``optimizer.zero_grad(set_to_none=True)``。

###############################################################################
# 融合点运算
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 点运算 (元素级加法、乘法、数学函数 - ``sin()``、``cos()``、``sigmoid()`` 等) 可以融合为单个内核,
# 从而分摊内存访问时间和内核启动时间。
#
# `PyTorch JIT <https://pytorch.org/docs/stable/jit.html>`_ 可以自动融合内核,
# 尽管编译器中可能还有未实现的其他融合机会,并且并非所有设备类型都得到同等支持。
#
# 点运算是内存密集型的,PyTorch 会为每个操作启动单独的内核。
# 每个内核都会从内存加载数据、执行计算(这一步通常是廉价的)并将结果存储回内存。
#
# 融合的算子只为多个融合的点运算启动一个内核,并且只需要一次从内存加载/存储数据。
# 这使得 JIT 非常适用于激活函数、优化器、自定义 RNN 单元等。
#
# 在最简单的情况下,可以通过将
# `torch.jit.script <https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script>`_
# 装饰器应用于函数定义来启用融合,例如:

@torch.jit.script
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

###############################################################################
# 有关更高级用法,请参阅
# `TorchScript 文档 <https://pytorch.org/docs/stable/jit.html>`_。

###############################################################################
# 为计算机视觉模型启用 channels_last 内存格式
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch 1.5 引入了对卷积网络 ``channels_last`` 内存格式的支持。
# 此格式旨在与
# `AMP <https://pytorch.org/docs/stable/amp.html>`_ 结合使用,
# 进一步加速使用
# `Tensor Cores <https://www.nvidia.com/en-us/data-center/tensor-cores/>`_ 的卷积神经网络。
#
# 对 ``channels_last`` 的支持是实验性的,但预计可以用于标准计算机视觉模型(例如 ResNet-50、SSD)。
# 要将模型转换为 ``channels_last`` 格式,请按照
# `Channels Last Memory Format Tutorial <https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_ 中的说明操作。
# 该教程包括一节关于
# `转换现有模型 <https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models>`_。

###############################################################################
# 检查点中间缓冲区
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 缓冲区检查点是一种技术,用于缓解模型训练的内存容量负担。
# 与存储所有层的输入以计算反向传播中的上游梯度不同,
# 它存储少数几层的输入,其余层的输入在反向传播过程中重新计算。
# 减少的内存需求使得可以增加批量大小,从而提高利用率。
#
# 应谨慎选择检查点目标。最好不要存储具有小重新计算成本的大型层输出。
# 示例目标层包括激活函数(例如 ``ReLU``、``Sigmoid``、``Tanh``)、
# 上/下采样以及具有小累积深度的矩阵-向量运算。
#
# PyTorch 支持原生
# `torch.utils.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_
# 自动执行检查点和重新计算的API。

###############################################################################
# 禁用调试API
# ~~~~~~~~~~~~~~~~~~~~~~
# 许多PyTorch API旨在用于调试,在常规训练运行时应该禁用:
#
# * 异常检测:
#   `torch.autograd.detect_anomaly <https://pytorch.org/docs/stable/autograd.html#torch.autograd.detect_anomaly>`_
#   或
#   `torch.autograd.set_detect_anomaly(True) <https://pytorch.org/docs/stable/autograd.html#torch.autograd.set_detect_anomaly>`_
# * 与profiler相关:
#   `torch.autograd.profiler.emit_nvtx <https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx>`_,
#   `torch.autograd.profiler.profile <https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile>`_
# * autograd `gradcheck`:
#   `torch.autograd.gradcheck <https://pytorch.org/docs/stable/autograd.html#torch.autograd.gradcheck>`_
#   或
#   `torch.autograd.gradgradcheck <https://pytorch.org/docs/stable/autograd.html#torch.autograd.gradgradcheck>`_
#

###############################################################################
# CPU特定优化
# --------------------------

###############################################################################
# 利用非均匀内存访问(NUMA)控制
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NUMA或非均匀内存访问是一种内存布局设计,用于多内存控制器和内存块的多套接字机器中,旨在利用本地内存的局部性。一般来说,所有深度学习工作负载(训练或推理)都能从不跨NUMA节点访问硬件资源中获得更好的性能。因此,可以使用多个实例运行推理,每个实例在一个套接字上运行,以提高吞吐量。对于单节点上的训练任务,建议使用分布式训练,使每个训练进程在一个套接字上运行。
#
# 通常,以下命令仅在第N个节点上的核心上执行PyTorch脚本,并避免跨套接字内存访问,从而减少内存访问开销。
#
# .. code-block:: sh
#
#    numactl --cpunodebind=N --membind=N python <pytorch_script>

###############################################################################
# 更详细的描述可以在 `这里 <https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html>`_ 找到。

###############################################################################
# 利用OpenMP
# ~~~~~~~~~~~~~~
# OpenMP用于为并行计算任务带来更好的性能。
# `OMP_NUM_THREADS` 是可用于加速计算的最简单开关,它决定了用于OpenMP计算的线程数。
# CPU亲和性设置控制如何在多个核心上分配工作负载。它会影响通信开销、缓存行失效开销或页面抖动,因此正确设置CPU亲和性会带来性能优势。`GOMP_CPU_AFFINITY` 或 `KMP_AFFINITY` 决定如何将OpenMP线程绑定到物理处理单元。详细信息可以在 `这里 <https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html>`_ 找到。

###############################################################################
# 使用以下命令,PyTorch将在N个OpenMP线程上运行任务。
#
# .. code-block:: sh
#
#    export OMP_NUM_THREADS=N

###############################################################################
# 通常,使用以下环境变量来设置GNU OpenMP实现的CPU亲和性。`OMP_PROC_BIND` 指定线程是否可以在处理器之间移动。将其设置为CLOSE可以使OpenMP线程靠近主线程在连续的分区中。`OMP_SCHEDULE` 决定了OpenMP线程的调度方式。`GOMP_CPU_AFFINITY` 将线程绑定到特定的CPU。
#
# .. code-block:: sh
#
#    export OMP_SCHEDULE=STATIC
#    export OMP_PROC_BIND=CLOSE
#    export GOMP_CPU_AFFINITY="N-M"

###############################################################################
# Intel OpenMP运行时库 (`libiomp`)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 默认情况下,PyTorch使用GNU OpenMP (GNU `libgomp`)进行并行计算。在Intel平台上,Intel OpenMP运行时库(`libiomp`)提供了OpenMP API规范支持。与`libgomp`相比,它有时会带来更多的性能优势。利用环境变量`LD_PRELOAD`可以将OpenMP库切换到`libiomp`:
#
# .. code-block:: sh
#
#    export LD_PRELOAD=<path>/libiomp5.so:$LD_PRELOAD

###############################################################################
# 与GNU OpenMP中的CPU亲和性设置类似,`libiomp`中也提供了环境变量来控制CPU亲和性设置。
# `KMP_AFFINITY` 将OpenMP线程绑定到物理处理单元。`KMP_BLOCKTIME` 设置线程在完成并行区域执行后等待睡眠之前的时间(以毫秒为单位)。在大多数情况下,将`KMP_BLOCKTIME`设置为1或0可以获得良好的性能。
# 以下命令显示了使用Intel OpenMP运行时库的常见设置。
#
# .. code-block:: sh
#
#    export KMP_AFFINITY=granularity=fine,compact,1,0
#    export KMP_BLOCKTIME=1

###############################################################################
# 切换内存分配器
# ~~~~~~~~~~~~~~~~~~~~~~~
# 对于深度学习工作负载,与默认的`malloc`函数相比,`Jemalloc`或`TCMalloc`可以通过尽可能重用内存获得更好的性能。`Jemalloc <https://github.com/jemalloc/jemalloc>`_ 是一个通用的`malloc`实现,强调避免碎片和可扩展的并发支持。`TCMalloc <https://google.github.io/tcmalloc/overview.html>`_ 也具有一些优化,可以加速程序执行。其中一个优化是在缓存中保存内存,以加快常用对象的访问速度。即使在释放内存后,保持这些缓存也有助于避免昂贵的系统调用,如果稍后重新分配这些内存。
# 使用环境变量`LD_PRELOAD`来利用其中之一。
#
# .. code-block:: sh
#
#    export LD_PRELOAD=<jemalloc.so/tcmalloc.so>:$LD_PRELOAD

###############################################################################
# 使用oneDNN Graph与TorchScript进行推理
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# oneDNN Graph可以显著提高推理性能。它将一些计算密集型操作(如卷积、矩阵乘法)与其相邻操作融合。
# 在PyTorch 2.0中,它作为测试版功能支持`Float32`和`BFloat16`数据类型。
# oneDNN Graph接收模型的图形,并根据示例输入的形状识别运算符融合的候选对象。
# 模型应该使用示例输入进行JIT跟踪。
# 对于与示例输入具有相同形状的输入,在几次热身迭代后就会观察到加速。
# 下面的代码片段是针对resnet50的,但它们也可以很好地扩展到使用自定义模型的oneDNN Graph。

# 只需要这一行额外的代码即可使用oneDNN Graph
torch.jit.enable_onednn_fusion(True)

###############################################################################
# 使用oneDNN Graph API进行Float32推理只需要一行额外的代码。
# 如果您正在使用oneDNN Graph,请避免调用 `torch.jit.optimize_for_inference` 。

# 示例输入应该与预期输入具有相同的形状
sample_input = [torch.rand(32, 3, 224, 224)]
# 在此示例中使用torchvision中的resnet50进行说明,
# 但下面的代码确实可以修改为使用自定义模型。
model = getattr(torchvision.models, "resnet50")().eval()
# 使用示例输入跟踪模型
traced_model = torch.jit.trace(model, sample_input)
# 调用torch.jit.freeze
traced_model = torch.jit.freeze(traced_model)

###############################################################################
# 一旦使用示例输入对模型进行了JIT跟踪,就可以在几次热身运行后用于推理。

with torch.no_grad():
    # 几次热身运行
    traced_model(*sample_input)
    traced_model(*sample_input)
    # 在热身运行后会观察到加速
    traced_model(*sample_input)

###############################################################################
# 虽然oneDNN Graph的JIT融合器也支持`BFloat16`数据类型的推理,
# 但只有具有AVX512_BF16指令集架构(ISA)的机器才能从oneDNN Graph中获得性能优势。
# 以下代码片段是使用`BFloat16`数据类型进行oneDNN Graph推理的示例:

# JIT模式下的AMP默认启用,并且与其eager模式对应版本不同
torch._C._jit_set_autocast_mode(False)

with torch.no_grad(), torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16):
    # 当使用AMP时,应使用`torch.fx.experimental.optimization.fuse`进行基于CNN的视觉模型的Conv-BatchNorm折叠
    import torch.fx.experimental.optimization as optimization
    # 请注意,当不使用AMP时,无需调用optimization.fuse
    model = optimization.fuse(model)
    model = torch.jit.trace(model, (example_input))
    model = torch.jit.freeze(model)
    # 几次热身运行
    model(example_input)
    model(example_input)
    # 在后续运行中会观察到加速。
    model(example_input)


###############################################################################
# 使用PyTorch `DistributedDataParallel` (DDP)功能在CPU上训练模型
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 对于小型模型或内存限制型模型(如DLRM),在CPU上进行训练也是一个不错的选择。在具有多个套接字的机器上,
# 分布式训练可以带来高效的硬件资源使用,从而加速训练过程。
# `Torch-ccl <https://github.com/intel/torch-ccl>`_ 使用Intel(R) `oneCCL` (集体通信库)进行了优化,
# 用于高效的分布式深度学习训练,实现了诸如 `allreduce`、`allgather`、`alltoall` 等集体操作,
# 实现了PyTorch C10D `ProcessGroup` API,并可以作为外部 `ProcessGroup` 动态加载。
# 在PyTorch DDP模块中实现的优化之上, `torch-ccl` 加速了通信操作。
# 除了对通信内核进行优化外, `torch-ccl` 还支持同步计算和通信功能。

###############################################################################
# GPU 特定优化
# --------------------------

###############################################################################
# 启用 cuDNN 自动调优器
# ~~~~~~~~~~~~~~~~~~~~~~~
# `NVIDIA cuDNN <https://developer.nvidia.com/cudnn>`_ 支持许多算法来计算卷积。
# 自动调优器会运行一个简短的基准测试,并为给定的硬件和输入大小选择性能最佳的内核。
#
# 对于卷积网络(目前其他类型尚不支持),可以在启动训练循环之前启用 cuDNN 自动调优器,方法是设置:

torch.backends.cudnn.benchmark = True
###############################################################################
#
# * 自动调优器的决策可能是非确定性的;不同的算法可能会在不同的运行中被选择。
#   有关更多详细信息,请参阅 `PyTorch: 可重复性 <https://pytorch.org/docs/stable/notes/randomness.html?highlight=determinism>`_
# * 在某些罕见的情况下,例如输入大小高度可变时,最好在禁用自动调优器的情况下运行卷积网络,
#   以避免为每个输入大小选择算法所带来的开销。
#

###############################################################################
# 避免不必要的 CPU-GPU 同步
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 避免不必要的同步,尽可能让 CPU 领先于加速器运行,以确保加速器工作队列中包含许多操作。
#
# 如果可能,请避免需要同步的操作,例如:
#
# * ``print(cuda_tensor)``
# * ``cuda_tensor.item()``
# * 内存复制: ``tensor.cuda()``, ``cuda_tensor.cpu()`` 和等效的 ``tensor.to(device)`` 调用
# * ``cuda_tensor.nonzero()``
# * 依赖于在 CUDA 张量上执行的操作结果的 python 控制流,例如 ``if (cuda_tensor != 0).all()``
#

###############################################################################
# 直接在目标设备上创建张量
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 不要调用 ``torch.rand(size).cuda()`` 来生成随机张量,而是直接在目标设备上生成输出:
# ``torch.rand(size, device='cuda')``。
#
# 这适用于所有创建新张量并接受 ``device`` 参数的函数:
# `torch.rand() <https://pytorch.org/docs/stable/generated/torch.rand.html#torch.rand>`_,
# `torch.zeros() <https://pytorch.org/docs/stable/generated/torch.zeros.html#torch.zeros>`_,
# `torch.full() <https://pytorch.org/docs/stable/generated/torch.full.html#torch.full>`_
# 和类似函数。

###############################################################################
# 使用混合精度和 AMP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 混合精度利用 `Tensor Cores <https://www.nvidia.com/en-us/data-center/tensor-cores/>`_,
# 在 Volta 及更新的 GPU 架构上可提供高达 3 倍的整体加速。要使用 Tensor Cores,需要启用 AMP,
# 并且矩阵/张量的维度需要满足调用使用 Tensor Cores 的内核的要求。
#
# 要使用 Tensor Cores:
#
# * 将大小设置为 8 的倍数(以映射到 Tensor Cores 的维度)
#
#   * 请参阅 `深度学习性能文档 <https://docs.nvidia.com/deeplearning/performance/index.html#optimizing-performance>`_
#     以获取更多详细信息和特定于层类型的指南
#   * 如果层大小是由其他参数而不是固定值派生的,它仍然可以显式填充,例如 NLP 模型中的词汇量大小
#
# * 启用 AMP
#
#   * 混合精度训练和 AMP 介绍:
#     `视频 <https://www.youtube.com/watch?v=jF4-_ZK_tyc&feature=youtu.be>`_,
#     `幻灯片 <https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf>`_
#   * PyTorch 从 1.6 版本开始提供原生 AMP 支持:
#     `文档 <https://pytorch.org/docs/stable/amp.html>`_,
#     `示例 <https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples>`_,
#     `教程 <https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html>`_
#

###############################################################################
# 在输入长度可变的情况下预分配内存
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 用于语音识别或 NLP 的模型通常在具有可变序列长度的输入张量上进行训练。
# 可变长度可能会对 PyTorch 缓存分配器造成问题,并导致性能降低或意外的内存不足错误。
# 如果一个短序列长度的批次后面紧跟着另一个长序列长度的批次,那么 PyTorch 就被迫释放前一次迭代的中间缓冲区,
# 并重新分配新的缓冲区。这个过程是耗时的,并会在缓存分配器中造成碎片,从而可能导致内存不足错误。
#
# 一个典型的解决方案是实现预分配。它包括以下步骤:
#
# #. 生成一个(通常是随机的)具有最大序列长度的输入批次(要么对应于训练数据集中的最大长度,
#    要么对应于某个预定义的阈值)
# #. 使用生成的批次执行前向和后向传递,不执行优化器或学习率调度器,这一步预分配了最大大小的缓冲区,
#    可在后续训练迭代中重用
# #. 将梯度归零
# #. 继续常规训练
#

###############################################################################
# 分布式优化
# -------------------------

###############################################################################
# 使用高效的数据并行后端
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch 有两种方式来实现数据并行训练:
#
# * `torch.nn.DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel>`_
# * `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
#
# ``DistributedDataParallel`` 提供了更好的性能和多 GPU 扩展能力。
# 有关更多信息,请参阅 PyTorch 文档中 `相关 CUDA 最佳实践部分 <https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel>`_。

###############################################################################
# 如果在使用 ``DistributedDataParallel`` 和梯度累积进行训练时,跳过不必要的 all-reduce
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 默认情况下,
# `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
# 在每次反向传播后执行梯度 all-reduce,以计算参与训练的所有工作进程上的平均梯度。
# 如果训练使用了 N 步梯度累积,那么在每个训练步骤后都不需要执行 all-reduce,
# 只需在最后一次调用 backward 之后,在执行优化器之前执行 all-reduce。
#
# ``DistributedDataParallel`` 提供了
# `no_sync() <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync>`_
# 上下文管理器,用于在特定迭代中禁用梯度 all-reduce。
# ``no_sync()`` 应该应用于梯度累积的前 ``N-1`` 次迭代,最后一次迭代应该遵循默认执行,并执行所需的梯度 all-reduce。

###############################################################################
# 如果使用 ``DistributedDataParallel(find_unused_parameters=True)``,则在构造函数和执行期间匹配层的顺序
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_
# 使用 ``find_unused_parameters=True`` 时,会根据模型构造函数中层和参数的顺序来构建 ``DistributedDataParallel`` 梯度 all-reduce 的桶。
# ``DistributedDataParallel`` 会与反向传播重叠 all-reduce。只有当给定桶中的所有参数的梯度都可用时,
# 才会异步触发该桶的 all-reduce。
#
# 为了最大化重叠量,模型构造函数中的顺序应该大致与执行期间的顺序相匹配。
# 如果顺序不匹配,那么整个桶的 all-reduce 将等待最后到达的梯度,这可能会减少反向传播和 all-reduce 之间的重叠,
# all-reduce 可能会暴露出来,从而减慢训练速度。
#
# ``DistributedDataParallel`` 使用 ``find_unused_parameters=False`` (这是默认设置)
# 依赖于基于反向传播期间遇到的操作顺序的自动桶形成。
# 使用 ``find_unused_parameters=False`` 时,无需重新排列层或参数即可获得最佳性能。

###############################################################################
# 在分布式设置中平衡工作负载
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 对于处理序列数据的模型(语音识别、翻译、语言模型等),通常可能会发生负载不均衡。
# 如果一个设备收到的批次数据的序列长度比其他设备长,那么所有设备都要等待完成最后的工作进程。
# 在使用 `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel>`_ 后端的分布式设置中,
# 反向传播函数作为一个隐式的同步点。
#
# 有多种方法可以解决负载平衡问题。核心思想是在每个全局批次中尽可能均匀地将工作负载分布到所有工作进程。
# 例如,Transformer 通过形成具有大约恒定令牌数(而不是序列数)的批次来解决不平衡问题,
# 其他模型通过对具有相似序列长度的样本进行分桶或甚至对数据集按序列长度进行排序来解决不平衡问题。
