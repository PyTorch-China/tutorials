开始使用 DeviceMesh
=====================================================

**作者**: `Iris Zhang <https://github.com/wz337>`__, `Wanchao Liang <https://github.com/wanchaol>`__

.. note::
   |edit| 在 `github <https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_device_mesh.rst>`__ 上查看和编辑本教程。

先决条件:

- `分布式通信包 - torch.distributed <https://pytorch.org/docs/stable/distributed.html>`__
- Python 3.8 - 3.11
- PyTorch 2.2


为分布式训练设置分布式通信器（例如 NVIDIA 集体通信库 (NCCL) 通信器）可能是一个重大挑战。
对于需要组合不同并行性的工作负载，用户需要为每个并行性解决方案手动设置和管理 NCCL 通信器（例如，:class:`ProcessGroup`）。
这个过程可能很复杂且容易出错。:class:`DeviceMesh` 可以简化这个过程，使其更易于管理和减少错误。

什么是 DeviceMesh
------------------
:class:`DeviceMesh` 是一个管理 :class:`ProcessGroup` 的高级抽象。它允许用户轻松创建节点间和节点内进程组，
而无需担心如何为不同的子进程组正确设置等级。
用户还可以通过 :class:`DeviceMesh` 轻松管理多维并行性的底层进程组/设备。

.. figure:: /_static/img/distributed/device_mesh.png
   :width: 100%
   :align: center
   :alt: PyTorch DeviceMesh

为什么 DeviceMesh 有用
------------------------
当处理多维并行性（例如 3-D 并行）时，DeviceMesh 非常有用，因为这种情况需要并行性组合。
例如，当您的并行性解决方案需要跨主机和每个主机内部进行通信时。上图显示，我们可以创建一个 2D 网格，
连接每个主机内的设备，并在同构设置中将每个设备与其他主机上的对应设备连接起来。

如果没有 DeviceMesh，用户在应用任何并行性之前需要手动设置每个进程上的 NCCL 通信器和 CUDA 设备，这可能相当复杂。
以下代码片段说明了如何在没有 :class:`DeviceMesh` 的情况下设置混合分片 2-D 并行模式。
首先，我们需要手动计算分片组和复制组。然后，我们需要为每个等级分配正确的分片和复制组。

.. code-block:: python

    import os

    import torch
    import torch.distributed as dist

    # 了解世界拓扑
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"在 {rank=} 上运行示例，世界大小为 {world_size=}")

    # 创建进程组以管理 2-D 类似的并行模式
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    # 创建分片组（例如 (0, 1, 2, 3), (4, 5, 6, 7)）
    # 并为每个等级分配正确的分片组
    num_node_devices = torch.cuda.device_count()
    shard_rank_lists = list(range(0, num_node_devices // 2)), list(range(num_node_devices // 2, num_node_devices))
    shard_groups = (
        dist.new_group(shard_rank_lists[0]),
        dist.new_group(shard_rank_lists[1]),
    )
    current_shard_group = (
        shard_groups[0] if rank in shard_rank_lists[0] else shard_groups[1]
    )

    # 创建复制组（例如，(0, 4), (1, 5), (2, 6), (3, 7)）
    # 并为每个等级分配正确的复制组
    current_replicate_group = None
    shard_factor = len(shard_rank_lists[0])
    for i in range(num_node_devices // 2):
        replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
        replicate_group = dist.new_group(replicate_group_ranks)
        if rank in replicate_group_ranks:
            current_replicate_group = replicate_group

要运行上面的代码片段，我们可以利用 PyTorch Elastic。让我们创建一个名为 ``2d_setup.py`` 的文件。
然后，运行以下 `torch elastic/torchrun <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ 命令。

.. code-block:: python

    torchrun --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint=localhost:29400 2d_setup.py

.. note::
    为了简化演示，我们仅使用一个节点模拟 2D 并行。请注意，此代码片段也可用于多主机设置。

借助 :func:`init_device_mesh`，我们可以仅用两行代码完成上述 2D 设置，并且如果需要，
我们仍然可以访问底层的 :class:`ProcessGroup`。


.. code-block:: python

    from torch.distributed.device_mesh import init_device_mesh
    mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))

    # 用户可以通过 `get_group` API 访问底层进程组。
    replicate_group = mesh_2d.get_group(mesh_dim="replicate")
    shard_group = mesh_2d.get_group(mesh_dim="shard")

让我们创建一个名为 ``2d_setup_with_device_mesh.py`` 的文件。
然后，运行以下 `torch elastic/torchrun <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ 命令。

.. code-block:: python

    torchrun --nproc_per_node=8 2d_setup_with_device_mesh.py


如何将 DeviceMesh 与 HSDP 一起使用
-------------------------------

混合分片数据并行（HSDP）是一种 2D 策略，在主机内执行 FSDP，在主机间执行 DDP。

让我们看一个示例，说明 DeviceMesh 如何帮助将 HSDP 应用到您的模型，使用简单的设置。使用 DeviceMesh，
用户无需手动创建和管理分片组和复制组。

.. code-block:: python

    import torch
    import torch.nn as nn

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    # HSDP: MeshShape(2, 4)
    mesh_2d = init_device_mesh("cuda", (2, 4))
    model = FSDP(
        ToyModel(), device_mesh=mesh_2d, sharding_strategy=ShardingStrategy.HYBRID_SHARD
    )

让我们创建一个名为 ``hsdp.py`` 的文件。
然后，运行以下 `torch elastic/torchrun <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ 命令。

.. code-block:: python

    torchrun --nproc_per_node=8 hsdp.py

结论
----------
总之，我们已经了解了 :class:`DeviceMesh` 和 :func:`init_device_mesh`，以及如何
使用它们来描述集群中设备的布局。

欲了解更多信息，请参阅以下内容：

- `将张量/序列并行与 FSDP 结合的 2D 并行 <https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py>`__
- `使用 PT2 的可组合 PyTorch 分布式 <chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://static.sched.com/hosted_files/pytorch2023/d1/%5BPTC%2023%5D%20Composable%20PyTorch%20Distributed%20with%20PT2.pdf>`__
