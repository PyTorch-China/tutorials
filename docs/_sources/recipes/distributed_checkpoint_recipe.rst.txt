分布式检查点 (DCP) 入门
=====================================================

**作者**: `Iris Zhang <https://github.com/wz337>`__, `Rodrigo Kumpera <https://github.com/kumpera>`__, `Chien-Chin Huang <https://github.com/fegin>`__, `Lucas Pasqualin <https://github.com/lucasllc>`__

.. note::
   |edit| 在 `github <https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_checkpoint_recipe.rst>`__ 上查看和编辑本教程。


先决条件:

-  `FullyShardedDataParallel API 文档 <https://pytorch.org/docs/master/fsdp.html>`__
-  `torch.load API 文档 <https://pytorch.org/docs/stable/generated/torch.load.html>`__


在分布式训练过程中对 AI 模型进行检查点保存可能具有挑战性，因为参数和梯度分布在不同的训练器上，而且恢复训练时可用的训练器数量可能会发生变化。
Pytorch 分布式检查点 (DCP) 可以帮助简化这个过程。

在本教程中，我们将展示如何使用 DCP API 处理一个简单的 FSDP 包装模型。


DCP 如何工作
--------------

:func:`torch.distributed.checkpoint` 允许并行地从多个 rank 保存和加载模型。您可以使用此模块在任意数量的 rank 上并行保存，
然后在加载时重新分片到不同的集群拓扑结构。

此外，通过使用 :func:`torch.distributed.checkpoint.state_dict` 中的模块，
DCP 提供了在分布式设置中优雅处理 ``state_dict`` 生成和加载的支持。
这包括管理模型和优化器之间的全限定名称 (FQN) 映射，以及为 PyTorch 提供的并行性设置默认参数。

DCP 与 :func:`torch.save` 和 :func:`torch.load` 在几个重要方面有所不同：

* 它为每个检查点生成多个文件，每个 rank 至少一个。
* 它就地操作，这意味着模型应该首先分配其数据，DCP 使用该存储而不是创建新的存储。

.. note::
  本教程中的代码在 8-GPU 服务器上运行，但可以轻松地推广到其他环境。

如何使用 DCP
--------------

这里我们使用一个用 FSDP 包装的玩具模型进行演示。同样，这些 API 和逻辑可以应用于更大的模型进行检查点保存。

保存
~~~~~~

现在，让我们创建一个玩具模块，用 FSDP 包装它，用一些虚拟输入数据对其进行训练，然后保存它。

.. code-block:: python

    import os

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.multiprocessing as mp
    import torch.nn as nn

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.checkpoint.state_dict import get_state_dict
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    CHECKPOINT_DIR = "checkpoint"


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(16, 16)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(16, 8)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355 "

        # 初始化进程组
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


    def cleanup():
        dist.destroy_process_group()


    def run_fsdp_checkpoint_save_example(rank, world_size):
        print(f"在 rank {rank} 上运行基本的 FSDP 检查点保存示例。")
        setup(rank, world_size)

        # 创建一个模型并将其移动到 ID 为 rank 的 GPU 上
        model = ToyModel().to(rank)
        model = FSDP(model)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        optimizer.zero_grad()
        model(torch.rand(8, 16, device="cuda")).sum().backward()
        optimizer.step()

        # 这行代码自动管理 FSDP FQN，并将默认状态字典类型设置为 FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict
        }
        dcp.save(state_dict,checkpoint_id=CHECKPOINT_DIR)


        cleanup()


    if __name__ == "__main__":
        world_size = torch.cuda.device_count()
        print(f"在 {world_size} 个设备上运行 FSDP 检查点示例。")
        mp.spawn(
            run_fsdp_checkpoint_save_example,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

请查看 `checkpoint` 目录。您应该看到 8 个检查点文件，如下所示。

.. figure:: /_static/img/distributed/distributed_checkpoint_generated_files.png
   :width: 100%
   :align: center
   :alt: 分布式检查点

加载
~~~~~~~

保存之后，让我们创建相同的 FSDP 包装模型，并从存储中加载保存的状态字典到模型中。您可以在相同的世界大小或不同的世界大小中加载。

请注意，您需要在加载之前调用 :func:`model.state_dict`，并将其传递给 DCP 的 :func:`load_state_dict` API。
这与 :func:`torch.load` 有根本的不同，因为 :func:`torch.load` 只需要加载前的检查点路径。
我们需要在加载之前提供 ``state_dict`` 的原因是：

* DCP 使用模型状态字典中预分配的存储来从检查点目录加载。在加载过程中，传入的状态字典将被就地更新。
* DCP 在加载之前需要模型的分片信息以支持重新分片。

.. code-block:: python

    import os

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    import torch.multiprocessing as mp
    import torch.nn as nn

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    CHECKPOINT_DIR = "checkpoint"


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(16, 16)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(16, 8)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355 "

        # 初始化进程组
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


    def cleanup():
        dist.destroy_process_group()


    def run_fsdp_checkpoint_load_example(rank, world_size):
        print(f"在 rank {rank} 上运行基本的 FSDP 检查点加载示例。")
        setup(rank, world_size)

        # 创建一个模型并将其移动到 ID 为 rank 的 GPU 上
        model = ToyModel().to(rank)
        model = FSDP(model)

        # 生成我们将加载到的状态字典
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict
        }
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=CHECKPOINT_DIR,
        )
        # 在加载完成后，将我们的状态字典设置到模型和优化器上
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optimizer_state_dict
        )

        cleanup()


    if __name__ == "__main__":
        world_size = torch.cuda.device_count()
        print(f"在 {world_size} 个设备上运行 FSDP 检查点示例。")
        mp.spawn(
            run_fsdp_checkpoint_load_example,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

如果您想在非分布式设置中将保存的检查点加载到非 FSDP 包装的模型中，可能是为了推理，您也可以使用 DCP 来实现。
默认情况下，DCP 以单程序多数据 (SPMD) 风格保存和加载分布式 ``state_dict``。但是，如果没有初始化进程组，
DCP 会推断意图是以"非分布式"方式保存或加载，这意味着完全在当前进程中进行。

.. note::
  多程序多数据的分布式检查点支持仍在开发中。

.. code-block:: python

    import os

    import torch
    import torch.distributed.checkpoint as DCP
    import torch.nn as nn


    CHECKPOINT_DIR = "checkpoint"


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(16, 16)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(16, 8)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def run_checkpoint_load_example():
        # 创建非 FSDP 包装的玩具模型
        model = ToyModel()
        state_dict = {
            "model": model.state_dict(),
        }

        # 由于没有初始化进程组，DCP 将禁用任何集体操作
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=CHECKPOINT_DIR,
        )
        model.load_state_dict(state_dict["model"])

    if __name__ == "__main__":
        print(f"运行基本的 DCP 检查点加载示例。")
        run_checkpoint_load_example()


结论
----------
总之，我们学习了如何使用 DCP 的 :func:`save` 和 :func:`load` API，以及它们与 :func:`torch.save` 和 :func:`torch.load` 的不同之处。
此外，我们还学习了如何使用 :func:`get_state_dict` 和 :func:`set_state_dict` 在状态字典生成和加载期间自动管理并行性特定的 FQN 和默认值。

更多信息，请参阅以下内容：

-  `保存和加载模型教程 <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`__
-  `FullyShardedDataParallel 入门教程 <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__
