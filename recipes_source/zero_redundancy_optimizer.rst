使用ZeroRedundancyOptimizer分片优化器状态
=========================================

在本教程中，您将学习：

- `ZeroRedundancyOptimizer <https://pytorch.org/docs/master/distributed.optim.html>`__ 的高级概念。
- 如何在分布式训练中使用 `ZeroRedundancyOptimizer <https://pytorch.org/docs/master/distributed.optim.html>`__
  及其影响。


要求
----

- PyTorch 1.8+
- `分布式数据并行入门 <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_


什么是 ``ZeroRedundancyOptimizer``？
------------------------------------

`ZeroRedundancyOptimizer <https://pytorch.org/docs/master/distributed.optim.html>`__
的想法来自 `DeepSpeed/ZeRO 项目 <https://github.com/microsoft/DeepSpeed>`_ 和
`Marian <https://github.com/marian-nmt/marian-dev>`_，它们在分布式数据并行进程中
分片优化器状态，以减少每个进程的内存占用。

在 `分布式数据并行入门 <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_
教程中，我们展示了如何使用
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_
(DDP) 来训练模型。在该教程中，每个进程都保留一个专用的优化器副本。由于DDP已经在反向传播中同步了梯度，
所有优化器副本在每次迭代中都将对相同的参数和梯度值进行操作，这就是DDP保持模型副本处于相同状态的方式。
通常，优化器还会维护本地状态。例如， ``Adam`` 优化器使用每个参数的 ``exp_avg`` 和 ``exp_avg_sq`` 状态。
因此， ``Adam`` 优化器的内存消耗至少是模型大小的两倍。基于这个观察，我们可以通过在DDP进程之间分片
优化器状态来减少优化器的内存占用。具体来说，不是为所有参数创建每个参数的状态，而是每个DDP进程中的优化器实例
只保留所有模型参数中一个分片的优化器状态。优化器的 ``step()`` 函数只更新其分片中的参数，
然后将更新后的参数广播到所有其他对等DDP进程，以便所有模型副本仍然处于相同的状态。

如何使用 ``ZeroRedundancyOptimizer``？
--------------------------------------

以下代码演示了如何使用
`ZeroRedundancyOptimizer <https://pytorch.org/docs/master/distributed.optim.html>`__。
大部分代码与 `分布式数据并行说明 <https://pytorch.org/docs/stable/notes/ddp.html>`_
中的简单DDP示例相似。主要区别在于 ``example`` 函数中的 ``if-else`` 子句，它包装了优化器构造，
在 `ZeroRedundancyOptimizer <https://pytorch.org/docs/master/distributed.optim.html>`__
和 ``Adam`` 优化器之间切换。


::

    import os
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from torch.nn.parallel import DistributedDataParallel as DDP

    def print_peak_memory(prefix, device):
        if device == 0:
            print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

    def example(rank, world_size, use_zero):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        # 创建默认进程组
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # 创建本地模型
        model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
        print_peak_memory("创建本地模型后的最大内存分配", rank)

        # 构建DDP模型
        ddp_model = DDP(model, device_ids=[rank])
        print_peak_memory("创建DDP后的最大内存分配", rank)

        # 定义损失函数和优化器
        loss_fn = nn.MSELoss()
        if use_zero:
            optimizer = ZeroRedundancyOptimizer(
                ddp_model.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=0.01
            )
        else:
            optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)

        # 前向传播
        outputs = ddp_model(torch.randn(20, 2000).to(rank))
        labels = torch.randn(20, 2000).to(rank)
        # 反向传播
        loss_fn(outputs, labels).backward()

        # 更新参数
        print_peak_memory("优化器step()之前的最大内存分配", rank)
        optimizer.step()
        print_peak_memory("优化器step()之后的最大内存分配", rank)

        print(f"参数总和为: {sum(model.parameters()).sum()}")



    def main():
        world_size = 2
        print("=== Using ZeroRedundancyOptimizer ===")
        mp.spawn(example,
            args=(world_size, True),
            nprocs=world_size,
            join=True)

        print("=== Not Using ZeroRedundancyOptimizer ===")
        mp.spawn(example,
            args=(world_size, False),
            nprocs=world_size,
            join=True)

    if __name__=="__main__":
        main()

输出如下所示。当使用 ``ZeroRedundancyOptimizer`` 和 ``Adam`` 时，优化器 ``step()``的峰值内存消耗
是普通 ``Adam`` 内存消耗的一半。这符合我们的预期，因为我们在两个进程之间分片了 ``Adam`` 优化器状态。
输出还显示，使用 ``ZeroRedundancyOptimizer`` 时，模型参数在一次迭代后仍然得到相同的值
（使用和不使用 ``ZeroRedundancyOptimizer`` 时参数总和相同）。

::

    === Using ZeroRedundancyOptimizer ===
    Max memory allocated after creating local model: 335.0MB
    Max memory allocated after creating DDP: 656.0MB
    Max memory allocated before optimizer step(): 992.0MB
    Max memory allocated after optimizer step(): 1361.0MB
    params sum is: -3453.6123046875
    params sum is: -3453.6123046875
    === Not Using ZeroRedundancyOptimizer ===
    Max memory allocated after creating local model: 335.0MB
    Max memory allocated after creating DDP: 656.0MB
    Max memory allocated before optimizer step(): 992.0MB
    Max memory allocated after optimizer step(): 1697.0MB
    params sum is: -3453.6123046875
    params sum is: -3453.6123046875
