使用TensorPipe CUDA RPC进行设备到设备通信
==============================================================

.. note:: 直接设备到设备RPC（CUDA RPC）在PyTorch 1.8中作为原型功能引入。此API可能会发生变化。

在本教程中，您将学习：

- CUDA RPC的高级概念。
- 如何使用CUDA RPC。


要求
------------

- PyTorch 1.8+
- `分布式RPC框架入门 <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`_


什么是CUDA RPC？
------------------------------------

CUDA RPC支持直接从本地CUDA内存向远程CUDA内存发送张量。在1.8版本发布之前，PyTorch RPC只接受CPU张量。
因此，当应用程序需要通过RPC发送CUDA张量时，它必须首先将张量移动到调用方的CPU，通过RPC发送，
然后在被调用方将其移动到目标设备，这会导致不必要的同步和D2H和H2D复制。从1.8版本开始，RPC允许用户使用
`set_device_map <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map>`_
API配置每个进程的全局设备映射，指定如何将本地设备映射到远程设备。具体来说，如果``worker0``的设备映射有一个条目
``"worker1" : {"cuda:0" : "cuda:1"}``，所有来自 ``worker0`` 的 ``"cuda:0"`` 上的RPC参数
将直接发送到 ``worker1``的``"cuda:1"`` 。RPC的响应将使用调用方设备映射的逆映射，即如果 ``worker1``
返回 ``"cuda:1"`` 上的张量，它将直接发送到 ``worker0`` 的 ``"cuda:0"`` 。
所有预期的设备到设备直接通信必须在每个进程的设备映射中指定。否则，只允许CPU张量。

在底层，PyTorch RPC依赖于 `TensorPipe <https://github.com/pytorch/tensorpipe>`_ 作为通信后端。
PyTorch RPC从每个请求或响应中提取所有张量到一个列表中，并将其他所有内容打包成二进制负载。
然后，TensorPipe将根据张量设备类型和调用方和被调用方的通道可用性，自动为每个张量选择通信通道。
现有的 TensorPipe 通道涵盖 NVLink、InfiniBand、SHM、CMA、TCP 等。

如何使用CUDA RPC？
---------------------------------------

以下代码展示了如何使用CUDA RPC。该模型包含两个线性层，被分成两个分片。这两个分片分别放置在 ``worker0`` 和 ``worker1`` 上，
``worker0`` 作为主节点驱动前向和后向传播。请注意，我们有意跳过了
`DistributedOptimizer <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`_
以突出使用 CUDA RPC 时的性能改进。实验重复前向和后向传播 10 次，并测量总执行时间。它比较了使用 CUDA RPC 与手动暂存到 CPU 内存
并使用 CPU RPC 的情况。


::

    import torch
    import torch.distributed.autograd as autograd
    import torch.distributed.rpc as rpc
    import torch.multiprocessing as mp
    import torch.nn as nn

    import os
    import time


    class MyModule(nn.Module):
        def __init__(self, device, comm_mode):
            super().__init__()
            self.device = device
            self.linear = nn.Linear(1000, 1000).to(device)
            self.comm_mode = comm_mode

        def forward(self, x):
            # 如果x已经在self.device上，x.to()是一个空操作
            y = self.linear(x.to(self.device))
            return y.cpu() if self.comm_mode == "cpu" else y

        def parameter_rrefs(self):
            return [rpc.RRef(p) for p in self.parameters()]


    def measure(comm_mode):
        # "worker0/cuda:0"上的本地模块
        lm = MyModule("cuda:0", comm_mode)
        # "worker1/cuda:1"上的远程模块
        rm = rpc.remote("worker1", MyModule, args=("cuda:1", comm_mode))
        # 准备随机输入
        x = torch.randn(1000, 1000).cuda(0)

        tik = time.time()
        for _ in range(10):
            with autograd.context() as ctx:
                y = rm.rpc_sync().forward(lm(x))
                autograd.backward(ctx, [y.sum()])
        # 在"cuda:0"上同步，以确保所有待处理的CUDA操作都包含在测量中
        torch.cuda.current_stream("cuda:0").synchronize()
        tok = time.time()
        print(f"{comm_mode} RPC总执行时间：{tok - tik}")


    def run_worker(rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

        if rank == 0:
            options.set_device_map("worker1", {0: 1})
            rpc.init_rpc(
                f"worker{rank}",
                rank=rank,
                world_size=2,
                rpc_backend_options=options
            )
            measure(comm_mode="cpu")
            measure(comm_mode="cuda")
        else:
            rpc.init_rpc(
                f"worker{rank}",
                rank=rank,
                world_size=2,
                rpc_backend_options=options
            )

        # 阻塞直到所有rpc完成
        rpc.shutdown()


    if __name__=="__main__":
        world_size = 2
        mp.spawn(run_worker, nprocs=world_size, join=True)

输出显示如下，表明在这个实验中，CUDA RPC 可以帮助实现 34 倍的速度提升，相比于 CPU RPC。

::

    cpu RPC总执行时间：2.3145179748535156秒
    cuda RPC总执行时间：0.06867480278015137秒
