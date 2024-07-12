支持 TorchScript 的分布式优化器
==============================================================

.. note:: 支持 TorchScript 的分布式优化器在 PyTorch 1.8 中作为 beta 功能引入。
    此 API 可能会发生变化。

在本教程中，您将学习：

- 支持 TorchScript 的分布式优化器的高级概念及其带来的功能
- 如何编写支持 TorchScript 的自定义分布式优化器


要求
------------

- PyTorch 1.8+
- `分布式 RPC 框架入门 <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`_


什么是分布式优化器？
------------------------------------

`DistributedOptimizer <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`_
接受一个远程参数列表（RRef），并在参数所在的工作节点上本地运行优化器。它通常与分布式 RPC/Autograd 一起使用，
用于模型并行训练。它可以使用任何本地优化器算法（无论是 ``torch.optim`` 中预定义的算法还是自定义的算法）在每个工作节点上应用梯度。


什么是支持 TorchScript 的分布式优化器？
-------------------------------------------------------

分布式优化器在分布式模型并行训练中被广泛使用。在一些常见用例中，由于性能考虑和资源利用，训练需要以多线程方式进行，
而不是多进程方式（或至少部分多线程，例如参数服务器托管部分模型和参数，新线程根据请求更新参数）。
PyTorch 本身由于 Python 的全局解释器锁（GIL）而不支持原生多线程训练，但它可以利用
`TorchScript <https://pytorch.org/docs/stable/jit.html>`_ 来摆脱 GIL 并以多线程方式运行模型。

对于关键的模型训练工作负载，提高训练性能是一个重要话题。研究人员经常希望通过图表示实现不同的优化策略（例如通过算子融合）
或实现自定义算子内核以加速训练。

支持 TorchScript 的分布式优化器可以帮助摆脱 GIL，从而提高 PyTorch 在多线程环境中的训练性能，
它还解锁了使用 TorchScript 提供的高级编译器技术（例如 CPU/GPU 融合）来进一步提升性能的潜力。


如何编写支持 TorchScript 的自定义分布式优化器？
-------------------------------------------------------------------------

以下代码展示了如何基于现有的本地优化器实现编写自定义分布式优化器，从而解锁 TorchScript 的优势，包括 GIL 移除和性能改进机会。

假设您已经有一个在训练中使用的本地优化器，在这个例子中，我们将使用
`准超曲面动量（QHM）<https://github.com/facebookresearch/qhoptim/blob/e81dea3f2765780cf4fbb90b87b22ba7604b8625/qhoptim/pyt/qhm.py#L12>`_
来展示如何启用 TorchScript 支持。请注意，这也适用于任何继承自 ``torch.optim.Optimizer`` 的自定义优化器。

首先，我们需要将计算和状态管理从优化器实现中分离出来，这样我们就可以提取计算部分并将其变成一个独立函数，这对 TorchScript 友好。
这有两个好处：

1. 计算逻辑变得更容易检查，我们可以快速将参数更新/计算部分转换为 TorchScript，
并利用 TorchScript IR 进行进一步优化（算子融合等）
2. 分布式优化器底层使用不同的机制来获取梯度和更新参数（我们单独存储梯度，而不是在反向传播期间直接填充 ``param.grad`` 字段）。
分离计算允许分布式优化器在多线程模式下进行优化器更新，因为它消除了对 ``param.grad`` 的可能竞争条件。


::

    import torch
    from torch import Tensor
    from typing import List


    def qhm_update(params: List[Tensor],
                dp_list: List[Tensor],
                momentum_buffer_list: List[Tensor],
                lr: float,
                nu: float,
                weight_decay: float,
                weight_decay_type: str,
                momentum: float):

        for p, d_p, momentum_buffer in zip(params, dp_list, momentum_buffer_list):
            if weight_decay != 0:
                if weight_decay_type == "grad":
                    d_p.add_(weight_decay, p)
                elif weight_decay_type == "direct":
                    p.mul_(1.0 - lr * weight_decay)
                else:
                    raise ValueError("Invalid weight decay type provided")

            momentum_buffer.mul_(momentum).add_(1.0 - momentum, d_p)

            p.data.add_(-lr * nu, momentum_buffer)
            p.data.add_(-lr * (1.0 - nu), d_p)



接下来，我们将定义一个支持 TorchScript 的分布式函数式优化器来管理优化器状态，并调用我们上面定义的兼容 TorchScript 的更新函数。
请注意，与普通自定义优化器相比，有几个约定不同：

1. 我们不继承 ``torch.optim.Optimizer``，因为 TorchScript 不支持多态
2. ``step`` 接受梯度列表而不是损失闭包。

::

    import torch
    from torch import Tensor
    from typing import List, Optional, Dict

    # 将其定义为 TorchScript 类
    @torch.jit.script
    class FunctionalQHM(object):
        def __init__(self,
                    params: List[Tensor],
                    lr: float,
                    momentum: float,
                    nu: float,
                    weight_decay: float = 0.0,
                    weight_decay_type: str = "grad"):
            if lr < 0.0:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if momentum < 0.0:
                raise ValueError("Invalid momentum value: {}".format(momentum))
            if weight_decay < 0.0:
                raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
            if weight_decay_type not in ("grad", "direct"):
                raise ValueError("Invalid weight_decay_type value: {}".format(weight_decay_type))

            self.defaults = {
                "lr": lr,
                "momentum": momentum,
                "nu": nu,
                "weight_decay": weight_decay,
            }
            self.weight_decay_type = weight_decay_type

            # 注意：我们这里只有一个参数组，不允许用户添加额外的参数组，因为这不是常见用例。
            self.param_group = {"params": params}

            self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

        def step(self, gradients: List[Optional[Tensor]]):
            params = self.param_group['params']
            params_with_grad = []
            grads = []
            momentum_buffer_list: List[Tensor] = []

            if len(params) != len(gradients):
                raise ValueError(
                    "传入的梯度数量与参数数量不相等！"
                    + f"参数长度：{len(params)}。"
                    + f"梯度长度：{len(gradients)}"
                )

            for param, gradient in zip(self.param_group['params'], gradients):
                if gradient is not None:
                    params_with_grad.append(param)
                    grads.append(gradient)
                    state = self.state[param]
                    state['momentum_buffer'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    momentum_buffer_list.append(state['momentum_buffer'])

            # 调用我们刚刚定义的更新函数
            with torch.no_grad():
                qhm_update(params_with_grad,
                        grads,
                        momentum_buffer_list,
                        self.defaults['lr'],
                        self.defaults['nu'],
                        self.defaults['weight_decay'],
                        self.weight_decay_type,
                        self.defaults['momentum'])



最后，我们将新定义的分布式函数式优化器注册到 ``functional_optim_map`` 中。
这样 ``DistributedOptimizer`` 就会尝试使用我们的自定义实现，而不是预定义的默认实现。

::

    from torch.distributed.optim import DistributedOptimizer

    DistributedOptimizer.functional_optim_map[QHM] = FunctionalQHM

现在，您可以在分布式训练中正常使用 ``QHM`` 优化器，只需将其传递给
`DistributedOptimizer <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`_


::

    ...
    remote_params_list = [...]
    dist_optim = DistributedOptimizer(
        QHM, remote_params_list, *args, **kwargs
    )

DistributedOptimizer 将自动在底层将 QHM 优化器转换为 ``FunctionalQHM``，
并启用 TorchScript 支持。这将解锁多线程训练带来的性能提升，
并为进一步改进提供更多潜力（例如 TorchScript 融合等）。

请注意，大多数 PyTorch 内置优化器已经使用这种方法来加速分布式训练。
如果您看到有关某些优化器尚未转换的警告，您可以按照本教程编写自己的转换。
