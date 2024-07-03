"""
从检查点加载 ``nn.Module`` 的技巧
===================================================
**作者:** `Mikayla Gawarecki <https://github.com/mikaylagawarecki>`_

如果你要加载一个检查点并希望尽可能减少计算和内存的使用，本教程将分享一些推荐的做法。特别是我们将讨论以下几点:

1. ``torch.load`` 中的 ``mmap`` 关键字参数
2. ``torch.device()`` 上下文管理器
3. ``nn.Module.load_state_dict()`` 中的 ``assign`` 关键字参数

.. note::
   本教程需要 PyTorch 2.1.0 或更高版本。
"""

import time

###############################################################################
# 让我们考虑一个简单的 ``nn.Module``，它包含一个线性层列表:
import torch
from torch import nn


class SomeModule(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(size, size) for i in range(10)])

    def forward(self, x):
        return self.linears(x)


m = SomeModule(1000)
torch.save(m.state_dict(), "checkpoint.pth")

#################################################################################
# 以下代码片段演示了如何使用 ``torch.load`` 中的 ``mmap`` 关键字参数、``torch.device()`` 上下文管理器和 ``nn.Module.load_state_dict()`` 中的 ``assign`` 关键字参数。

state_dict = torch.load("checkpoint.pth", mmap=True)
with torch.device("meta"):
    meta_m = SomeModule(1000)
meta_m.load_state_dict(state_dict, assign=True)

#############################################################################
# 将下面的代码片段与上面的进行比较:

state_dict = torch.load("checkpoint.pth")
m = SomeModule(1000)
m.load_state_dict(state_dict)

#############################################################################
# 第二个示例没有使用上面列出的任何特性，因此在加载检查点时计算和内存效率会较低。在下面的部分中，我们将详细讨论每个特性。

#####################################################################################
# 使用 ``torch.load(mmap=True)``
# -------------------------------
# 首先，让我们考虑使用 ``torch.load`` 加载检查点时会发生什么。
# 当我们使用 ``torch.save`` 保存检查点时，张量存储会被标记为保存时所在的设备。
# 使用 ``torch.load`` 时，张量存储将被加载到它们被标记的设备上(除非使用 ``map_location`` 标志覆盖此行为)。
# 为了解释方便，我们假设张量是保存在 CPU 上的。这意味着在第一行中，所有张量存储将被加载到 CPU 内存中，在以下情况下这是不可取的:

# * CPU 内存小于检查点的大小。
# * 在执行一些每张量处理之前等待整个检查点被加载到内存中。

start_time = time.time()
state_dict = torch.load("checkpoint.pth")
end_time = time.time()
print(f"不使用 mmap 的加载时间={end_time - start_time}")

#################################################################################
# ``torch.load`` 中的 ``mmap`` 关键字参数试图解决上述两个问题。
# 顾名思义，``torch.load`` 中的 ``mmap`` 关键字参数使用了 `mmap 调用 <https://man7.org/linux/man-pages/man2/mmap.2.html>`_,
# 它将磁盘上的文件映射到虚拟内存中,并让操作系统自动处理加载和卸载到物理内存。
# 当传递此标志时,张量存储将被内存映射。

start_time = time.time()
state_dict = torch.load("checkpoint.pth", mmap=True)
end_time = time.time()
print(f"使用 mmap 的加载时间={end_time - start_time}")


######################################################################################
# 如上所述,可以使用此参数在不将所有张量存储加载到 CPU 内存中的情况下对检查点执行每张量处理。例如:
def my_special_routine(t, device):
    # 这可能是一个更复杂的操作
    return t.to(dtype=torch.bfloat16, device=device)


def my_processing_function(key, device):
    t = state_dict[key]
    processed_t = my_special_routine(t, device)
    del t
    state_dict[key] = processed_t


for key in state_dict.keys():
    device = torch.device("cuda")
    my_processing_function(key, device)

##################################################
# 使用 ``torch.device('meta')``
# ------------------------------
# 接下来,让我们考虑模块的创建。
m = SomeModule(1000)

#######################################################################################################
# 这将为所有参数/缓冲区分配内存并根据 ``SomeModule.__init__()`` 中定义的默认初始化方案对其进行初始化,
# 当我们想要加载检查点时,这是浪费的,原因如下:

# * 初始化内核的结果将被 ``load_state_dict()`` 覆盖而从未被使用,因此初始化是浪费的。
# * 我们在 RAM 中为这些参数/缓冲区分配了内存,而 ``torch.load`` 保存的状态字典也在 RAM 中为检查点中的参数/缓冲区分配了内存。

# 为了解决这两个问题,我们可以在实例化 ``nn.Module()`` 时使用 ``device='meta'`` 的 ``torch.device()`` 上下文管理器。

# `torch.device() <https://pytorch.org/docs/main/tensor_attributes.html#torch-device>`_
# 上下文管理器确保工厂调用将被视为传递了指定的 ``device`` 作为参数。
# 在 ``torch.device('meta')`` 上的张量不携带数据。
# 但是,它们具有张量所携带的其他元数据,如 ``.size()``, ``.stride()``, ``.requires_grad`` 等。
with torch.device("meta"):
    new_m = SomeModule(1000)

########################################################
# 使用 ``load_state_dict(assign=True)``
# --------------------------------------
# 接下来,我们考虑加载状态字典。

m.load_state_dict(state_dict)

######################################################################################
# ``nn.Module.load_state_dict()`` 通常是通过 ``param_in_model.copy_(param_in_state_dict)`` 的就地复制实现的。
# 这意味着状态字典中对应键的参数/缓冲区将被复制到 ``nn.Module`` 中的参数/缓冲区。

# 然而,对 ``meta`` 设备上的张量进行就地复制是无操作的。
# 为了避免这种情况,我们可以在 ``load_state_dict()`` 中传递 ``assign=True`` 关键字参数。

# 这里的一个警告是,由于优化器持有对 ``nn.Module.parameters()`` 的引用,
# 如果传递了 ``assign=True``,则必须在从状态字典加载模块后初始化优化器。

# 从 PyTorch 2.3.0 开始,可以使用 ``torch.__future__.set_swap_module_params_on_conversion`` 来避免这个警告。
# 这个 `教程 <https://pytorch.org/tutorials/recipes/recipes/swap_tensors.html>`_ 提供了更多细节。

new_m.load_state_dict(state_dict, assign=True)
# 在 2.3.0 之前,这一步必须在 load_state_dict 使用 assign 之后完成。
# 在版本 >= 2.3.0 中,可以考虑设置 ``torch.__future__.set_swap_module_params_on_conversion``
opt = torch.optim.SGD(new_m.parameters(), lr=1e-3)

###############################################################################
# 结论
# -------------
#
# 总结一下,在本教程中,我们学习了 ``torch.load(mmap=True)``、``device='meta'`` 的 ``torch.device()`` 上下文管理器和 ``nn.Module.load_state_dict(assign=True)``
# 以及如何在从检查点加载模型时使用这些工具来提高效率。
