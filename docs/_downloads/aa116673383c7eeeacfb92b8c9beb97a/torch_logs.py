"""
(Beta) 使用 TORCH_LOGS python API 与 torch.compile
==========================================================================================
**作者:** `Michael Lazos <https://github.com/mlazos>`_
"""

import logging

######################################################################
#
# 本教程介绍了 ``TORCH_LOGS`` 环境变量以及 Python API,并演示了如何将其应用于观察 ``torch.compile`` 的各个阶段。
#
# .. note::
#
#   本教程需要 PyTorch 2.2.0 或更高版本。
#
#


######################################################################
# 设置
# ~~~~~~~~~~~~~~~~~~~~~
# 在这个例子中,我们将设置一个简单的 Python 函数,执行元素级加法,并使用 ``TORCH_LOGS`` Python API 观察编译过程。
#
# .. note::
#
#   还有一个环境变量 ``TORCH_LOGS``,可用于在命令行中更改日志设置。每个示例都显示了等效的环境变量设置。

import torch

# 如果设备不支持 torch.compile,则干净地退出
if torch.cuda.get_device_capability() < (7, 0):
    print("跳过,因为此设备不支持 torch.compile。")
else:

    @torch.compile()
    def fn(x, y):
        z = x + y
        return z + 2

    inputs = (torch.ones(2, 2, device="cuda"), torch.zeros(2, 2, device="cuda"))

    # 在每个示例之间打印分隔符并重置 dynamo
    def separator(name):
        print(f"==================={name}=========================")
        torch._dynamo.reset()

    separator("Dynamo 跟踪")
    # 查看 dynamo 跟踪
    # TORCH_LOGS="+dynamo"
    torch._logging.set_logs(dynamo=logging.DEBUG)
    fn(*inputs)

    separator("跟踪的图形")
    # 查看跟踪的图形
    # TORCH_LOGS="graph"
    torch._logging.set_logs(graph=True)
    fn(*inputs)

    separator("融合决策")
    # 查看融合决策
    # TORCH_LOGS="fusion"
    torch._logging.set_logs(fusion=True)
    fn(*inputs)

    separator("输出代码")
    # 查看 inductor 生成的输出代码
    # TORCH_LOGS="output_code"
    torch._logging.set_logs(output_code=True)
    fn(*inputs)

    separator("")

######################################################################
# 结论
# ~~~~~~~~~~
#
# 在本教程中,我们介绍了 TORCH_LOGS 环境变量和 python API,并通过实验了一小部分可用的日志选项。
# 要查看所有可用选项的描述,请运行任何导入 torch 的 python 脚本,并将 TORCH_LOGS 设置为 "help"。
#
# 或者,您可以查看 `torch._logging 文档`_ 以查看所有可用日志选项的描述。
#
# 有关 torch.compile 的更多信息,请参阅 `torch.compile 教程`_。
#
# .. _torch._logging 文档: https://pytorch.org/docs/main/logging.html
# .. _torch.compile 教程: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
