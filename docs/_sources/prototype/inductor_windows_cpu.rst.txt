如何在 Windows CPU 上使用 TorchInductor
=======================================

**Author**: `Zhaoqiong Zheng <https://github.com/ZhaoqiongZ>`_, `Xu, Han <https://github.com/xuhancn>`_



TorchInductor 是一个编译器后端，它将 TorchDynamo 生成的 FX 图转换为高度优化的 C++/Triton 内核。
本教程将指导在 Windows CPU 上使用 TorchInductor。

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * 如何在 Windows CPU 上编译和执行使用 PyTorch 的 Python 函数
       * TorchInductor 使用 C++/Triton 内核进行优化的基础知识

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch v2.5 或更高版本
       * Microsoft Visual C++ (MSVC)
       * Windows 版 Miniforge

安装所需软件
-----------------------------

首先，让我们安装所需的软件。TorchInductor 优化需要 C++ 编译器。
在本示例中，我们将使用 Microsoft Visual C++ (MSVC)。

1. 下载并安装 `MSVC <https://visualstudio.microsoft.com/downloads/>`_。

2. 在安装过程中，在 **工作负载** 表中的 **桌面和移动** 部分选择 **使用 C++ 进行桌面开发**。然后安装软件。

.. note::

     我们推荐使用 C++ 编译器 `Clang <https://github.com/llvm/llvm-project/releases>`_ 和 `Intel 编译器 <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_。
     请查看 `更高性能的替代编译器 <#alternative-compiler-for-better-performance>`_。

3. 下载并安装 `Miniforge3-Windows-x86_64.exe <https://github.com/conda-forge/miniforge/releases/latest/>`__。


设置环境
----------------------

#. 通过 ``cmd.exe`` 打开命令行环境。
#. 使用以下命令激活 ``MSVC``:

   .. code-block:: sh

    "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
#. 使用以下命令激活 ``conda``:

   .. code-block:: sh

    "C:/ProgramData/miniforge3/Scripts/activate.bat"
#. 创建并激活conda环境:
 
   .. code-block:: sh

    conda create -n inductor_cpu_windows python=3.10 -y 
    conda activate inductor_cpu_windows

#. 安装 `PyTorch 2.5 <https://pytorch.org/get-started/locally/>`_ 或更新版本。

在 Windows CPU 上使用 TorchInductor
----------------------------------

这里有一个简单的例子来演示如何使用 TorchInductor：

.. code-block:: python


    import torch
    def foo(x, y):
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b
    opt_foo1 = torch.compile(foo)
    print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))

以下是此代码可能返回的示例输出：

.. code-block:: sh

    tensor([[-3.9074e-02,  1.3994e+00,  1.3894e+00,  3.2630e-01,  8.3060e-01,
            1.1833e+00,  1.4016e+00,  7.1905e-01,  9.0637e-01, -1.3648e+00],
            [ 1.3728e+00,  7.2863e-01,  8.6888e-01, -6.5442e-01,  5.6790e-01,
            5.2025e-01, -1.2647e+00,  1.2684e+00, -1.2483e+00, -7.2845e-01],
            [-6.7747e-01,  1.2028e+00,  1.1431e+00,  2.7196e-02,  5.5304e-01,
            6.1945e-01,  4.6654e-01, -3.7376e-01,  9.3644e-01,  1.3600e+00],
            [-1.0157e-01,  7.7200e-02,  1.0146e+00,  8.8175e-02, -1.4057e+00,
            8.8119e-01,  6.2853e-01,  3.2773e-01,  8.5082e-01,  8.4615e-01],
            [ 1.4140e+00,  1.2130e+00, -2.0762e-01,  3.3914e-01,  4.1122e-01,
            8.6895e-01,  5.8852e-01,  9.3310e-01,  1.4101e+00,  9.8318e-01],
            [ 1.2355e+00,  7.9290e-02,  1.3707e+00,  1.3754e+00,  1.3768e+00,
            9.8970e-01,  1.1171e+00, -5.9944e-01,  1.2553e+00,  1.3394e+00],
            [-1.3428e+00,  1.8400e-01,  1.1756e+00, -3.0654e-01,  9.7973e-01,
            1.4019e+00,  1.1886e+00, -1.9194e-01,  1.3632e+00,  1.1811e+00],
            [-7.1615e-01,  4.6622e-01,  1.2089e+00,  9.2011e-01,  1.0659e+00,
            9.0892e-01,  1.1932e+00,  1.3888e+00,  1.3898e+00,  1.3218e+00],
            [ 1.4139e+00, -1.4000e-01,  9.1192e-01,  3.0175e-01, -9.6432e-01,
            -1.0498e+00,  1.4115e+00, -9.3212e-01, -9.0964e-01,  1.0127e+00],
            [ 5.7244e-04,  1.2799e+00,  1.3595e+00,  1.0907e+00,  3.7191e-01,
            1.4062e+00,  1.3672e+00,  6.8502e-02,  8.5216e-01,  8.6046e-01]])

使用替代编译器以获得更好的性能
-------------------------------------------

为了提高 Windows Inductor 的性能，您可以使用 Intel 编译器或 LLVM 编译器。然而，它们依赖于 Microsoft Visual C++ (MSVC) 的运行时库。因此，您的第一步应该是安装 MSVC。

Intel 编译器
^^^^^^^^^^^^^

#. 下载并安装 `Intel 编译器 <https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html>`_ 的 Windows 版本。
#. 使用 CXX 环境变量 ``set CXX=icx-cl`` 设置 Windows Inductor 编译器。

Intel 还提供了一个全面的分步指南，包含性能数据。请查看 `Intel® oneAPI DPC++/C++ Compiler Boosts PyTorch* Inductor Performance on Windows* for CPU Devices <https://www.intel.com/content/www/us/en/developer/articles/technical/boost-pytorch-inductor-performance-on-windows.html>`_。

LLVM 编译器
^^^^^^^^^^^^^

#. 下载并安装 `LLVM 编译器 <https://github.com/llvm/llvm-project/releases>`_ 并选择 win64 版本。
#. 使用 CXX 环境变量 ``set CXX=clang-cl`` 设置 Windows Inductor 编译器。

结论
----------

在本教程中，我们学习了如何在 Windows CPU 上使用 PyTorch 的 Inductor。此外，我们还讨论了使用 Intel 编译器和 LLVM 编译器进一步提高性能的方法。