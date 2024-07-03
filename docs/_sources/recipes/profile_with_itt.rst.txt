使用 Instrumentation and Tracing Technology (ITT) API 分析 PyTorch 工作负载
=====================================================================================

在本教程中,您将学习:

* 什么是 Intel® VTune™ Profiler
* 什么是 Instrumentation and Tracing Technology (ITT) API
* 如何在 Intel® VTune™ Profiler 中可视化 PyTorch 模型层次结构
* 一个简短的示例代码,展示如何使用 PyTorch ITT API


要求
------------

* PyTorch 1.13 或更高版本
* Intel® VTune™ Profiler

安装 PyTorch 的说明可在 `pytorch.org <https://pytorch.org/get-started/locally/>`__ 上找到。


什么是 Intel® VTune™ Profiler
------------------------------

Intel® VTune™ Profiler 是一款用于串行和多线程应用程序的性能分析工具。对于熟悉 Intel 架构的人来说,Intel® VTune™ Profiler 提供了丰富的指标集,帮助用户了解应用程序在 Intel 平台上的执行情况,从而了解性能瓶颈所在。

更多详细信息,包括入门指南,可在 `Intel 网站 <https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html>`__ 上找到。

什么是 Instrumentation and Tracing Technology (ITT) API
--------------------------------------------------------

`Instrumentation and Tracing Technology API (ITT API) <https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/api-support/instrumentation-and-tracing-technology-apis.html>`_ 由 Intel® VTune™ Profiler 提供,使目标应用程序能够在执行期间生成和控制跟踪数据的收集。

ITT 功能的优势在于能够在 Intel® VTune™ Profiler GUI 上标记单个 PyTorch 算子和自定义区域的时间跨度。当用户发现任何异常时,这将非常有助于定位哪个算子表现异常。

.. note::

   ITT API 已在 PyTorch 1.13 中集成。用户无需调用原始的 ITT C/C++ API,只需调用 PyTorch 中的 Python API 即可。更多详细信息可在 `PyTorch 文档 <https://pytorch.org/docs/stable/profiler.html#intel-instrumentation-and-tracing-technology-apis>`__ 中找到。

如何在 Intel® VTune™ Profiler 中可视化 PyTorch 模型层次结构
------------------------------------------------------------------

PyTorch 提供了两种使用方式:

1. 隐式调用: 默认情况下,所有通过 PyTorch 算子注册机制注册的算子在启用 ITT 功能时都会自动标记。

2. 显式调用: 如果需要自定义标记,用户可以在 `PyTorch 文档 <https://pytorch.org/docs/stable/profiler.html#intel-instrumentation-and-tracing-technology-apis>`__ 中使用显式 API 对所需范围进行标记。


要启用显式调用,需要在 `torch.autograd.profiler.emit_itt()` 作用域下调用预期标记的代码。例如:

.. code:: python3

   with torch.autograd.profiler.emit_itt():
     <code-to-be-profiled...>

启动 Intel® VTune™ Profiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

要验证功能,您需要启动一个 Intel® VTune™ Profiler 实例。启动 Intel® VTune™ Profiler 的步骤请查看 `Intel® VTune™ Profiler 用户指南 <https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/launch.html>`__。

一旦启动了 Intel® VTune™ Profiler GUI,您应该会看到如下用户界面:

.. figure:: /_static/img/itt_tutorial/vtune_start.png
   :width: 100%
   :align: center

左侧导航栏下的 `sample (matrix)` 项目中有三个示例结果。如果您不希望分析结果出现在此默认示例项目中,可以通过蓝色 `Configure Analysis...` 按钮下的 `New Project...` 按钮创建一个新项目。要启动新的分析,请单击蓝色的 `Configure Analysis...` 按钮以开始配置分析。

配置分析
~~~~~~~~~~~~~~~~~~~

单击 `Configure Analysis...` 按钮后,您应该会看到如下界面:

.. figure:: /_static/img/itt_tutorial/vtune_config.png
   :width: 100%
   :align: center

窗口的右侧分为三部分: `WHERE`(左上角)、`WHAT`(左下角)和 `HOW`(右侧)。在 `WHERE` 中,您可以指定要在哪台机器上运行分析。在 `WHAT` 中,您可以设置要分析的应用程序的路径。要分析 PyTorch 脚本,建议将所有手动步骤(包括激活 Python 环境和设置所需环境变量)封装到一个 bash 脚本中,然后对该 bash 脚本进行分析。在上面的截图中,我们将所有步骤封装到 `launch.sh` bash 脚本中,并将 `bash` 的参数设置为 `<path_of_launch.sh>` 的路径。在右侧的 `HOW` 中,您可以选择要分析的类型。Intel® VTune™ Profiler 提供了多种可选的分析类型。详情请查看 `Intel® VTune™ Profiler 用户指南 <https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance.html>`__。

读取分析结果
~~~~~~~~~~~~~~~~~~~~~

成功进行了带有 ITT 的分析后,您可以打开分析结果的 `Platform` 选项卡,在 Intel® VTune™ Profiler 时间线上查看标记。

.. figure:: /_static/img/itt_tutorial/vtune_timeline.png
   :width: 100%
   :align: center


时间线显示了顶部的主线程作为 `python` 线程,下面是各个 OpenMP 线程。标记的 PyTorch 算子和自定义区域显示在主线程行中。所有以 `aten::` 开头的算子都是由 PyTorch 中的 ITT 功能隐式标记的。标签 `iteration_N` 是使用特定的 API `torch.profiler.itt.range_push()`、`torch.profiler.itt.range_pop()` 或 `torch.profiler.itt.range()` 作用域显式标记的。请查看下一节中的示例代码以了解详情。

.. note::

   时间线中标记为 `convolution` 和 `reorder` 的红色框是由 Intel® oneAPI Deep Neural Network Library (oneDNN) 标记的。

如右侧导航栏所示,时间线行中的棕色部分显示了各个线程的 CPU 使用情况。在某个时间点,棕色部分在线程行中所占的高度百分比与该线程在该时间点的 CPU 使用率相对应。因此,从这个时间线可以直观地了解以下几点:

1. 每个线程的 CPU 核心利用率如何。
2. 所有线程的 CPU 核心利用率是否平衡。所有线程的 CPU 使用情况是否良好?
3. OpenMP 线程是否同步良好。启动 OpenMP 线程或 OpenMP 线程完成时是否存在抖动?

当然,Intel® VTune™ Profiler 还提供了更多丰富的分析功能,帮助您了解性能问题的根源。一旦您了解了性能问题的根源,就可以加以修复。更多详细的使用说明可在 `Intel® VTune™ Profiler 用户指南 <https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance.html>`__ 中找到。

一个简短的示例代码,展示如何使用 PyTorch ITT API
----------------------------------------------------------

下面的示例代码就是在上面的截图中用于分析的脚本。

该拓扑由两个算子 `Conv2d` 和 `Linear` 组成。进行了三次推理迭代,每次迭代都使用 PyTorch ITT API 标记为文本字符串 `iteration_N`。无论是使用 `torch.profile.itt.range_push` 和 `torch.profile.itt.range_pop` 的配对,还是使用 `torch.profile.itt.range` 作用域,都可以实现自定义标记功能。

.. code:: python3

   # sample.py

   import torch
   import torch.nn as nn
   
   class ITTSample(nn.Module):
     def __init__(self):
       super(ITTSample, self).__init__()
       self.conv = nn.Conv2d(3, 5, 3)
       self.linear = nn.Linear(292820, 1000)
   
     def forward(self, x):
       x = self.conv(x)
       x = x.view(x.shape[0], -1)
       x = self.linear(x)
       return x
   
   def main():
     m = ITTSample()
     x = torch.rand(10, 3, 244, 244)
     with torch.autograd.profiler.emit_itt():
       for i in range(3)
         # 使用 range_push 和 range_pop 配对标记区域
         #torch.profiler.itt.range_push(f'iteration_{i}')
         #m(x)
         #torch.profiler.itt.range_pop()
   
         # 使用 range 作用域标记区域
         with torch.profiler.itt.range(f'iteration_{i}'):
           m(x)
   
   if __name__ == '__main__':
     main()


下面是在 Intel® VTune™ Profiler GUI 截图中提到的 `launch.sh` bash 脚本,用于封装所有手动步骤。

.. code:: bash

   # launch.sh

   #!/bin/bash
   
   # 获取包含 sample.py 和 launch.sh 的目录路径,以便从任何目录调用此 bash 脚本
   BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
   <激活 Python 环境>
   cd ${BASEFOLDER}
   python sample.py
