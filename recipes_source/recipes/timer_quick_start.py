"""
Timer快速入门
=================

在本教程中,我们将介绍 `torch.utils.benchmark.Timer` 的主要API。
PyTorch Timer基于 `timeit.Timer <https://docs.python.org/3/library/timeit.html#timeit.Timer>`__ API,
并做了一些PyTorch特定的修改。本教程不要求读者熟悉内置的 `Timer` 类,但假设读者熟悉性能工作的基础知识。

有关更全面的性能调优教程,请参阅 `PyTorch Benchmark <https://pytorch.org/tutorials/recipes/recipes/benchmark.html>`__。


**内容:**
    1. `定义Timer <#defining-a-timer>`__
    2. `Wall时间: Timer.blocked_autorange(...) <#wall-time-timer-blocked-autorange>`__
    3. `C++代码片段 <#c-snippets>`__
    4. `指令计数: Timer.collect_callgrind(...) <#instruction-counts-timer-collect-callgrind>`__
    5. `指令计数: 深入探讨 <#instruction-counts-delving-deeper>`__
    6. `使用Callgrind进行A/B测试 <#a-b-testing-with-callgrind>`__
    7. `总结 <#wrapping-up>`__
    8. `脚注 <#footnotes>`__
"""


###############################################################################
# 1. 定义Timer
# ~~~~~~~~~~~~~~~~~~~
#
# `Timer` 用于定义任务。
#

from torch.utils.benchmark import Timer

timer = Timer(
    # 将在循环中运行并计时的计算。
    stmt="x * y",

    # `setup` 将在调用测量循环之前运行,用于填充 `stmt` 所需的任何状态
    setup="""
        x = torch.ones((128,))
        y = torch.ones((128,))
    """,

    # 或者,可以使用 ``globals`` 从外部作用域传递变量。
    # 
    #    globals={
    #        "x": torch.ones((128,)),
    #        "y": torch.ones((128,)),
    #    },

    # 控制PyTorch使用的线程数。(默认值: 1)
    num_threads=1,
)

###############################################################################
# 2. Wall时间: ``Timer.blocked_autorange(...)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 此方法将处理诸如选择合适的重复次数、固定线程数以及提供结果的方便表示等细节。
#

# Measurement对象存储多次重复的结果,并提供各种实用功能。
from torch.utils.benchmark import Measurement

m: Measurement = timer.blocked_autorange(min_run_time=1)
print(m)

###############################################################################
# .. code-block:: none
#    :caption: **Snippet wall time.**
#
#         <torch.utils.benchmark.utils.common.Measurement object at 0x7f1929a38ed0>
#         x * y
#         setup:
#           x = torch.ones((128,))
#           y = torch.ones((128,))
#
#           Median: 2.34 us
#           IQR:    0.07 us (2.31 to 2.38)
#           424 measurements, 1000 runs per measurement, 1 thread
#

###############################################################################
# 3. C++ 代码片段
# ~~~~~~~~~~~~~~~
#

from torch.utils.benchmark import Language

cpp_timer = Timer(
    "x * y;",
    """
        auto x = torch::ones({128});
        auto y = torch::ones({128});
    """,
    language=Language.CPP,
)

print(cpp_timer.blocked_autorange(min_run_time=1))

###############################################################################
# .. code-block:: none
#    :caption: **C++ snippet wall time.**
#
#         <torch.utils.benchmark.utils.common.Measurement object at 0x7f192b019ed0>
#         x * y;
#         setup:
#           auto x = torch::ones({128});
#           auto y = torch::ones({128});
#
#           Median: 1.21 us
#           IQR:    0.03 us (1.20 to 1.23)
#           83 measurements, 10000 runs per measurement, 1 thread
#

###############################################################################
# 不出所料,C++代码片段的速度更快,变化也更小。
#

###############################################################################
# 4. 指令计数: ``Timer.collect_callgrind(...)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 为了深入调查,`Timer.collect_callgrind` 封装了 
# `Callgrind <https://valgrind.org/docs/manual/cl-manual.html>`__ 以收集指令计数。
# 这些指令计数非常有用,因为它们提供了细粒度和确定性的(或在Python的情况下噪声很低的)见解,
# 说明了代码片段是如何运行的。
#

from torch.utils.benchmark import CallgrindStats, FunctionCounts

stats: CallgrindStats = cpp_timer.collect_callgrind()
print(stats)

###############################################################################
# .. code-block:: none
#    :caption: **C++ Callgrind stats (summary)**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.CallgrindStats object at 0x7f1929a35850>
#         x * y;
#         setup:
#           auto x = torch::ones({128});
#           auto y = torch::ones({128});
#
#                                 All          Noisy symbols removed
#             Instructions:       563600                     563600
#             Baseline:                0                          0
#         100 runs per measurement, 1 thread
#

###############################################################################
# 5. Instruction counts: Delving deeper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The string representation of ``CallgrindStats`` is similar to that of
# Measurement. `Noisy symbols` are a Python concept (removing calls in the
# CPython interpreter which are known to be noisy).
#
# For more detailed analysis, however, we will want to look at specific calls.
# ``CallgrindStats.stats()`` returns a ``FunctionCounts`` object to make this easier.
# Conceptually, ``FunctionCounts`` can be thought of as a tuple of pairs with some
# utility methods, where each pair is `(number of instructions, file path and
# function name)`.
#
# A note on paths:
#   One generally doesn't care about absolute path. For instance, the full path
#   and function name for a multiply call is something like:
#
# 5. 指令计数: 深入探讨
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# `CallgrindStats` 的字符串表示形式类似于 Measurement。`Noisy symbols` 是一个Python概念(移除了在CPython解释器中已知的噪声调用)。
#
# 然而,为了进行更详细的分析,我们需要查看特定的调用。`CallgrindStats.stats()` 返回一个 `FunctionCounts` 对象,以便于此操作。从概念上讲,`FunctionCounts` 可以被视为一个带有一些实用方法的成对元组,其中每一对都是 `(指令数量,文件路径和函数名称)`。
#
# 关于路径的说明:
#   通常我们不关心绝对路径。例如,一个乘法调用的完整路径和函数名是这样的:
#
# .. code-block:: sh
#
#    /the/prefix/to/your/pytorch/install/dir/pytorch/build/aten/src/ATen/core/TensorMethods.cpp:at::Tensor::mul(at::Tensor const&) const [/the/path/to/your/conda/install/miniconda3/envs/ab_ref/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so]
#
#   而实际上,我们感兴趣的所有信息都可以表示为:
#
# .. code-block:: sh
#
#    build/aten/src/ATen/core/TensorMethods.cpp:at::Tensor::mul(at::Tensor const&) const
#
#   ``CallgrindStats.as_standardized()`` 会尽最大努力去除文件路径中低信号部分,以及共享对象,通常建议使用。
#

inclusive_stats = stats.as_standardized().stats(inclusive=False)
print(inclusive_stats[:10])

###############################################################################
# .. code-block:: none
#    :caption: **C++ Callgrind stats (detailed)**
#
#         torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f192a6dfd90>
#           47264  ???:_int_free
#           25963  ???:_int_malloc
#           19900  build/../aten/src/ATen/TensorIter ... (at::TensorIteratorConfig const&)
#           18000  ???:__tls_get_addr
#           13500  ???:malloc
#           11300  build/../c10/util/SmallVector.h:a ... (at::TensorIteratorConfig const&)
#           10345  ???:_int_memalign
#           10000  build/../aten/src/ATen/TensorIter ... (at::TensorIteratorConfig const&)
#            9200  ???:free
#            8000  build/../c10/util/SmallVector.h:a ... IteratorBase::get_strides() const
#
#         Total: 173472
#

###############################################################################
# 这仍然有很多内容需要消化。让我们使用 `FunctionCounts.transform` 方法来去除一些函数路径,并丢弃函数调用。
# 这样做时,任何冲突(例如 `foo.h:a()` 和 `foo.h:b()` 都将映射到 `foo.h`)的计数将被累加。
#

import os
import re

def group_by_file(fn_name: str):
    if fn_name.startswith("???"):
        fn_dir, fn_file = fn_name.split(":")[:2]
    else:
        fn_dir, fn_file = os.path.split(fn_name.split(":")[0])
        fn_dir = re.sub("^.*build/../", "", fn_dir)
        fn_dir = re.sub("^.*torch/", "torch/", fn_dir)

    return f"{fn_dir:<15} {fn_file}"

print(inclusive_stats.transform(group_by_file)[:10])

###############################################################################
# .. code-block:: none
#    :caption: **Callgrind stats (condensed)**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f192995d750>
#           118200  aten/src/ATen   TensorIterator.cpp
#            65000  c10/util        SmallVector.h
#            47264  ???             _int_free
#            25963  ???             _int_malloc
#            20900  c10/util        intrusive_ptr.h
#            18000  ???             __tls_get_addr
#            15900  c10/core        TensorImpl.h
#            15100  c10/core        CPUAllocator.cpp
#            13500  ???             malloc
#            12500  c10/core        TensorImpl.cpp
#
#         Total: 352327
#

###############################################################################
# 6. 使用 ``Callgrind`` 进行A/B测试
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 指令计数最有用的特性之一是允许对计算进行细粒度比较,这在分析性能时至关重要。
#
# 为了看到这一点,让我们将两个大小为128的张量相乘与一个{128} x {1}的乘法进行比较,后者将对第二个张量进行广播:
#   result = {a0 * b0, a1 * b0, ..., a127 * b0}
#

broadcasting_stats = Timer(
    "x * y;",
    """
        auto x = torch::ones({128});
        auto y = torch::ones({1});
    """,
    language=Language.CPP,
).collect_callgrind().as_standardized().stats(inclusive=False)

###############################################################################
# 我们经常需要对两种不同的环境进行A/B测试。(例如测试一个PR,或尝试不同的编译标志。)这很简单,
# 因为 `CallgrindStats`、`FunctionCounts` 和 Measurement 都是可pickle化的。
# 只需在每个环境中保存测量结果,然后在单个进程中加载它们进行分析。
#


import pickle

broadcasting_stats = pickle.loads(pickle.dumps(broadcasting_stats))


delta = broadcasting_stats - inclusive_stats

def extract_fn_name(fn: str):
    """Trim everything except the function name."""
    fn = ":".join(fn.split(":")[1:])
    return re.sub(r"\(.+\)", "(...)", fn)

print(delta.transform(extract_fn_name))


###############################################################################
# .. code-block:: none
#    :caption: **Instruction count delta**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f192995d750>
#             17600  at::TensorIteratorBase::compute_strides(...)
#             12700  at::TensorIteratorBase::allocate_or_resize_outputs()
#             10200  c10::SmallVectorImpl<long>::operator=(...)
#              7400  at::infer_size(...)
#              6200  at::TensorIteratorBase::invert_perm(...) const
#              6064  _int_free
#              5100  at::TensorIteratorBase::reorder_dimensions()
#              4300  malloc
#              4300  at::TensorIteratorBase::compatible_stride(...) const
#               ...
#               -28  _int_memalign
#              -100  c10::impl::check_tensor_options_and_extract_memory_format(...)
#              -300  __memcmp_avx2_movbe
#              -400  at::detail::empty_cpu(...)
#             -1100  at::TensorIteratorBase::numel() const
#             -1300  void at::native::(...)
#             -2400  c10::TensorImpl::is_contiguous(...) const
#             -6100  at::TensorIteratorBase::compute_fast_setup_type(...)
#            -22600  at::TensorIteratorBase::fast_set_up(...)
#
#         Total: 58091
#

###############################################################################
# 所以广播版本每次调用需要额外580条指令(回想一下我们收集了100次运行的样本),约占10%。
# 有相当多的 `TensorIterator` 调用,所以让我们深入研究这些调用。
# `FunctionCounts.filter` 可以很容易地做到这一点。


print(delta.transform(extract_fn_name).filter(lambda fn: "TensorIterator" in fn))

###############################################################################
# .. code-block:: none
#    :caption: **Instruction count delta (filter)**
#
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7f19299544d0>
#             17600  at::TensorIteratorBase::compute_strides(...)
#             12700  at::TensorIteratorBase::allocate_or_resize_outputs()
#              6200  at::TensorIteratorBase::invert_perm(...) const
#              5100  at::TensorIteratorBase::reorder_dimensions()
#              4300  at::TensorIteratorBase::compatible_stride(...) const
#              4000  at::TensorIteratorBase::compute_shape(...)
#              2300  at::TensorIteratorBase::coalesce_dimensions()
#              1600  at::TensorIteratorBase::build(...)
#             -1100  at::TensorIteratorBase::numel() const
#             -6100  at::TensorIteratorBase::compute_fast_setup_type(...)
#            -22600  at::TensorIteratorBase::fast_set_up(...)
#
#         Total: 24000
#

###############################################################################
# 这说明了正在发生的情况:在 TensorIterator 设置中有一条快速路径,
# 但在 {128} x {1} 的情况下,我们错过了它,不得不进行更通用的分析,这更加昂贵。
# 被过滤器省略的最显著的调用是 c10::SmallVectorImpl<long>::operator=(...),
# 这也是更通用设置的一部分。

###############################################################################
# 7. 总结
# ~~~~~~~~~~~~~~
# 总之,使用 Timer.blocked_autorange 来收集墙上时间。如果计时变化过高,
# 请增加 min_run_time,或者如果方便的话,转移到 C++ 代码片段。
# 对于细粒度分析,使用 Timer.collect_callgrind 来测量指令计数,
# 并使用 FunctionCounts.(__add__ / __sub__ / transform / filter)
# 来切分和处理它们。

###############################################################################
# 8. 脚注
# ~~~~~~~~~~~~
# - 隐含的 import torch
# 如果 globals 不包含 "torch",Timer 将自动填充它。这意味着 Timer("torch.empty(())") 将正常工作。
# (不过其他导入应该放在 setup 中,
# 例如 Timer("np.zeros(())", "import numpy as np"))
# - REL_WITH_DEB_INFO
# 为了提供有关执行的 PyTorch 内部信息的完整信息,Callgrind 需要访问 C++ 调试符号。
# 这是通过在构建 PyTorch 时设置 REL_WITH_DEB_INFO=1 来实现的。
# 否则函数调用将是不透明的。(生成的 CallgrindStats 将在缺少调试符号时发出警告。)