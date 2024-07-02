"""
PyTorch Benchmark
====================================
本教程提供了使用 PyTorch ``benchmark`` 模块来测量和比较代码性能的快速入门指南。

介绍
------------
基准测试是编写代码时的一个重要步骤。它帮助我们验证代码是否满足性能预期,比较解决同一问题的不同方法,并防止性能裂化。

对于基准测试 PyTorch 代码有许多选择,包括 Python 内置的 ``timeit`` 模块。
然而,基准测试 PyTorch 代码有许多容易被忽视的注意事项,例如管理线程数量和同步 CUDA 设备。
此外,为基准测试生成张量输入可能相当繁琐。

本教程演示了如何使用 PyTorch ``benchmark`` 模块来避免常见错误,同时更容易比较不同代码的性能、为基准测试生成输入等。

设置
-----
在开始之前,如果尚未安装 ``torch``,请先安装。

::

   pip install torch

"""

######################################################################
# 具体步骤
# -----
#
# 1. 定义要基准测试的函数
# 2. 使用 ``timeit.Timer`` 进行基准测试
# 3. 使用 ``torch.utils.benchmark.Timer`` 进行基准测试
# 4. 使用 ``Blocked Autorange`` 进行基准测试
# 5. 比较基准测试结果
# 6. 保存/加载基准测试结果
# 7. 使用 ``Fuzzed Parameters`` 生成输入
# 8. 使用 ``Callgrind`` 收集指令计数
#
# 1. 定义要基准测试的函数
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 在撰写本文时, `torch.dot <https://pytorch.org/docs/stable/generated/torch.dot.html?highlight=dot#torch.dot>`__
# 不支持批量模式,因此我们将比较使用现有 ``torch`` 运算符实现它的两种方法:一种方法使用 ``mul`` 和 ``sum`` 的组合,另一种方法使用 ``bmm``。
#

import torch


def batched_dot_mul_sum(a, b):
    """Computes batched dot by multiplying and summing"""
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    """Computes batched dot by reducing to ``bmm``"""
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


# Input for benchmarking
x = torch.randn(10000, 64)

# Ensure that both functions compute the same output
assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))


######################################################################
# 2. 使用 ``timeit.Timer`` 进行基准测试
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 首先,让我们使用 Python 内置的 ``timeit`` 模块对代码进行基准测试。
# 我们在这里保持基准测试代码简单,以便我们可以比较 ``timeit`` 和 ``torch.utils.benchmark`` 的默认设置。
#

import timeit

t0 = timeit.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
)

t1 = timeit.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
)

print(f"mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us")
print(f"bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us")

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     mul_sum(x, x):  111.6 us
#     bmm(x, x):       70.0 us
#


######################################################################
# 3. 使用 ``torch.utils.benchmark.Timer`` 进行基准测试
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch ``benchmark``模块的设计使得对于那些曾经使用过 ``timeit`` 模块的人来说,它看起来很熟悉。
# 然而,它的默认设置使得它更容易且更安全地用于对 PyTorch 代码进行基准测试。
# 首先让我们对比一下基本API的使用。

import torch.utils.benchmark as benchmark

t0 = benchmark.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
)

t1 = benchmark.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
)

print(t0.timeit(100))
print(t1.timeit(100))

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb10400d0f0>
#     batched_dot_mul_sum(x, x)
#     setup: from __main__ import batched_dot_mul_sum
#       379.29 us
#       1 measurement, 100 runs , 1 thread
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb103d67048>
#     batched_dot_bmm(x, x)
#     setup: from __main__ import batched_dot_bmm
#       716.42 us
#       1 measurement, 100 runs , 1 thread
#

######################################################################
# 虽然基本功能的API是相同的,但是还是有一些重要的区别。
# ``benchmark.Timer.timeit()``返回的是每次运行的时间,而不是 ``timeit.Timer.timeit()`` 返回的总运行时间。
# PyTorch ``benchmark``模块还提供了格式化的字符串表示,用于打印结果。
#
# 另一个重要的区别,也是结果不同的原因,是PyTorch基准测试模块默认在单线程中运行。
# 我们可以使用``num_threads``参数来更改线程数量。
#
# ``torch.utils.benchmark.Timer``接受几个额外的参数,包括: ``label``、``sub_label``、``description``和``env``,
# 这些参数会改变返回的测量对象的__repr__,并用于对结果进行分组(稍后会详细介绍)。
#

num_threads = torch.get_num_threads()
print(f"Benchmarking on {num_threads} threads")

t0 = benchmark.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
    num_threads=num_threads,
    label="Multithreaded batch dot",
    sub_label="Implemented using mul and sum",
)

t1 = benchmark.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
    num_threads=num_threads,
    label="Multithreaded batch dot",
    sub_label="Implemented using bmm",
)

print(t0.timeit(100))
print(t1.timeit(100))

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     Benchmarking on 40 threads
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb103d54080>
#     Multithreaded batch dot: Implemented using mul and sum
#     setup: from __main__ import batched_dot_mul_sum
#       118.47 us
#       1 measurement, 100 runs , 40 threads
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb16935d2e8>
#     Multithreaded batch dot: Implemented using bmm
#     setup: from __main__ import batched_dot_bmm
#       68.21 us
#       1 measurement, 100 runs , 40 threads

######################################################################
# 使用所有可用线程运行 ``benchmark`` 会得到与 ``timeit`` 模块类似的结果。
# 更重要的是,哪个版本更快取决于我们使用多少线程运行代码。
# 这就是为什么在基准测试时,使用与实际用例相符的线程设置非常重要。
# 另一个需要记住的重要事情是,在 GPU 上进行基准测试时,要同步CPU和CUDA。
# 让我们再次在CUDA张量上运行上面的基准测试,看看会发生什么。
#

x = torch.randn(10000, 1024, device="cuda")

t0 = timeit.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
)

t1 = timeit.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
)

# Ran each twice to show difference before/after warm-up
print(f"mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us")
print(f"mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us")
print(f"bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us")
print(f"bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us")

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     mul_sum(x, x):   27.6 us
#     mul_sum(x, x):   25.3 us
#     bmm(x, x):      2775.5 us
#     bmm(x, x):       22.4 us
#

t0 = benchmark.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
)

t1 = benchmark.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
)

# Run only once since benchmark module does warm-up for us
print(t0.timeit(100))
print(t1.timeit(100))

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb10400d080>
#     batched_dot_mul_sum(x, x)
#     setup: from __main__ import batched_dot_mul_sum
#       232.93 us
#       1 measurement, 100 runs , 1 thread
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb10400d0f0>
#     batched_dot_bmm(x, x)
#     setup: from __main__ import batched_dot_bmm
#       181.04 us
#       1 measurement, 100 runs , 1 thread
#

######################################################################
# 结果揭示了一些有趣的事情。使用 `timeit` 模块运行 `bmm` 版本的第一次运行比第二次运行慢很多。
# 这是因为 `bmm` 需要调用 `cuBLAS`,第一次调用时需要加载它,这需要一些时间。
# 这就是为什么在基准测试之前做一次预热运行很重要,幸运的是, PyTorch 的 `benchmark` 模块为我们处理了这个问题。
#
# `timeit` 模块和 `benchmark` 模块之间结果的差异是因为 `timeit` 模块没有同步 CUDA,因此只计时了启动内核的时间。
# PyTorch 的 `benchmark` 模块为我们做了同步。


######################################################################
# 4. 使用 `Blocked Autorange` 进行基准测试
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 虽然 `timeit.Timer.autorange` 采取至少 0.2 秒的单次连续测量,
# 但 `torch.utils.benchmark.blocked_autorange` 采取多次测量,其总时间至少为 0.2 秒(可通过 `min_run_time` 参数更改),
# 并且测量开销只占总体测量的一小部分。
# 这是通过首先以递增的循环次数运行,直到运行时间远大于测量开销(这也起到了热身的作用),
# 然后进行测量直到达到目标时间。这有一个有用的特性,即它浪费的数据更少,并且允许我们计算统计数据来估计测量的可靠性。
#

m0 = t0.blocked_autorange()
m1 = t1.blocked_autorange()

print(m0)
print(m1)

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb10400d0f0>
#     batched_dot_mul_sum(x, x)
#     setup: from __main__ import batched_dot_mul_sum
#       231.79 us
#       1 measurement, 1000 runs , 1 thread
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb10400d080>
#     batched_dot_bmm(x, x)
#     setup: from __main__ import batched_dot_bmm
#       Median: 162.08 us
#       2 measurements, 1000 runs per measurement, 1 thread
#

######################################################################
# 我们还可以查看返回的测量对象中获得的各个统计数据。

print(f"Mean:   {m0.mean * 1e6:6.2f} us")
print(f"Median: {m0.median * 1e6:6.2f} us")

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     Mean:   231.79 us
#     Median: 231.79 us
#

######################################################################
# 5. 比较基准测试结果
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 到目前为止,我们一直在比较我们的两个批量点积版本对同一输入的表现。
# 在实践中,我们希望尝试不同的输入组合以及不同的线程数量。
# `Compare` 类帮助我们以格式化表格的形式显示多个测量结果。
# 它使用上面描述的注释( `label`、 `sub_label`、 `num_threads` 等)以及 `description` 来对表格进行分组和组织。
# 让我们使用 `Compare` 来看看我们的函数在不同的输入大小和线程数量下的表现如何。
#

from itertools import product

# Compare takes a list of measurements which we'll save in results.
results = []

sizes = [1, 64, 1024, 10000]
for b, n in product(sizes, sizes):
    # label and sub_label are the rows
    # description is the column
    label = "Batched dot"
    sub_label = f"[{b}, {n}]"
    x = torch.ones((b, n))
    for num_threads in [1, 4, 16, 32]:
        results.append(
            benchmark.Timer(
                stmt="batched_dot_mul_sum(x, x)",
                setup="from __main__ import batched_dot_mul_sum",
                globals={"x": x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description="mul/sum",
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt="batched_dot_bmm(x, x)",
                setup="from __main__ import batched_dot_bmm",
                globals={"x": x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description="bmm",
            ).blocked_autorange(min_run_time=1)
        )

compare = benchmark.Compare(results)
compare.print()

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     [--------------- Batched dot ----------------]
#                           |  mul/sum   |    bmm
#     1 threads: -----------------------------------
#           [1, 1]          |       5.9  |      11.2
#           [1, 64]         |       6.4  |      11.4
#           [1, 1024]       |       6.7  |      14.2
#           [1, 10000]      |      10.2  |      23.7
#           [64, 1]         |       6.3  |      11.5
#           [64, 64]        |       8.6  |      15.4
#           [64, 1024]      |      39.4  |     204.4
#           [64, 10000]     |     274.9  |     748.5
#           [1024, 1]       |       7.7  |      17.8
#           [1024, 64]      |      40.3  |      76.4
#           [1024, 1024]    |     432.4  |    2795.9
#           [1024, 10000]   |   22657.3  |   11899.5
#           [10000, 1]      |      16.9  |      74.8
#           [10000, 64]     |     300.3  |     609.4
#           [10000, 1024]   |   23098.6  |   27246.1
#           [10000, 10000]  |  267073.7  |  118823.7
#     4 threads: -----------------------------------
#           [1, 1]          |       6.0  |      11.5
#           [1, 64]         |       6.2  |      11.2
#           [1, 1024]       |       6.8  |      14.3
#           [1, 10000]      |      10.2  |      23.7
#           [64, 1]         |       6.3  |      16.2
#           [64, 64]        |       8.8  |      18.2
#           [64, 1024]      |      41.5  |     189.1
#           [64, 10000]     |      91.7  |     849.1
#           [1024, 1]       |       7.6  |      17.4
#           [1024, 64]      |      43.5  |      33.5
#           [1024, 1024]    |     135.4  |    2782.3
#           [1024, 10000]   |    7471.1  |   11874.0
#           [10000, 1]      |      16.8  |      33.9
#           [10000, 64]     |     118.7  |     173.2
#           [10000, 1024]   |    7264.6  |   27824.7
#           [10000, 10000]  |  100060.9  |  121499.0
#     16 threads: ----------------------------------
#           [1, 1]          |       6.0  |      11.3
#           [1, 64]         |       6.2  |      11.2
#           [1, 1024]       |       6.9  |      14.2
#           [1, 10000]      |      10.3  |      23.8
#           [64, 1]         |       6.4  |      24.1
#           [64, 64]        |       9.0  |      23.8
#           [64, 1024]      |      54.1  |     188.5
#           [64, 10000]     |      49.9  |     748.0
#           [1024, 1]       |       7.6  |      23.4
#           [1024, 64]      |      55.5  |      28.2
#           [1024, 1024]    |      66.9  |    2773.9
#           [1024, 10000]   |    6111.5  |   12833.7
#           [10000, 1]      |      16.9  |      27.5
#           [10000, 64]     |      59.5  |      73.7
#           [10000, 1024]   |    6295.9  |   27062.0
#           [10000, 10000]  |   71804.5  |  120365.8
#     32 threads: ----------------------------------
#           [1, 1]          |       5.9  |      11.3
#           [1, 64]         |       6.2  |      11.3
#           [1, 1024]       |       6.7  |      14.2
#           [1, 10000]      |      10.5  |      23.8
#           [64, 1]         |       6.3  |      31.7
#           [64, 64]        |       9.1  |      30.4
#           [64, 1024]      |      72.0  |     190.4
#           [64, 10000]     |     103.1  |     746.9
#           [1024, 1]       |       7.6  |      28.4
#           [1024, 64]      |      70.5  |      31.9
#           [1024, 1024]    |      65.6  |    2804.6
#           [1024, 10000]   |    6764.0  |   11871.4
#           [10000, 1]      |      17.8  |      31.8
#           [10000, 64]     |     110.3  |      56.0
#           [10000, 1024]   |    6640.2  |   27592.2
#           [10000, 10000]  |   73003.4  |  120083.2
#
#     Times are in microseconds (us).
#

######################################################################
# 上面的结果表明,对于在多线程上运行的较大张量, `bmm` 的版本效果更好,
# 而对于较小和/或单线程代码,另一个版本效果更好。
#
# `Compare` 还提供了用于更改表格格式的函数

compare.trim_significant_figures()
compare.colorize()
compare.print()


######################################################################
# 6. 保存/加载基准测试结果
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# `Measurements` (和第8节中描述的 `CallgrindStats` )可以通过 `pickle` 模块序列化。
# 这使得A/B测试变得很容易,因为您可以从两个独立的环境中收集测量结果,
# 将它们序列化,然后在单个环境中加载两者。Timer甚至接受一个 `env`
# 构造函数参数,以便这种A/B测试可以无缝衔接。
#
# 假设 add/sum 和 `bmm` 方法不是两个Python函数,而是 PyTorch 的两个不同版本。
# 下面的示例演示了如何进行A/B测试。为了简单起见,我们只使用了一部分数据,
# 并简单地通过pickle来回传结果,而不是实际使用多个环境并将结果写入磁盘。
#

import pickle

ab_test_results = []
for env in ("environment A: mul/sum", "environment B: bmm"):
    for b, n in ((1, 1), (1024, 10000), (10000, 1)):
        x = torch.ones((b, n))
        dot_fn = (
            batched_dot_mul_sum if env == "environment A: mul/sum" else batched_dot_bmm
        )
        m = benchmark.Timer(
            stmt="batched_dot(x, x)",
            globals={"x": x, "batched_dot": dot_fn},
            num_threads=1,
            label="Batched dot",
            description=f"[{b}, {n}]",
            env=env,
        ).blocked_autorange(min_run_time=1)
        ab_test_results.append(pickle.dumps(m))

ab_results = [pickle.loads(i) for i in ab_test_results]
compare = benchmark.Compare(ab_results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     [------------------------------------- Batched dot -------------------------------------]
#                                                    |  [1, 1]  |  [1024, 10000]  |  [10000, 1]
#     1 threads: ------------------------------------------------------------------------------
#       (environment A: mul/sum)  batched_dot(x, x)  |     7    |      36000      |      21
#       (environment B: bmm)      batched_dot(x, x)  |    14    |      40000      |      85
#
#     Times are in microseconds (us).
#

# 仅为展示可以将之前所有的结果通过 pickle 进行回传:
round_tripped_results = pickle.loads(pickle.dumps(results))
assert str(benchmark.Compare(results)) == str(benchmark.Compare(round_tripped_results))


######################################################################
# 7. 使用 `Fuzzed Parameters` 生成输入
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 正如我们在上一节中看到的,根据输入张量的不同,性能差异可能会很大。
# 因此,在多个不同的输入上运行基准测试是一个好主意。
# 但是,创建所有这些输入张量可能会很麻烦,这就是 `torch.utils.benchmark.Fuzzer`
# 和相关类的用武之地。让我们看看如何使用 `Fuzzer` 来创建一些用于基准测试的测试用例。
#

from torch.utils.benchmark import FuzzedParameter, FuzzedTensor, Fuzzer, ParameterAlias

# 生成随机张量,元素数量在 128 到 10000000 之间,大小 k0 和 k1 从 [1, 10000] 的 `loguniform` 分布中选择,
# 其中平均 40% 将是不连续的。
example_fuzzer = Fuzzer(
    parameters=[
        FuzzedParameter("k0", minval=1, maxval=10000, distribution="loguniform"),
        FuzzedParameter("k1", minval=1, maxval=10000, distribution="loguniform"),
    ],
    tensors=[
        FuzzedTensor(
            "x",
            size=("k0", "k1"),
            min_elements=128,
            max_elements=10000000,
            probability_contiguous=0.6,
        )
    ],
    seed=0,
)

results = []
for tensors, tensor_params, params in example_fuzzer.take(10):
    # description is the column label
    sub_label = f"{params['k0']:<6} x {params['k1']:<4} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
    results.append(
        benchmark.Timer(
            stmt="batched_dot_mul_sum(x, x)",
            setup="from __main__ import batched_dot_mul_sum",
            globals=tensors,
            label="Batched dot",
            sub_label=sub_label,
            description="mul/sum",
        ).blocked_autorange(min_run_time=1)
    )
    results.append(
        benchmark.Timer(
            stmt="batched_dot_bmm(x, x)",
            setup="from __main__ import batched_dot_bmm",
            globals=tensors,
            label="Batched dot",
            sub_label=sub_label,
            description="bmm",
        ).blocked_autorange(min_run_time=1)
    )

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.print()

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     [--------------------- Batched dot ---------------------]
#                                          |  mul/sum  |   bmm
#     1 threads: ----------------------------------------------
#           725    x 257                   |      87   |    180
#           49     x 383                   |      15   |     30
#           34     x 1468                  |      30   |    118
#           187    x 5039                  |     400   |   1200
#           2140   x 1296 (discontiguous)  |    2000   |  41000
#           78     x 1598                  |      74   |    310
#           519    x 763                   |     190   |   1500
#           141    x 1082                  |      87   |    500
#           78     x 5    (discontiguous)  |       9   |     20
#           187    x 1                     |      12   |     10
#
#     Times are in microseconds (us).
#

######################################################################
# 定义自己的 `fuzzers` 有很大的灵活性,这对于创建强大的输入集进行基准测试非常有用。
# 但为了让事情变得更简单, PyTorch 基准测试模块为常见的基准测试需求提供了一些内置的 `fuzzers`。
# 让我们看看如何使用其中一个内置的 `fuzzers` 。
#

from torch.utils.benchmark.op_fuzzers import binary

results = []
for tensors, tensor_params, params in binary.BinaryOpFuzzer(seed=0).take(10):
    sub_label = f"{params['k0']:<6} x {params['k1']:<4} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
    results.append(
        benchmark.Timer(
            stmt="batched_dot_mul_sum(x, x)",
            setup="from __main__ import batched_dot_mul_sum",
            globals=tensors,
            label="Batched dot",
            sub_label=sub_label,
            description="mul/sum",
        ).blocked_autorange(min_run_time=1)
    )
    results.append(
        benchmark.Timer(
            stmt="batched_dot_bmm(x, x)",
            setup="from __main__ import batched_dot_bmm",
            globals=tensors,
            label="Batched dot",
            sub_label=sub_label,
            description="bmm",
        ).blocked_autorange(min_run_time=1)
    )

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize(rowwise=True)
compare.print()

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     [----------------------- Batched dot ------------------------]
#                                              |  mul/sum  |   bmm
#     1 threads: ---------------------------------------------------
#           64     x 473  (discontiguous)      |    10000  |   40000
#           16384  x 12642115 (discontiguous)  |       31  |      78
#           8192   x 892                       |     4800  |   20400
#           512    x 64   (discontiguous)      |   110000  |  400000
#           493    x 27   (discontiguous)      |     1100  |    2440
#           118    x 32   (discontiguous)      |      870  |    2030
#           16     x 495  (discontiguous)      |    23600  |   24000
#           488    x 62374                     |    90000  |  100000
#           240372 x 69                        |    40000  |   16000
#           40156  x 32   (discontiguous)      |     2670  |    5000
#
#     Times are in microseconds (us).
#

######################################################################
# 8. 使用 `Callgrind` 收集指令计数
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 优化代码的一个挑战是时间的变化和不透明性。有许多不确定性的来源,
# 从自适应时钟速度到与其他进程的资源争用。此外,端到端时间并不能揭示时间花费在哪里,
# 而这正是我们在优化代码时感兴趣的。
#
# 一种补充方法是也收集指令计数。这些计数是一种代理指标,并不能捕获性能的所有方面
# (例如内存或I/O绑定任务),但它们确实具有一些有用的特性。指令计数是可重复的,
# 不受环境变化的影响,并且可以提供对程序在哪里花费周期的细粒度洞察。
#
# 为了看到指令计数的实用性,让我们看看如何减少 `batched_dot_mul_sum` 的开销。
# 显而易见的解决方案是将其移至 C++ ,这样我们就可以避免在 Python 和 C++ 之间多次来回切换。
#
# 幸运的是,源代码几乎是相同的。在 C++ 中我们必须问的一个问题是,
# 我们是通过值还是引用来传递参数。
#

batched_dot_src = """\
/* ---- Python ---- */
// def batched_dot_mul_sum(a, b):
//     return a.mul(b).sum(-1)

torch::Tensor batched_dot_mul_sum_v0(
    const torch::Tensor a,
    const torch::Tensor b) {
  return a.mul(b).sum(-1);
}

torch::Tensor batched_dot_mul_sum_v1(
    const torch::Tensor& a,
    const torch::Tensor& b) {
  return a.mul(b).sum(-1);
}
"""


# PyTorch 提供一个实用程序来 JIT 编译 C++ 源代码为 Python 扩展,
# 使得测试我们的 C++ 实现变得很容易:
import os

from torch.utils import cpp_extension

cpp_lib = cpp_extension.load_inline(
    name="cpp_lib",
    cpp_sources=batched_dot_src,
    extra_cflags=["-O3"],
    extra_include_paths=[
        # `load_inline`需要知道`pybind11`头文件的位置。
        os.path.join(os.getenv("CONDA_PREFIX"), "include")
    ],
    functions=["batched_dot_mul_sum_v0", "batched_dot_mul_sum_v1"],
)

# `load_inline` 将创建一个共享对象,并加载到Python中。当我们收集指令计数时,
# Timer将创建一个子进程,因此我们需要重新导入它。对于C扩展,导入过程略有不同,
# 但这就是我们在这里所做的。
module_import_str = f"""\
# https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
import importlib.util
spec = importlib.util.spec_from_file_location("cpp_lib", {repr(cpp_lib.__file__)})
cpp_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp_lib)"""

import textwrap


def pretty_print(result):
    """Import machinery for ``cpp_lib.so`` can get repetitive to look at."""
    print(
        repr(result).replace(
            textwrap.indent(module_import_str, "  "), "  import cpp_lib"
        )
    )


t_baseline = benchmark.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="""\
from __main__ import batched_dot_mul_sum
x = torch.randn(2, 2)""",
)

t0 = benchmark.Timer(
    stmt="cpp_lib.batched_dot_mul_sum_v0(x, x)",
    setup=f"""\
{module_import_str}
x = torch.randn(2, 2)""",
)

t1 = benchmark.Timer(
    stmt="cpp_lib.batched_dot_mul_sum_v1(x, x)",
    setup=f"""\
{module_import_str}
x = torch.randn(2, 2)""",
)

# 转移到 C++ 确实减少了开销,但很难判断哪种调用约定更有效。v1(使用引用调用)似乎稍快一些,但在测量误差范围内。
pretty_print(t_baseline.blocked_autorange())
pretty_print(t0.blocked_autorange())
pretty_print(t1.blocked_autorange())

######################################################################
# .. code-block:: none
#    :caption: Output
#
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb16935d2e8>
#     batched_dot_mul_sum(x, x)
#     setup:
#       from __main__ import batched_dot_mul_sum
#       x = torch.randn(2, 2)
#
#       6.92 us
#       1 measurement, 100000 runs , 1 thread
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb16935d2e8>
#     cpp_lib.batched_dot_mul_sum_v0(x, x)
#     setup:
#       import cpp_lib
#       x = torch.randn(2, 2)
#
#       5.29 us
#       1 measurement, 100000 runs , 1 thread
#     <torch.utils.benchmark.utils.common.Measurement object at 0x7fb16935d2e8>
#     cpp_lib.batched_dot_mul_sum_v1(x, x)
#     setup:
#       import cpp_lib
#       x = torch.randn(2, 2)
#
#       5.22 us
#       1 measurement, 100000 runs , 1 thread
#

# 让我们使用 ``Callgrind`` 来确定哪种方式更好。
stats_v0 = t0.collect_callgrind()
stats_v1 = t1.collect_callgrind()

pretty_print(stats_v0)
pretty_print(stats_v1)

# `.as_standardized` 移除了文件名和某些路径前缀,使函数符号更易读。
stats_v0 = stats_v0.as_standardized()
stats_v1 = stats_v1.as_standardized()

# `.delta` 对指令计数进行差分, `.denoise` 则移除了 Python 解释器中已知存在显著抖动的几个函数。
delta = stats_v1.delta(stats_v0).denoise()

# `.transform` 是一个转换函数名的便利 API。它在进行 ``diff-ing`` 时很有用,因为可以增加抵消,同时也能提高可读性。
replacements = (
    ("???:void pybind11", "pybind11"),
    ("batched_dot_mul_sum_v0", "batched_dot_mul_sum_v1"),
    ("at::Tensor, at::Tensor", "..."),
    ("at::Tensor const&, at::Tensor const&", "..."),
    ("auto torch::detail::wrap_pybind_function_impl_", "wrap_pybind_function_impl_"),
)
for before, after in replacements:
    delta = delta.transform(lambda l: l.replace(before, after))

# 我们可以使用打印选项来控制显示函数的多少内容。
torch.set_printoptions(linewidth=160)

# 解析后,指令计数清楚地表明,通过引用传递 `a` 和 `b` 更有效,
# 因为它跳过了一些 `c10::TensorImpl` 中间张量的簿记操作,并且与 `pybind11` 也更兼容。
# 这与我们有噪声时间观察结果一致。
print(delta)

######################################################################
# .. code-block::
#
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.CallgrindStats object at 0x7fb0f06e7630>
#     cpp_lib.batched_dot_mul_sum_v0(x, x)
#     setup:
#       import cpp_lib
#       x = torch.randn(2, 2)
#                                All          Noisy symbols removed
#         Instructions:      2392671                    2392671
#         Baseline:             4367                       4367
#     100 runs per measurement, 1 thread
#     Warning: PyTorch was not built with debug symbols.
#              Source information may be limited. Rebuild with
#              REL_WITH_DEB_INFO=1 for more detailed results.
#     <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.CallgrindStats object at 0x7fb10400d208>
#     cpp_lib.batched_dot_mul_sum_v1(x, x)
#     setup:
#       import cpp_lib
#       x = torch.randn(2, 2)
#                                All          Noisy symbols removed
#         Instructions:      2378978                    2378978
#         Baseline:             4367                       4367
#         100 runs per measurement, 1 thread
#         Warning: PyTorch was not built with debug symbols.
#                  Source information may be limited. Rebuild with
#                  REL_WITH_DEB_INFO=1 for more detailed results.
#         <torch.utils.benchmark.utils.valgrind_wrapper.timer_interface.FunctionCounts object at 0x7fb1000ab358>
#               86  ???:0x000000000020d9e0
#           56  ???:0x000000000020db10
#        -1100  pybind11::cpp_function::initialize<wrap_pybind_function_impl_<at::Tensor ... r (&)(...), std::integer_sequence<unsigned long, 0ul, 1ul>)::{lambda(...)
#        -1600  ???:wrap_pybind_function_impl_<at::Tensor (&)(...), 0ul, 1ul>(at::Tensor (&)(...), std::integer_sequence<unsigned long, 0ul, 1ul>)::{lambda(...)
#        -5200  ???:c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reset_()
#        -5935  ???:0x000000000022c0e0
#     Total: -13693
#


######################################################################
# 学习更多
# ----------
#
# 查看其他教程继续学习:
#
# -  `PyTorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler.html>`_
#
