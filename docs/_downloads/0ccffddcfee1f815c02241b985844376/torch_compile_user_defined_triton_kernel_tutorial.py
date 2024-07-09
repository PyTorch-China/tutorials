# -*- coding: utf-8 -*-

"""
使用 ``torch.compile`` 和用户自定义的 Triton 内核
=========================================================
**作者:** `Oguz Ulgen <https://github.com/oulgen>`_
"""

######################################################################
# 用户自定义的 Triton 内核可用于优化模型计算的特定部分。这些内核是用 Triton 语言编写的,
# 旨在更容易实现硬件的峰值性能。通过在 ``torch.compile`` 中使用用户自定义的 Triton 内核,
# 您可以将这些优化过的计算集成到 PyTorch 模型中,从而可能获得显著的性能提升。
#
# 本教程演示了如何在 ``torch.compile`` 中使用用户自定义的 Triton 内核。
#
# 先决条件
# -------------------
#
# 在开始本教程之前,请确保您具备以下条件:
#
# * 对 ``torch.compile`` 和 Triton 有基本的了解。参见:
#
#   * `torch.compiler API 文档 <https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler>`__
#   * `torch.compile 介绍 <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__
#   * `Triton 语言文档 <https://triton-lang.org/main/index.html>`__
#
# * PyTorch 2.3 或更高版本
# * 支持 Triton 的 GPU
#

import torch
from torch.utils._triton import has_triton

######################################################################
# 基本用法
# --------------------
#
# 在此示例中,我们将使用来自 Triton 文档的一个简单向量加法内核与 ``torch.compile``。
# 参考 `Triton 文档 <https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html>`__。
#

if not has_triton():
    print("由于此设备不支持 triton,因此跳过。")
else:
    import triton
    from triton import language as tl

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @torch.compile(fullgraph=True)
    def add_fn(x, y):
        output = torch.zeros_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)
        return output

    x = torch.randn(4, device="cuda")
    y = torch.randn(4, device="cuda")
    out = add_fn(x, y)
    print(f"向量加法\nX:\t{x}\nY:\t{y}\n结果为\n{out}")

######################################################################
# 高级用法
# -------------------------------------------------------------------
#
# Triton 的自动调优功能是一个强大的工具,可自动优化 Triton 内核的配置参数。
# 它探索一系列可能的配置,并选择为您的特定用例提供最佳性能的配置。
#
# 与 ``torch.compile`` 一起使用时, ``triton.autotune`` 可以帮助确保您的 PyTorch 模型以最高效的方式运行。
# 下面是使用 ``torch.compile`` 和 ``triton.autotune`` 的示例。
#
# .. note::
#
#   ``torch.compile`` 仅支持 ``triton.autotune`` 的配置和关键参数。

if not has_triton():
    print("由于此设备不支持 triton,因此跳过。")
else:
    import triton
    from triton import language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 4}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 4}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2}, num_stages=4, num_warps=4),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @torch.compile(fullgraph=True)
    def add_fn(x, y):
        output = torch.zeros_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel_autotuned[grid](x, y, output, n_elements)
        return output

    x = torch.randn(4, device="cuda")
    y = torch.randn(4, device="cuda")
    out = add_fn(x, y)
    print(f"向量加法\nX:\t{x}\nY:\t{y}\n结果为\n{out}")

######################################################################
# 可组合性和限制
# --------------------------------------------------------------------
#
# 从 PyTorch 2.3 开始, ``torch.compile`` 中对用户自定义 Triton 内核的支持包括动态形状、
# ``torch.autograd.Function``、JIT inductor 和 AOT inductor。
# 您可以将这些功能组合在一起构建复杂的高性能模型。
#
# 但是,也需要注意一些限制:
#
# * **Tensor 子类:** 目前不支持张量子类和其他高级功能。
# * **Triton 功能:** 虽然 ``triton.heuristics`` 可以单独使用或在 ``triton.autotune`` 之前使用,
#   但不能在 ``triton.autotune`` 之后使用。这意味着如果要一起使用 ``triton.heuristics`` 和 ``triton.autotune``,
#   则必须先使用 ``triton.heuristics``。
#
# 结论
# -----------
# 在本教程中,我们探讨了如何在 ``torch.compile`` 中使用用户自定义的 Triton 内核。
# 我们深入研究了使用简单向量加法内核的基本用法,以及涉及 Triton 自动调优功能的高级用法。
# 我们还讨论了用户自定义 Triton 内核与其他 PyTorch 功能的可组合性,并强调了一些当前的限制。
#
# 另请参阅
# ---------
#
# * `编译优化器 <https://pytorch.org/tutorials/recipes/compiling_optimizer.html>`__
# * `使用缩放点积注意力实现高性能 Transformer <https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html>`__
