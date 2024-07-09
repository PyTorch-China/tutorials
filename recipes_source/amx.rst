==============================================
利用英特尔®高级矩阵扩展(Intel® Advanced Matrix Extensions)
==============================================

简介
====

高级矩阵扩展(AMX)，也称为英特尔®高级矩阵扩展(Intel® AMX)，是一种x86扩展，引入了两个新组件:一个称为"tile"的二维寄存器文件和一个能够在这些tile上进行矩阵乘法(TMUL)的加速器。AMX旨在加速CPU上的深度学习训练和推理工作负载,非常适合自然语言处理、推荐系统和图像识别等工作负载。

英特尔通过第4代英特尔®至强®可扩展处理器和英特尔®AMX推进了AI能力,相比上一代产品,推理和训练性能提高了3倍至10倍,详见`使用Intel® AMX加速AI工作负载`_。与运行Intel®高级矢量扩展512神经网络指令(Intel® AVX-512 VNNI)的第3代英特尔至强可扩展处理器相比,运行Intel AMX的第4代英特尔至强可扩展处理器每周期可执行2,048个INT8操作,而不是256个INT8操作;它们还可以每周期执行1,024个BF16操作,而不是64个FP32操作,详见`使用Intel® AMX加速AI工作负载`_第4页。有关AMX的更多详细信息,请参阅`Intel® AMX概述`_。

PyTorch中的AMX
==============

PyTorch通过其后端oneDNN利用AMX来计算BFloat16和INT8量化的计算密集型算子,从而在支持AMX的x86 CPU上获得更高的性能。
有关oneDNN的更多详细信息,请参阅`oneDNN`_。

操作完全由oneDNN根据生成的执行代码路径处理。例如,当支持的操作在支持AMX的硬件平台上执行到oneDNN实现时,AMX指令将在oneDNN内部自动调用。
由于oneDNN是PyTorch CPU的默认加速库,因此无需手动操作即可启用AMX支持。

利用AMX加速工作负载的指南
-------------------------------------------

本节提供了如何利用AMX加速各种工作负载的指南。

- BFloat16数据类型:

  - 使用``torch.cpu.amp``或``torch.autocast("cpu")``将利用AMX加速支持的算子。

   ::

      model = model.to(memory_format=torch.channels_last)
      with torch.cpu.amp.autocast():
         output = model(input)

.. note:: 使用``torch.channels_last``内存格式可获得更好的性能。

- 量化:

  - 应用量化将利用AMX加速支持的算子。

- torch.compile:

  - 当生成的图模型运行到oneDNN实现的支持算子时,AMX加速将被激活。

.. note:: 在支持AMX的CPU上使用PyTorch时,框架将默认自动启用AMX使用。这意味着PyTorch将尽可能利用AMX功能来加速矩阵乘法操作。但是,重要的是要注意,是否调度到AMX内核最终取决于PyTorch所依赖的oneDNN库和量化后端的内部优化策略。PyTorch和oneDNN库内部如何处理AMX利用的具体细节可能会随着框架的更新和改进而发生变化。


可利用AMX的CPU算子:
------------------------------------

可利用AMX的BF16 CPU算子:

- ``conv1d``
- ``conv2d``
- ``conv3d``
- ``conv_transpose1d``
- ``conv_transpose2d``
- ``conv_transpose3d``
- ``bmm``
- ``mm``
- ``baddbmm``
- ``addmm``
- ``addbmm``
- ``linear``
- ``matmul``

可利用AMX的量化CPU算子:

- ``conv1d``
- ``conv2d``
- ``conv3d``
- ``conv_transpose1d``
- ``conv_transpose2d``
- ``conv_transpose3d``
- ``linear``



确认AMX正在被利用
------------------------------

设置环境变量``export ONEDNN_VERBOSE=1``或使用``torch.backends.mkldnn.verbose``以启用oneDNN转储详细消息。

::

   with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
       with torch.cpu.amp.autocast():
           model(input)

例如,获取oneDNN详细输出:

::

   onednn_verbose,info,oneDNN v2.7.3 (commit 6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e)
   onednn_verbose,info,cpu,runtime:OpenMP,nthr:128
   onednn_verbose,info,cpu,isa:Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with bfloat16 and 8-bit integer support
   onednn_verbose,info,gpu,runtime:none
   onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
   onednn_verbose,exec,cpu,reorder,simple:any,undef,src_f32::blocked:a:f0 dst_f32::blocked:a:f0,attr-scratchpad:user ,,2,5.2561
   ...
   onednn_verbose,exec,cpu,convolution,jit:avx512_core_amx_bf16,forward_training,src_bf16::blocked:acdb:f0 wei_bf16:p:blocked:ABcd16b16a2b:f0 bia_f32::blocked:a:f0 dst_bf16::blocked:acdb:f0,attr-scratchpad:user ,alg:convolution_direct,mb7_ic2oc1_ih224oh111kh3sh2dh1ph1_iw224ow111kw3sw2dw1pw1,0.628906
   ...
   onednn_verbose,exec,cpu,matmul,brg:avx512_core_amx_int8,undef,src_s8::blocked:ab:f0 wei_s8:p:blocked:BA16a64b4a:f0 dst_s8::blocked:ab:f0,attr-scratchpad:user ,,1x30522:30522x768:1x768,7.66382
   ...

如果你获得了``avx512_core_amx_bf16``的详细输出(用于BFloat16)或``avx512_core_amx_int8``(用于INT8量化),则表示AMX已被激活。


结论
----------

在本教程中,我们简要介绍了AMX、如何在PyTorch中利用AMX来加速工作负载,以及如何确认AMX正在被利用。

随着PyTorch和oneDNN的改进和更新,AMX的利用情况可能会相应发生变化。

如果您遇到任何问题或有任何疑问,您可以使用`论坛 <https://discuss.pytorch.org/>`_或`GitHub issues <https://github.com/pytorch/pytorch/issues>`_与我们联系。


.. _使用Intel® AMX加速AI工作负载: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/ai-solution-brief.html

.. _Intel® AMX概述: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html

.. _oneDNN: https://oneapi-src.github.io/oneDNN/index.html
