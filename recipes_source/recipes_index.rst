PyTorch 示例
---------------------------------------------
Recipes are bite-sized, actionable examples of how to use specific PyTorch features, different from our full-length tutorials.

与入门教程不同，此系列通过简洁实用的示例，展示了如何使用PyTorch的特性。





.. raw:: html

        </div>
    </div>

    <div id="tutorial-cards-container">

    <nav class="navbar navbar-expand-lg navbar-light tutorials-nav col-12">
        <div class="tutorial-tags-container">
            <div id="dropdown-filter-tags">
                <div class="tutorial-filter-menu">
                    <div class="tutorial-filter filter-btn all-tag-selected" data-tag="all">All</div>
                </div>
            </div>
        </div>
    </nav>

    <hr class="tutorials-hr">

    <div class="row">

    <div id="tutorial-cards">
    <div class="list">

.. Add recipe cards below this line

.. Basics

.. customcarditem::
   :header: PyTorch 加载数据
   :card_description: 学习如何使用 PyTorch 来准备和加载常见的数据集。
   :image: ../_static/img/thumbnails/cropped/loading-data.PNG
   :link: ../recipes/recipes/loading_data_recipe.html
   :tags: Basics


.. customcarditem::
   :header: PyTorch 创建神经网络
   :card_description: 学习如何使用torch.nn，为MNIST数据集创建一个神经网络。
   :image: ../_static/img/thumbnails/cropped/defining-a-network.PNG
   :link: ../recipes/recipes/defining_a_neural_network.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch 中 state_dict 是什么
   :card_description: 学习如何使用 `state_dict` 对象和 Python 字典在 PyTorch 中保存或加载模型。
   :image: ../_static/img/thumbnails/cropped/what-is-a-state-dict.PNG
   :link: ../recipes/recipes/what_is_state_dict.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch 保存和加载模型
   :card_description: 在PyTorch中保存和加载模型用于推理的两种方式 - state_dict和完整模型。
   :image: ../_static/img/thumbnails/cropped/saving-and-loading-models-for-inference.PNG
   :link: ../recipes/recipes/saving_and_loading_models_for_inference.html
   :tags: Basics


.. customcarditem::
   :header: PyTorch 保存和加载通用检查点
   :card_description: 保存和加载一个通用的检查点模型,可以帮助您从上次停止的地方继续推理或训练。在这个示例中,探索如何保存和加载多个检查点。
   :image: ../_static/img/thumbnails/cropped/saving-and-loading-general-checkpoint.PNG
   :link: ../recipes/recipes/saving_and_loading_a_general_checkpoint.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch 在一个文件中保存和加载多个模型
   :card_description: 在这个示例中,学习保存和加载多个模型,有助于重用您之前训练过的模型。
   :image: ../_static/img/thumbnails/cropped/saving-multiple-models.PNG
   :link: ../recipes/recipes/saving_multiple_models_in_one_file.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch 使用不同模型的参数对模型进行热启动
   :card_description: 了解如何通过部分加载模型或加载部分模型方式来热启动训练过程,这可以帮助您的模型比从头开始训练收敛得更快。
   :image: ../_static/img/thumbnails/cropped/warmstarting-models.PNG
   :link: ../recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch 跨设备保存和加载模型
   :card_description: 了解如何使用PyTorch在不同设备(CPU和GPU)之间保存和加载模型。
   :image: ../_static/img/thumbnails/cropped/saving-and-loading-models-across-devices.PNG
   :link: ../recipes/recipes/save_load_across_devices.html
   :tags: Basics

.. customcarditem::
   :header:  PyTorch 清零梯度
   :card_description: 了解何时应该清零梯度,以及这样做如何有助于提高模型的精度。
   :image: ../_static/img/thumbnails/cropped/zeroing-out-gradients.PNG
   :link: ../recipes/recipes/zeroing_out_gradients.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Benchmark
   :card_description: 学习如何使用 PyTorch Benchmark 模块来测量和比较代码性能
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/benchmark.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Benchmark Timer 快速入门
   :card_description: 学习如何测量代码片段的运行时间和收集指令。
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/timer_quick_start.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Profiler
   :card_description: 学习如何使用 PyTorch Profiler 来测量算子的时间和内存消耗。
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/profiler_recipe.html
   :tags: Basics

.. customcarditem::
   :header: PyTorch Profiler with Instrumentation and Tracing Technology API (ITT API) support
   :card_description: 学习如何使用支持 Instrumentation and Tracing Technology API (ITT API) 的 PyTorch Profiler,在 Intel® VTune™ Profiler GUI 中可视化算子标签
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/profile_with_itt.html
   :tags: Basics

.. customcarditem::
   :header: Torch Compile IPEX 后端
   :card_description: 学习如何使用 torch.compile IPEX 后端
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_compile_backend_ipex.html
   :tags: Basics

.. customcarditem::
   :header: 在 PyTorch 中推理形状
   :card_description: 学习如何使用 meta 设备来推理模型中的形状。
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/recipes/reasoning_about_shapes.html
   :tags: Basics

.. customcarditem::
   :header: 从检查点加载 nn.Module 的技巧
   :card_description: 学习从检查点加载 nn.Module 的技巧。
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/recipes/module_load_state_dict_tips.html
   :tags: Basics

.. customcarditem::
   :header: (beta) 使用 TORCH_LOGS 观察 torch.compile
   :card_description: 学习如何使用 torch 日志 API 观察编译过程。
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_logs.html
   :tags: Basics

.. customcarditem::
   :header: nn.Module 中用于加载 state_dict 和张量子类的扩展点
   :card_description: nn.Module 中的新扩展点。
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/recipes/swap_tensors.html
   :tags: Basics


.. Interpretability

.. customcarditem::
   :header: 使用 Captum 进行模型可解释性
   :card_description: 学习如何使用 Captum 将图像分类器的预测归因于相应的图像特征,并可视化归因结果。
   :image: ../_static/img/thumbnails/cropped/model-interpretability-using-captum.png
   :link: ../recipes/recipes/Captum_Recipe.html
   :tags: Interpretability,Captum

.. customcarditem::
   :header: 如何在 PyTorch 中使用 TensorBoard
   :card_description: 学习在 PyTorch 中使用 TensorBoard 的基本用法,以及如何在 TensorBoard UI 中可视化数据
   :image: ../_static/img/thumbnails/tensorboard_scalars.png
   :link: ../recipes/recipes/tensorboard_with_pytorch.html
   :tags: Visualization,TensorBoard

.. Quantization

.. customcarditem::
   :header: 动态量化
   :card_description:  对一个简单的 LSTM 模型应用动态量化。
   :image: ../_static/img/thumbnails/cropped/using-dynamic-post-training-quantization.png
   :link: ../recipes/recipes/dynamic_quantization.html
   :tags: Quantization,Text,Model-Optimization


.. Production Development

.. customcarditem::
   :header: 部署时使用 TorchScript
   :card_description: 学习如何将训练好的模型导出为 TorchScript 格式,以及如何在 C++ 中加载 TorchScript 模型并进行推理。
   :image: ../_static/img/thumbnails/cropped/torchscript_overview.png
   :link: ../recipes/torchscript_inference.html
   :tags: TorchScript

.. customcarditem::
   :header: 使用 Flask 进行部署
   :card_description: 学习如何使用轻量级 Web 服务器 Flask 快速从训练好的 PyTorch 模型搭建 Web API。
   :image: ../_static/img/thumbnails/cropped/using-flask-create-restful-api.png
   :link: ../recipes/deployment_with_flask.html
   :tags: Production,TorchScript

.. customcarditem::
   :header: PyTorch 移动端性能优化技巧
   :card_description: 在移动端(Android 和 iOS)使用 PyTorch 时的一些性能优化技巧。
   :image: ../_static/img/thumbnails/cropped/mobile.png
   :link: ../recipes/mobile_perf.html
   :tags: Mobile,Model-Optimization

.. customcarditem::
   :header: 制作使用 PyTorch Android 预编译库的 Android 原生应用
   :card_description: 学习如何从头开始制作使用 LibTorch C++ API 和 TorchScript 模型(带自定义 C++ 算子)的 Android 应用。
   :image: ../_static/img/thumbnails/cropped/android.png
   :link: ../recipes/android_native_app_with_custom_op.html
   :tags: Mobile

.. customcarditem::
  :header: 融合模块技巧
  :card_description: 学习如何在量化之前将一系列 PyTorch 模块融合为单个模块,以减小模型大小。
  :image: ../_static/img/thumbnails/cropped/mobile.png
  :link: ../recipes/fuse.html
  :tags: Mobile

.. customcarditem::
  :header: 移动端量化技巧
  :card_description: 学习如何在不太损失精度的情况下减小模型大小并加快运行速度。
  :image: ../_static/img/thumbnails/cropped/mobile.png
  :link: ../recipes/quantization.html
  :tags: Mobile,Quantization

.. customcarditem::
  :header: 为移动端脚本化和优化
  :card_description: 学习如何将模型转换为 TorchScript,并(可选)为移动应用优化。
  :image: ../_static/img/thumbnails/cropped/mobile.png
  :link: ../recipes/script_optimized.html
  :tags: Mobile

.. customcarditem::
  :header: iOS 端模型准备技巧
  :card_description: 学习如何将模型添加到 iOS 项目中,以及如何使用 PyTorch pod for iOS。
  :image: ../_static/img/thumbnails/cropped/ios.png
  :link: ../recipes/model_preparation_ios.html
  :tags: Mobile

.. customcarditem::
  :header: Android 端模型准备技巧
  :card_description: 学习如何将模型添加到 Android 项目中,以及如何使用 PyTorch library for Android。
  :image: ../_static/img/thumbnails/cropped/android.png
  :link: ../recipes/model_preparation_android.html
  :tags: Mobile

.. customcarditem::
   :header: Android 和 iOS 上的移动端解释器工作流程
   :card_description: 学习如何在 iOS 和 Android 设备上使用移动端解释器。
   :image: ../_static/img/thumbnails/cropped/mobile.png
   :link: ../recipes/mobile_interpreter.html
   :tags: Mobile

.. customcarditem::
   :header: 分析基于 PyTorch RPC 的工作负载
   :card_description: 如何使用 PyTorch Profiler 分析基于 RPC 的工作负载。
   :image: ../_static/img/thumbnails/cropped/profile.png
   :link: ../recipes/distributed_rpc_profiling.html
   :tags: Production

.. Automatic Mixed Precision

.. customcarditem::
   :header: 自动混合精度
   :card_description: 使用 torch.cuda.amp 在 NVIDIA GPU 上减少运行时间并节省内存。
   :image: ../_static/img/thumbnails/cropped/amp.png
   :link: ../recipes/recipes/amp_recipe.html
   :tags: Model-Optimization

.. Performance

.. customcarditem::
   :header: 性能优化指南
   :card_description: 实现最佳性能的技巧。
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/recipes/tuning_guide.html
   :tags: Model-Optimization

.. customcarditem::
   :header: 在 AWS Graviton 处理器上优化 PyTorch 推理性能
   :card_description: 在 AWS Graviton CPU 上实现最佳推理性能的技巧
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/inference_tuning_on_aws_graviton.html
   :tags: Model-Optimization

.. Leverage Advanced Matrix Extensions

.. customcarditem::
   :header: 利用 Intel® 高级矩阵扩展
   :card_description: 学习如何利用 Intel® 高级矩阵扩展。
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/amx.html
   :tags: Model-Optimization

.. (beta) Compiling the Optimizer with torch.compile

.. customcarditem::
   :header: (beta) 使用 torch.compile 编译优化器
   :card_description: 使用 torch.compile 加速优化器
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/compiling_optimizer.html
   :tags: Model-Optimization

.. (beta) Running the compiled optimizer with an LR Scheduler

.. customcarditem::
   :header: (beta) 使用学习率调度器运行编译后的优化器
   :card_description: 使用 LRScheduler 和 torch.compiled 优化器加速训练
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/compiling_optimizer_lr_scheduler.html
   :tags: Model-Optimization

.. Using User-Defined Triton Kernels with ``torch.compile``

.. customcarditem::
   :header: 在 ``torch.compile`` 中使用用户定义的 Triton 内核
   :card_description: 学习如何在 ``torch.compile`` 中使用用户定义的内核
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torch_compile_user_defined_triton_kernel_tutorial.html
   :tags: Model-Optimization

.. Intel(R) Extension for PyTorch*

.. customcarditem::
   :header: Intel® Extension for PyTorch*
   :card_description: Intel® Extension for PyTorch* 介绍
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/intel_extension_for_pytorch.html
   :tags: Model-Optimization

.. Intel(R) Neural Compressor for PyTorch*

.. customcarditem::
   :header: Intel® Neural Compressor for PyTorch
   :card_description: 使用 Intel® Neural Compressor 轻松量化 PyTorch。
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/intel_neural_compressor_for_pytorch.html
   :tags: Quantization,Model-Optimization

.. Distributed Training

.. customcarditem::
   :header: DeviceMesh 入门
   :card_description: 学习如何使用 DeviceMesh
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/distributed_device_mesh.html
   :tags: Distributed-Training

.. customcarditem::
   :header: 使用 ZeroRedundancyOptimizer 分片优化器状态
   :card_description: 如何使用 ZeroRedundancyOptimizer 减少内存消耗。
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/zero_redundancy_optimizer.html
   :tags: Distributed-Training

.. customcarditem::
   :header: 使用 TensorPipe RPC 实现直接设备间通信
   :card_description: 如何使用支持直接 GPU 到 GPU 通信的 RPC。
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/cuda_rpc.html
   :tags: Distributed-Training

.. customcarditem::
   :header: 支持 TorchScript 的分布式优化器
   :card_description: 如何为分布式优化器启用 TorchScript 支持。
   :image: ../_static/img/thumbnails/cropped/profiler.png
   :link: ../recipes/distributed_optim_torchscript.html
   :tags: Distributed-Training,TorchScript

.. customcarditem::
   :header: 分布式检查点 (DCP) 入门
   :card_description: 学习如何使用分布式检查点包检查点分布式模型。
   :image: ../_static/img/thumbnails/cropped/Getting-Started-with-DCP.png
   :link: ../recipes/distributed_checkpoint_recipe.html
   :tags: Distributed-Training

.. TorchServe

.. customcarditem::
   :header: 将 PyTorch Stable Diffusion 模型部署为 Vertex AI 端点
   :card_description: 学习如何使用 TorchServe 在 Vertex AI 中部署模型
   :image: ../_static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: ../recipes/torchserve_vertexai_tutorial.html
   :tags: Production

.. End of tutorial card section

.. raw:: html

    </div>

    <div class="pagination d-flex justify-content-center"></div>

    </div>

    </div>

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :hidden:

   /recipes/recipes/loading_data_recipe
   /recipes/recipes/defining_a_neural_network
   /recipes/torch_logs
   /recipes/recipes/what_is_state_dict
   /recipes/recipes/saving_and_loading_models_for_inference
   /recipes/recipes/saving_and_loading_a_general_checkpoint
   /recipes/recipes/saving_multiple_models_in_one_file
   /recipes/recipes/warmstarting_model_using_parameters_from_a_different_model
   /recipes/recipes/save_load_across_devices
   /recipes/recipes/zeroing_out_gradients
   /recipes/recipes/profiler_recipe
   /recipes/recipes/profile_with_itt
   /recipes/recipes/Captum_Recipe
   /recipes/recipes/tensorboard_with_pytorch
   /recipes/recipes/dynamic_quantization
   /recipes/recipes/amp_recipe
   /recipes/recipes/tuning_guide
   /recipes/recipes/intel_extension_for_pytorch
   /recipes/compiling_optimizer
   /recipes/torch_compile_backend_ipex
   /recipes/torchscript_inference
   /recipes/deployment_with_flask
   /recipes/distributed_rpc_profiling
   /recipes/zero_redundancy_optimizer
   /recipes/cuda_rpc
   /recipes/distributed_optim_torchscript
   /recipes/mobile_interpreter
