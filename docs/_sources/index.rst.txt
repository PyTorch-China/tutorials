欢迎来到 PyTorch 教程
============================

**PyTorch 新增教程**

* `使用自定义的 Triton 内核与 torch.compile <https://pytorch-cn.com/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html>`__
* `通过区域编译减少 torch.compile 冷启动编译时间 <https://pytorch-cn.com/tutorials/recipes/regional_compilation.html>`__
* `使用 Tensor Parallel (TP) 进行大规模 Transformer 模型训练 <https://pytorch-cn.com/tutorials/intermediate/TP_tutorial.html>`__
* `利用半结构化(2:4)稀疏性加速 BERT <https://pytorch-cn.com/tutorials/advanced/semi_structured_sparse.html>`__
* `torch.export 教程 <https://pytorch-cn.com/tutorials/intermediate/torch_export_tutorial.html>`__
* `nn.Module 中 load_state_dict 和张量子类的扩展点 <https://pytorch-cn.com/tutorials/recipes/recipes/swap_tensors.html>`__

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: 熟悉 PyTorch 的概念和模块。通过本快速入门指南，学习如何加载数据、构建深度神经网络、训练和保存模型。
   :header: 基础知识
   :button_link:  beginner/basics/intro.html
   :button_text: 开启 PyTorch 旅程

.. customcalloutitem::
   :description: 小巧易用、即时部署的 PyTorch 代码示例。
   :header: PyTorch 示例
   :button_link: recipes/recipes_index.html
   :button_text: 查看示例

.. End of callout item section

.. raw:: html

        </div>
    </div>

    <div id="tutorial-cards-container">

    <nav class="navbar navbar-expand-lg navbar-light tutorials-nav col-12">
        <div class="tutorial-tags-container">
            <div id="dropdown-filter-tags">
                <div class="tutorial-filter-menu">
                    <div class="tutorial-filter filter-btn all-tag-selected" data-tag="all">全部</div>
                </div>
            </div>
        </div>
    </nav>

    <hr class="tutorials-hr">

    <div class="row">

    <div id="tutorial-cards">
    <div class="list">

.. Add tutorial cards below this line

.. Learning PyTorch

.. customcarditem::
   :header: 基础知识
   :card_description: 逐步教你如何用PyTorch构建完整的机器学习流程。
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: beginner/basics/intro.html
   :tags: Getting-Started

.. customcarditem::
   :header: YouTube PyTorch 介绍视频
   :card_description: 用PyTorch构建完整的机器学习工作流程，PyTorch初学者系列。
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: beginner/introyt.html
   :tags: Getting-Started

.. customcarditem::
   :header: 通过示例学习 PyTorch
   :card_description: 本教程通过独立的示例介绍了 PyTorch 的基本概念。
   :image: _static/img/thumbnails/cropped/learning-pytorch-with-examples.png
   :link: beginner/pytorch_with_examples.html
   :tags: Getting-Started

.. customcarditem::
   :header: 什么是 torch.nn ?
   :card_description: 使用 torch.nn 来创建和训练神经网络。
   :image: _static/img/thumbnails/cropped/torch-nn.png
   :link: beginner/nn_tutorial.html
   :tags: Getting-Started

.. customcarditem::
   :header: 使用 TensorBoard 展现模型、数据和训练过程
   :card_description: 学习使用 TensorBoard 可视化数据集和模型训练过程。
   :image: _static/img/thumbnails/cropped/visualizing-with-tensorboard.png
   :link: intermediate/tensorboard_tutorial.html
   :tags: Interpretability,Getting-Started,TensorBoard

.. Image/Video

.. customcarditem::
   :header: TorchVision 目标检测微调教程
   :card_description: 微调预训练的 Mask R-CNN 模型。
   :image: _static/img/thumbnails/cropped/TorchVision-Object-Detection-Finetuning-Tutorial.png
   :link: intermediate/torchvision_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: 计算机视觉迁移学习教程
   :card_description: 使用迁移学习训练卷积神经网络进行图像分类。
   :image: _static/img/thumbnails/cropped/Transfer-Learning-for-Computer-Vision-Tutorial.png
   :link: beginner/transfer_learning_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: 优化视觉Transformer模型
   :card_description: 应用最前沿的、基于 attention-based transformer 模型到计算机视觉任务中。
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: beginner/vt_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: 对抗性样本生成
   :card_description: 使用迁移学习训练卷积神经网络进行图像分类。
   :image: _static/img/thumbnails/cropped/Adversarial-Example-Generation.png
   :link: beginner/fgsm_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: DCGAN 教程
   :card_description: Train a generative adversarial network (GAN) to generate new celebrities.
   :image: _static/img/thumbnails/cropped/DCGAN-Tutorial.png
   :link: beginner/dcgan_faces_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: Spatial Transformer Networks 教程
   :card_description: 学习如何通过视觉注意机制增强你的网络。
   :image: _static/img/stn/Five.gif
   :link: intermediate/spatial_transformer_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: 使用 TIAToolbox 对图像进行推理
   :card_description: 学习如何使用TIAToolbox对图像进行推理。
   :image: _static/img/thumbnails/cropped/TIAToolbox-Tutorial.png
   :link: intermediate/tiatoolbox_tutorial.html
   :tags: Image/Video

.. customcarditem::
   :header: 基于 USB 的半监督学习教程
   :card_description: 学习如何使用 USB 和 PyTorch 对自定义数据进行半监督学习算法的训练。
   :image: _static/img/usb_semisup_learn/code.png
   :link: advanced/usb_semisup_learn.html
   :tags: Image/Video

.. Audio

.. customcarditem::
   :header: Audio IO
   :card_description: 使用 torchaudio 加载数据。
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_io_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio 重采样
   :card_description: 学习使用 torchaudio 对音频波形进行重新采样。
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_resampling_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio 数据增强
   :card_description: 学习使用 torchaudio 应用数据增强。
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_data_augmentation_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio 特征提取
   :card_description: 学习使用 torchaudio 提取特征。
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_feature_extractions_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio 特征增强
   :card_description: 学习使用 torchaudio 对特征进行增强。
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_feature_augmentation_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: Audio 数据集
   :card_description: 学习如何使用 torchaudio 数据集。
   :image: _static/img/thumbnails/cropped/torchaudio-Tutorial.png
   :link: beginner/audio_datasets_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: 在 torchaudio 中使用 Wav2Vec2 进行自动语音识别
   :card_description: 学习如何使用 torchaudio 的预训练模型来构建语音识别应用程序。
   :image: _static/img/thumbnails/cropped/torchaudio-asr.png
   :link: intermediate/speech_recognition_pipeline_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: 语音命令分类
   :card_description: 学习如何正确格式化音频数据集，然后在该数据集上训练/测试音频分类器网络。
   :image: _static/img/thumbnails/cropped/torchaudio-speech.png
   :link: intermediate/speech_command_classification_with_torchaudio_tutorial.html
   :tags: Audio

.. customcarditem::
   :header: 使用 torchaudio 进行文本转语音
   :card_description: 学习如何使用 torchaudio 的预训练模型构建文本转语音应用程序。
   :image: _static/img/thumbnails/cropped/torchaudio-speech.png
   :link: intermediate/text_to_speech_with_torchaudio.html
   :tags: Audio

.. customcarditem::
   :header: 在 torchaudio 中使用 Wav2Vec2 进行对齐
   :card_description: 学习如何使用 torchaudio 的 Wav2Vec2 预训练模型对文本进行与语音对齐。
   :image: _static/img/thumbnails/cropped/torchaudio-alignment.png
   :link: intermediate/forced_alignment_with_torchaudio_tutorial.html
   :tags: Audio

.. Text

.. customcarditem::
   :header: 使用 Better Transformer 提升推理效率
   :card_description: 使用 Better Transformer 实现的 PyTorch Transformer 模型，以实现高性能的推断。
   :image: _static/img/thumbnails/cropped/pytorch-logo.png
   :link: beginner/bettertransformer_tutorial.html
   :tags: Production,Text

.. customcarditem::
   :header: 从零开始的自然语言处理：使用字符级 RNN 对姓名进行分类
   :card_description: 1.构建并训练一个基本的字符级循环神经网络，从零开始分类单词，而不使用 torchtext。
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Classifying-Names-with-a-Character-Level-RNN.png
   :link: intermediate/char_rnn_classification_tutorial
   :tags: Text

.. customcarditem::
   :header: 从零开始的自然语言处理：使用字符级 RNN 生成姓名
   :card_description: 2.在使用字符级循环神经网络对姓名进行分类之后，学习如何从语言中生成姓名。
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Generating-Names-with-a-Character-Level-RNN.png
   :link: intermediate/char_rnn_generation_tutorial.html
   :tags: Text

.. customcarditem::
   :header: 从零开始的自然语言处理：使用序列到序列网络和注意力进行翻译
   :card_description: 3.在这里我们编写自己的类和函数来预处理数据以执行我们的自然语言处理建模任务。
   :image: _static/img/thumbnails/cropped/NLP-From-Scratch-Translation-with-a-Sequence-to-Sequence-Network-and-Attention.png
   :link: intermediate/seq2seq_translation_tutorial.html
   :tags: Text

.. customcarditem::
   :header: 使用 Torchtext 进行文本分类
   :card_description: 学习如何使用 torchtext 库构建数据集并对文本进行分类。
   :image: _static/img/thumbnails/cropped/Text-Classification-with-TorchText.png
   :link: beginner/text_sentiment_ngrams_tutorial.html
   :tags: Text

.. customcarditem::
   :header: 使用 Transformer 进行语言翻译
   :card_description: 从零开始训练一个使用 Transformer 的语言翻译模型。
   :image: _static/img/thumbnails/cropped/Language-Translation-with-TorchText.png
   :link: beginner/translation_transformer.html
   :tags: Text

.. customcarditem::
   :header: 使用Torchtext预处理自定义文本数据集
   :card_description: 学习如何使用 torchtext 准备自定义数据集
   :image: _static/img/thumbnails/cropped/torch_text_logo.png
   :link: beginner/torchtext_custom_dataset_tutorial.html
   :tags: Text


.. ONNX

.. customcarditem::
   :header: （可选）使用 TorchDynamo 将 PyTorch 模型导出为 ONNX，并使用 ONNX Runtime 运行它
   :card_description: 构建一个 PyTorch 图像分类器模型，然后将其转换为 ONNX 格式，最后使用 ONNX Runtime 部署它。
   :image: _static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png
   :link: beginner/onnx/export_simple_model_to_onnx_tutorial.html
   :tags: Production,ONNX,Backends

.. customcarditem::
   :header: ONNX Registry 介绍
   :card_description: 演示如何通过使用 ONNX Registry 来解决不支持的操作符，从而实现端到端的流程。
   :image: _static/img/thumbnails/cropped/Exporting-PyTorch-Models-to-ONNX-Graphs.png
   :link: advanced/onnx_registry_tutorial.html
   :tags: Production,ONNX,Backends

.. Reinforcement Learning

.. customcarditem::
   :header: 强化学习 (DQN)
   :card_description: 学习如何使用 PyTorch 在 OpenAI Gym 的 CartPole-v0 任务上训练一个 Deep Q Learning（DQN）代理。
   :image: _static/img/cartpole.gif
   :link: intermediate/reinforcement_q_learning.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: 使用TorchRL进行强化学习（PPO）
   :card_description: Learn how to use PyTorch and TorchRL to train a Proximal Policy Optimization agent on the Inverted Pendulum task from Gym.
   :image: _static/img/invpendulum.gif
   :link: intermediate/reinforcement_ppo.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: 训练一个马里奥游戏的 RL Agent
   :card_description: Use PyTorch to train a Double Q-learning agent to play Mario.
   :image: _static/img/mario.gif
   :link: intermediate/mario_rl_tutorial.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Recurrent DQN
   :card_description: Use TorchRL to train recurrent policies
   :image: _static/img/rollout_recurrent.png
   :link: intermediate/dqn_with_rnn_tutorial.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Code a DDPG Loss
   :card_description: Use TorchRL to code a DDPG Loss
   :image: _static/img/half_cheetah.gif
   :link: advanced/coding_ddpg.html
   :tags: Reinforcement-Learning

.. customcarditem::
   :header: Writing your environment and transforms
   :card_description: Use TorchRL to code a Pendulum
   :image: _static/img/pendulum.gif
   :link: advanced/pendulum.html
   :tags: Reinforcement-Learning

.. Deploying PyTorch Models in Production


.. customcarditem::
   :header: 使用 Flask 在 Python 中部署 PyTorch REST API
   :card_description: Deploy a PyTorch model using Flask and expose a REST API for model inference using the example of a pretrained DenseNet 121 model which detects the image.
   :image: _static/img/thumbnails/cropped/Deploying-PyTorch-in-Python-via-a-REST-API-with-Flask.png
   :link: intermediate/flask_rest_api_tutorial.html
   :tags: Production

.. customcarditem::
   :header: Introduction to TorchScript
   :card_description: Introduction to TorchScript, an intermediate representation of a PyTorch model (subclass of nn.Module) that can then be run in a high-performance environment such as C++.
   :image: _static/img/thumbnails/cropped/Introduction-to-TorchScript.png
   :link: beginner/Intro_to_TorchScript_tutorial.html
   :tags: Production,TorchScript

.. customcarditem::
   :header: 在 C++ 中加载 TorchScript 模型
   :card_description:  Learn how PyTorch provides to go from an existing Python model to a serialized representation that can be loaded and executed purely from C++, with no dependency on Python.
   :image: _static/img/thumbnails/cropped/Loading-a-TorchScript-Model-in-Cpp.png
   :link: advanced/cpp_export.html
   :tags: Production,TorchScript

.. customcarditem::
   :header: (optional) Exporting a PyTorch Model to ONNX using TorchScript backend and Running it using ONNX Runtime
   :card_description:  Convert a model defined in PyTorch into the ONNX format and then run it with ONNX Runtime.
   :image: _static/img/thumbnails/cropped/optional-Exporting-a-Model-from-PyTorch-to-ONNX-and-Running-it-using-ONNX-Runtime.png
   :link: advanced/super_resolution_with_onnxruntime.html
   :tags: Production,ONNX

.. customcarditem::
   :header: Profiling PyTorch
   :card_description: Learn how to profile a PyTorch application
   :link: beginner/profiler.html
   :tags: Profiling

.. customcarditem::
   :header: Profiling PyTorch
   :card_description: Introduction to Holistic Trace Analysis
   :link: beginner/hta_intro_tutorial.html
   :tags: Profiling

.. customcarditem::
   :header: Profiling PyTorch
   :card_description: Trace Diff using Holistic Trace Analysis
   :link: beginner/hta_trace_diff_tutorial.html
   :tags: Profiling

.. Code Transformations with FX

.. customcarditem::
   :header: Building a Convolution/Batch Norm fuser in FX
   :card_description: Build a simple FX pass that fuses batch norm into convolution to improve performance during inference.
   :image: _static/img/thumbnails/cropped/Deploying-PyTorch-in-Python-via-a-REST-API-with-Flask.png
   :link: intermediate/fx_conv_bn_fuser.html
   :tags: FX

.. customcarditem::
   :header: Building a Simple Performance Profiler with FX
   :card_description: Build a simple FX interpreter to record the runtime of op, module, and function calls and report statistics
   :image: _static/img/thumbnails/cropped/Deploying-PyTorch-in-Python-via-a-REST-API-with-Flask.png
   :link: intermediate/fx_profiling_tutorial.html
   :tags: FX

.. Frontend APIs

.. customcarditem::
   :header: (beta) Channels Last Memory Format in PyTorch
   :card_description: Get an overview of Channels Last memory format and understand how it is used to order NCHW tensors in memory preserving dimensions.
   :image: _static/img/thumbnails/cropped/experimental-Channels-Last-Memory-Format-in-PyTorch.png
   :link: intermediate/memory_format_tutorial.html
   :tags: Memory-Format,Best-Practice,Frontend-APIs

.. customcarditem::
   :header: Using the PyTorch C++ Frontend
   :card_description: Walk through an end-to-end example of training a model with the C++ frontend by training a DCGAN – a kind of generative model – to generate images of MNIST digits.
   :image: _static/img/thumbnails/cropped/Using-the-PyTorch-Cpp-Frontend.png
   :link: advanced/cpp_frontend.html
   :tags: Frontend-APIs,C++

.. customcarditem::
   :header: Custom C++ and CUDA Extensions
   :card_description:  Create a neural network layer with no parameters using numpy. Then use scipy to create a neural network layer that has learnable weights.
   :image: _static/img/thumbnails/cropped/Custom-Cpp-and-CUDA-Extensions.png
   :link: advanced/cpp_extension.html
   :tags: Extending-PyTorch,Frontend-APIs,C++,CUDA

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Operators
   :card_description:  Implement a custom TorchScript operator in C++, how to build it into a shared library, how to use it in Python to define TorchScript models and lastly how to load it into a C++ application for inference workloads.
   :image: _static/img/thumbnails/cropped/Extending-TorchScript-with-Custom-Cpp-Operators.png
   :link: advanced/torch_script_custom_ops.html
   :tags: Extending-PyTorch,Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Extending TorchScript with Custom C++ Classes
   :card_description: This is a continuation of the custom operator tutorial, and introduces the API we’ve built for binding C++ classes into TorchScript and Python simultaneously.
   :image: _static/img/thumbnails/cropped/Extending-TorchScript-with-Custom-Cpp-Classes.png
   :link: advanced/torch_script_custom_classes.html
   :tags: Extending-PyTorch,Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Dynamic Parallelism in TorchScript
   :card_description: This tutorial introduces the syntax for doing *dynamic inter-op parallelism* in TorchScript.
   :image: _static/img/thumbnails/cropped/TorchScript-Parallelism.jpg
   :link: advanced/torch-script-parallelism.html
   :tags: Frontend-APIs,TorchScript,C++

.. customcarditem::
   :header: Real Time Inference on Raspberry Pi 4
   :card_description: This tutorial covers how to run quantized and fused models on a Raspberry Pi 4 at 30 fps.
   :image: _static/img/thumbnails/cropped/realtime_rpi.png
   :link: intermediate/realtime_rpi.html
   :tags: TorchScript,Model-Optimization,Image/Video,Quantization

.. customcarditem::
   :header: Autograd in C++ Frontend
   :card_description: The autograd package helps build flexible and dynamic nerural netorks. In this tutorial, exploreseveral examples of doing autograd in PyTorch C++ frontend
   :image: _static/img/thumbnails/cropped/Autograd-in-Cpp-Frontend.png
   :link: advanced/cpp_autograd.html
   :tags: Frontend-APIs,C++

.. customcarditem::
   :header: Registering a Dispatched Operator in C++
   :card_description: The dispatcher is an internal component of PyTorch which is responsible for figuring out what code should actually get run when you call a function like torch::add.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/dispatcher.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. customcarditem::
   :header: Extending Dispatcher For a New Backend in C++
   :card_description: Learn how to extend the dispatcher to add a new device living outside of the pytorch/pytorch repo and maintain it to keep in sync with native PyTorch devices.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/extend_dispatcher.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. customcarditem::
   :header: Facilitating New Backend Integration by PrivateUse1
   :card_description: Learn how to integrate a new backend living outside of the pytorch/pytorch repo and maintain it to keep in sync with the native PyTorch backend.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: advanced/privateuseone.html
   :tags: Extending-PyTorch,Frontend-APIs,C++

.. customcarditem::
   :header: Custom Function Tutorial: Double Backward
   :card_description: Learn how to write a custom autograd Function that supports double backward.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/custom_function_double_backward_tutorial.html
   :tags: Extending-PyTorch,Frontend-APIs

.. customcarditem::
   :header: Custom Function Tutorial: Fusing Convolution and Batch Norm
   :card_description: Learn how to create a custom autograd Function that fuses batch norm into a convolution to improve memory usage.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/custom_function_conv_bn_tutorial.html
   :tags: Extending-PyTorch,Frontend-APIs

.. customcarditem::
   :header: Forward-mode Automatic Differentiation
   :card_description: Learn how to use forward-mode automatic differentiation.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/forward_ad_usage.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Jacobians, Hessians, hvp, vhp, and more
   :card_description: Learn how to compute advanced autodiff quantities using torch.func
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/jacobians_hessians.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Model Ensembling
   :card_description: Learn how to ensemble models using torch.vmap
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/ensembling.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Per-Sample-Gradients
   :card_description: Learn how to compute per-sample-gradients using torch.func
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/per_sample_grads.html
   :tags: Frontend-APIs

.. customcarditem::
   :header: Neural Tangent Kernels
   :card_description: Learn how to compute neural tangent kernels using torch.func
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/neural_tangent_kernels.html
   :tags: Frontend-APIs

.. Model Optimization

.. customcarditem::
   :header: Performance Profiling in PyTorch
   :card_description: Learn how to use the PyTorch Profiler to benchmark your module's performance.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: beginner/profiler.html
   :tags: Model-Optimization,Best-Practice,Profiling

.. customcarditem::
   :header: Performance Profiling in TensorBoard
   :card_description: Learn how to use the TensorBoard plugin to profile and analyze your model's performance.
   :image: _static/img/thumbnails/cropped/profiler.png
   :link: intermediate/tensorboard_profiler_tutorial.html
   :tags: Model-Optimization,Best-Practice,Profiling,TensorBoard

.. customcarditem::
   :header: Hyperparameter Tuning Tutorial
   :card_description: Learn how to use Ray Tune to find the best performing set of hyperparameters for your model.
   :image: _static/img/ray-tune.png
   :link: beginner/hyperparameter_tuning_tutorial.html
   :tags: Model-Optimization,Best-Practice

.. customcarditem::
   :header: Parametrizations Tutorial
   :card_description: Learn how to use torch.nn.utils.parametrize to put constraints on your parameters (e.g. make them orthogonal, symmetric positive definite, low-rank...)
   :image: _static/img/thumbnails/cropped/parametrizations.png
   :link: intermediate/parametrizations.html
   :tags: Model-Optimization,Best-Practice

.. customcarditem::
   :header: Pruning Tutorial
   :card_description: Learn how to use torch.nn.utils.prune to sparsify your neural networks, and how to extend it to implement your own custom pruning technique.
   :image: _static/img/thumbnails/cropped/Pruning-Tutorial.png
   :link: intermediate/pruning_tutorial.html
   :tags: Model-Optimization,Best-Practice

.. customcarditem::
   :header: How to save memory by fusing the optimizer step into the backward pass
   :card_description: Learn a memory-saving technique through fusing the optimizer step into the backward pass using memory snapshots.
   :image: _static/img/thumbnails/cropped/pytorch-logo.png
   :link: intermediate/optimizer_step_in_backward_tutorial.html
   :tags: Model-Optimization,Best-Practice,CUDA,Frontend-APIs

.. customcarditem::
   :header: (beta) Accelerating BERT with semi-structured sparsity
   :card_description: Train BERT, prune it to be 2:4 sparse, and then accelerate it to achieve 2x inference speedups with semi-structured sparsity and torch.compile. 
   :image: _static/img/thumbnails/cropped/Pruning-Tutorial.png
   :link: advanced/semi_structured_sparse.html
   :tags: Text,Model-Optimization

.. customcarditem::
   :header: (beta) Dynamic Quantization on an LSTM Word Language Model
   :card_description: Apply dynamic quantization, the easiest form of quantization, to a LSTM-based next word prediction model.
   :image: _static/img/thumbnails/cropped/experimental-Dynamic-Quantization-on-an-LSTM-Word-Language-Model.png
   :link: advanced/dynamic_quantization_tutorial.html
   :tags: Text,Quantization,Model-Optimization

.. customcarditem::
   :header: (beta) Dynamic Quantization on BERT
   :card_description: Apply the dynamic quantization on a BERT (Bidirectional Embedding Representations from Transformers) model.
   :image: _static/img/thumbnails/cropped/experimental-Dynamic-Quantization-on-BERT.png
   :link: intermediate/dynamic_quantization_bert_tutorial.html
   :tags: Text,Quantization,Model-Optimization

.. customcarditem::
   :header: (beta) Quantized Transfer Learning for Computer Vision Tutorial
   :card_description: Extends the Transfer Learning for Computer Vision Tutorial using a quantized model.
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: intermediate/quantized_transfer_learning_tutorial.html
   :tags: Image/Video,Quantization,Model-Optimization

.. customcarditem::
   :header: (beta) Static Quantization with Eager Mode in PyTorch
   :card_description: This tutorial shows how to do post-training static quantization.
   :image: _static/img/thumbnails/cropped/60-min-blitz.png
   :link: advanced/static_quantization_tutorial.html
   :tags: Quantization

.. customcarditem::
   :header: Grokking PyTorch Intel CPU Performance from First Principles
   :card_description: A case study on the TorchServe inference framework optimized with Intel® Extension for PyTorch.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/torchserve_with_ipex
   :tags: Model-Optimization,Production

.. customcarditem::
   :header: Grokking PyTorch Intel CPU Performance from First Principles (Part 2)
   :card_description: A case study on the TorchServe inference framework optimized with Intel® Extension for PyTorch (Part 2).
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/torchserve_with_ipex_2
   :tags: Model-Optimization,Production

.. customcarditem::
   :header: Multi-Objective Neural Architecture Search with Ax
   :card_description: Learn how to use Ax to search over architectures find optimal tradeoffs between accuracy and latency.
   :image: _static/img/ax_logo.png
   :link: intermediate/ax_multiobjective_nas_tutorial.html
   :tags: Model-Optimization,Best-Practice,Ax,TorchX

.. customcarditem::
   :header: torch.compile Tutorial
   :card_description: Speed up your models with minimal code changes using torch.compile, the latest PyTorch compiler solution.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/torch_compile_tutorial.html
   :tags: Model-Optimization

.. customcarditem::
   :header: Inductor CPU Backend Debugging and Profiling
   :card_description: Learn the usage, debugging and performance profiling for ``torch.compile`` with Inductor CPU backend.
   :image: _static/img/thumbnails/cropped/generic-pytorch-logo.png
   :link: intermediate/inductor_debug_cpu.html
   :tags: Model-Optimization

.. customcarditem::
   :header: (beta) Implementing High-Performance Transformers with SCALED DOT PRODUCT ATTENTION
   :card_description: This tutorial explores the new torch.nn.functional.scaled_dot_product_attention and how it can be used to construct Transformer components.
   :image: _static/img/thumbnails/cropped/pytorch-logo.png
   :link: intermediate/scaled_dot_product_attention_tutorial.html
   :tags: Model-Optimization,Attention,Transformer

.. customcarditem::
   :header: Knowledge Distillation in Convolutional Neural Networks
   :card_description:  Learn how to improve the accuracy of lightweight models using more powerful models as teachers.
   :image: _static/img/thumbnails/cropped/knowledge_distillation_pytorch_logo.png
   :link: beginner/knowledge_distillation_tutorial.html
   :tags: Model-Optimization,Image/Video

.. Parallel-and-Distributed-Training



.. customcarditem::
   :header: PyTorch Distributed Overview
   :card_description: Briefly go over all concepts and features in the distributed package. Use this document to find the distributed training technology that can best serve your application.
   :image: _static/img/thumbnails/cropped/PyTorch-Distributed-Overview.png
   :link: beginner/dist_overview.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Distributed Data Parallel in PyTorch - Video Tutorials
   :card_description: This series of video tutorials walks you through distributed training in PyTorch via DDP.
   :image: _static/img/thumbnails/cropped/PyTorch-Distributed-Overview.png
   :link: beginner/ddp_series_intro.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Single-Machine Model Parallel Best Practices
   :card_description:  Learn how to implement model parallel, a distributed training technique which splits a single model onto different GPUs, rather than replicating the entire model on each GPU
   :image: _static/img/thumbnails/cropped/Model-Parallel-Best-Practices.png
   :link: intermediate/model_parallel_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Getting Started with Distributed Data Parallel
   :card_description: Learn the basics of when to use distributed data paralle versus data parallel and work through an example to set it up.
   :image: _static/img/thumbnails/cropped/Getting-Started-with-Distributed-Data-Parallel.png
   :link: intermediate/ddp_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Writing Distributed Applications with PyTorch
   :card_description: Set up the distributed package of PyTorch, use the different communication strategies, and go over some the internals of the package.
   :image: _static/img/thumbnails/cropped/Writing-Distributed-Applications-with-PyTorch.png
   :link: intermediate/dist_tuto.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Large Scale Transformer model training with Tensor Parallel
   :card_description: Learn how to train large models with Tensor Parallel package.
   :image: _static/img/thumbnails/cropped/Large-Scale-Transformer-model-training-with-Tensor-Parallel.png
   :link: intermediate/TP_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Customize Process Group Backends Using Cpp Extensions
   :card_description: Extend ProcessGroup with custom collective communication implementations.
   :image: _static/img/thumbnails/cropped/Customize-Process-Group-Backends-Using-Cpp-Extensions.png
   :link: intermediate/process_group_cpp_extension_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Getting Started with Distributed RPC Framework
   :card_description: Learn how to build distributed training using the torch.distributed.rpc package.
   :image: _static/img/thumbnails/cropped/Getting Started with Distributed-RPC-Framework.png
   :link: intermediate/rpc_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Implementing a Parameter Server Using Distributed RPC Framework
   :card_description: Walk through a through a simple example of implementing a parameter server using PyTorch’s Distributed RPC framework.
   :image: _static/img/thumbnails/cropped/Implementing-a-Parameter-Server-Using-Distributed-RPC-Framework.png
   :link: intermediate/rpc_param_server_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Distributed Pipeline Parallelism Using RPC
   :card_description: Demonstrate how to implement distributed pipeline parallelism using RPC
   :image: _static/img/thumbnails/cropped/Distributed-Pipeline-Parallelism-Using-RPC.png
   :link: intermediate/dist_pipeline_parallel_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Implementing Batch RPC Processing Using Asynchronous Executions
   :card_description: Learn how to use rpc.functions.async_execution to implement batch RPC
   :image: _static/img/thumbnails/cropped/Implementing-Batch-RPC-Processing-Using-Asynchronous-Executions.png
   :link: intermediate/rpc_async_execution.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Combining Distributed DataParallel with Distributed RPC Framework
   :card_description: Walk through a through a simple example of how to combine distributed data parallelism with distributed model parallelism.
   :image: _static/img/thumbnails/cropped/Combining-Distributed-DataParallel-with-Distributed-RPC-Framework.png
   :link: advanced/rpc_ddp_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Training Transformer models using Distributed Data Parallel and Pipeline Parallelism
   :card_description: Walk through a through a simple example of how to train a transformer model using Distributed Data Parallel and Pipeline Parallelism
   :image: _static/img/thumbnails/cropped/Training-Transformer-Models-using-Distributed-Data-Parallel-and-Pipeline-Parallelism.png
   :link: advanced/ddp_pipeline.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Getting Started with Fully Sharded Data Parallel(FSDP)
   :card_description: Learn how to train models with Fully Sharded Data Parallel package.
   :image: _static/img/thumbnails/cropped/Getting-Started-with-FSDP.png
   :link: intermediate/FSDP_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. customcarditem::
   :header: Advanced Model Training with Fully Sharded Data Parallel (FSDP)
   :card_description: Explore advanced model training with Fully Sharded Data Parallel package.
   :image: _static/img/thumbnails/cropped/Getting-Started-with-FSDP.png
   :link: intermediate/FSDP_adavnced_tutorial.html
   :tags: Parallel-and-Distributed-Training

.. Edge

.. customcarditem::
   :header: Exporting to ExecuTorch Tutorial
   :card_description: Learn about how to use ExecuTorch, a unified ML stack for lowering PyTorch models to edge devices.
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html
   :tags: Edge

.. customcarditem::
   :header: Running an ExecuTorch Model in C++ Tutorial
   :card_description: Learn how to load and execute an ExecuTorch model in C++
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://pytorch.org/executorch/stable/running-a-model-cpp-tutorial.html
   :tags: Edge

.. customcarditem::
   :header: Using the ExecuTorch SDK to Profile a Model
   :card_description: Explore how to use the ExecuTorch SDK to profile, debug, and visualize ExecuTorch models 
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://pytorch.org/executorch/stable/tutorials/sdk-integration-tutorial.html
   :tags: Edge

.. customcarditem::
   :header: Building an ExecuTorch iOS Demo App
   :card_description: Explore how to set up the ExecuTorch iOS Demo App, which uses the MobileNet v3 model to process live camera images leveraging three different backends: XNNPACK, Core ML, and Metal Performance Shaders (MPS).
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://pytorch.org/executorch/stable/demo-apps-ios.html
   :tags: Edge

.. customcarditem::
   :header: Building an ExecuTorch Android Demo App
   :card_description: Learn how to set up the ExecuTorch Android Demo App for image segmentation tasks using the DeepLab v3 model and XNNPACK FP32 backend.
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://pytorch.org/executorch/stable/demo-apps-android.html
   :tags: Edge

.. customcarditem::
   :header: Lowering a Model as a Delegate
   :card_description: Learn to accelerate your program using ExecuTorch by applying delegates through three methods: lowering the whole module, composing it with another module, and partitioning parts of a module.
   :image: _static/img/ExecuTorch-Logo-cropped.svg
   :link: https://pytorch.org/executorch/stable/examples-end-to-end-to-lower-model-to-delegate.html
   :tags: Edge


.. Recommendation Systems

.. customcarditem::
   :header: Introduction to TorchRec
   :card_description: TorchRec is a PyTorch domain library built to provide common sparsity & parallelism primitives needed for large-scale recommender systems.
   :image: _static/img/thumbnails/torchrec.png
   :link: intermediate/torchrec_tutorial.html
   :tags: TorchRec,Recommender

.. customcarditem::
   :header: Exploring TorchRec sharding
   :card_description: This tutorial covers the sharding schemes of embedding tables by using <code>EmbeddingPlanner</code> and <code>DistributedModelParallel</code> API.
   :image: _static/img/thumbnails/torchrec.png
   :link: advanced/sharding.html
   :tags: TorchRec,Recommender

.. Multimodality

.. customcarditem::
   :header: Introduction to TorchMultimodal
   :card_description: TorchMultimodal is a library that provides models, primitives and examples for training multimodal tasks
   :image: _static/img/thumbnails/torchrec.png
   :link: beginner/flava_finetuning_tutorial.html
   :tags: TorchMultimodal


.. End of tutorial card section

.. raw:: html

    </div>

    <div class="pagination d-flex justify-content-center"></div>

    </div>

    </div>
    <br>
    <br>


更多资源
============================

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :header: PyTorch 示例
   :description: 视觉、文本、强化学习的 PyTorch 示例，可以将其融入现有工作内容。
   :button_link: https://pytorch.org/examples?utm_source=examples&utm_medium=examples-landing
   :button_text: 查看示例

.. customcalloutitem::
   :header: PyTorch Cheat Sheet
   :description: PyTorch 基础内容速览。
   :button_link: beginner/ptcheat.html
   :button_text: 打开

.. customcalloutitem::
   :header: GitHub 上的教程
   :description: 获取 GitHub 上的 PyTorch 教程。
   :button_link: https://github.com/pytorch/tutorials
   :button_text: 打开 GitHub

.. customcalloutitem::
   :header: Google Colab 上运行教程
   :description: 学习如何将教程数据复制到 Google Drive，以便您可以在 Google Colab 上运行教程。
   :button_link: beginner/colab.html
   :button_text: 打开

.. End of callout section

.. raw:: html

        </div>
    </div>

    <div style='clear:both'></div>

.. -----------------------------------------
.. Page TOC
.. -----------------------------------------
.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: PyTorch 示例

   所有示例 <recipes/recipes_index>
   原型示例 <prototype/prototype_index>

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: PyTorch 入门

   beginner/basics/intro
   beginner/basics/quickstart_tutorial
   beginner/basics/tensorqs_tutorial
   beginner/basics/data_tutorial
   beginner/basics/transforms_tutorial
   beginner/basics/buildmodel_tutorial
   beginner/basics/autogradqs_tutorial
   beginner/basics/optimization_tutorial
   beginner/basics/saveloadrun_tutorial

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: PyTorch 视频教程

   beginner/introyt
   beginner/introyt/introyt1_tutorial
   beginner/introyt/tensors_deeper_tutorial
   beginner/introyt/autogradyt_tutorial
   beginner/introyt/modelsyt_tutorial
   beginner/introyt/tensorboardyt_tutorial
   beginner/introyt/trainingyt
   beginner/introyt/captumyt

.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: 学习 PyTorch

   beginner/deep_learning_60min_blitz
   beginner/pytorch_with_examples
   beginner/nn_tutorial
   intermediate/tensorboard_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 图片与视频

   intermediate/torchvision_tutorial
   beginner/transfer_learning_tutorial
   beginner/fgsm_tutorial
   beginner/dcgan_faces_tutorial
   intermediate/spatial_transformer_tutorial
   beginner/vt_tutorial
   intermediate/tiatoolbox_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 音频

   beginner/audio_io_tutorial
   beginner/audio_resampling_tutorial
   beginner/audio_data_augmentation_tutorial
   beginner/audio_feature_extractions_tutorial
   beginner/audio_feature_augmentation_tutorial
   beginner/audio_datasets_tutorial
   intermediate/speech_recognition_pipeline_tutorial
   intermediate/speech_command_classification_with_torchaudio_tutorial
   intermediate/text_to_speech_with_torchaudio
   intermediate/forced_alignment_with_torchaudio_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 文本

   beginner/bettertransformer_tutorial
   intermediate/char_rnn_classification_tutorial
   intermediate/char_rnn_generation_tutorial
   intermediate/seq2seq_translation_tutorial
   beginner/text_sentiment_ngrams_tutorial
   beginner/translation_transformer
   beginner/torchtext_custom_dataset_tutorial


.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 后端

   beginner/onnx/intro_onnx

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 强化学习

   intermediate/reinforcement_q_learning
   intermediate/reinforcement_ppo
   intermediate/mario_rl_tutorial
   advanced/pendulum

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 部署 PyTorch 模型

   beginner/onnx/intro_onnx
   intermediate/flask_rest_api_tutorial
   beginner/Intro_to_TorchScript_tutorial
   advanced/cpp_export
   advanced/super_resolution_with_onnxruntime
   intermediate/realtime_rpi

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 分析 PyTorch

   beginner/profiler
   beginner/hta_intro_tutorial
   beginner/hta_trace_diff_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: FX 代码转换

   intermediate/fx_conv_bn_fuser
   intermediate/fx_profiling_tutorial

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 前端 APIs

   intermediate/memory_format_tutorial
   intermediate/forward_ad_usage
   intermediate/jacobians_hessians
   intermediate/ensembling
   intermediate/per_sample_grads
   intermediate/neural_tangent_kernels.py
   advanced/cpp_frontend
   advanced/torch-script-parallelism
   advanced/cpp_autograd

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: PyTorch 扩展

   intermediate/custom_function_double_backward_tutorial
   intermediate/custom_function_conv_bn_tutorial
   advanced/cpp_extension
   advanced/torch_script_custom_ops
   advanced/torch_script_custom_classes
   advanced/dispatcher
   advanced/extend_dispatcher
   advanced/privateuseone

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 优化模型

   beginner/profiler
   intermediate/tensorboard_profiler_tutorial
   beginner/hyperparameter_tuning_tutorial
   beginner/vt_tutorial
   intermediate/parametrizations
   intermediate/pruning_tutorial
   advanced/dynamic_quantization_tutorial
   intermediate/dynamic_quantization_bert_tutorial
   intermediate/quantized_transfer_learning_tutorial
   advanced/static_quantization_tutorial
   intermediate/torchserve_with_ipex
   intermediate/torchserve_with_ipex_2
   intermediate/nvfuser_intro_tutorial
   intermediate/ax_multiobjective_nas_tutorial
   intermediate/torch_compile_tutorial
   intermediate/inductor_debug_cpu
   intermediate/scaled_dot_product_attention_tutorial
   beginner/knowledge_distillation_tutorial


.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 分布式并行训练

   distributed/home
   beginner/dist_overview
   beginner/ddp_series_intro
   intermediate/model_parallel_tutorial
   intermediate/ddp_tutorial
   intermediate/dist_tuto
   intermediate/FSDP_tutorial
   intermediate/FSDP_adavnced_tutorial
   intermediate/TP_tutorial
   intermediate/process_group_cpp_extension_tutorial
   intermediate/rpc_tutorial
   intermediate/rpc_param_server_tutorial
   intermediate/dist_pipeline_parallel_tutorial
   intermediate/rpc_async_execution
   advanced/rpc_ddp_tutorial
   advanced/ddp_pipeline
   advanced/generic_join

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: Edge with ExecuTorch

   导出到 ExecuTorch 教程 <https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html>
   使用C++运行 ExecuTorch 教程 < https://pytorch.org/executorch/stable/running-a-model-cpp-tutorial.html>
   使用 ExecuTorch SDK 分析模型 <https://pytorch.org/executorch/stable/tutorials/sdk-integration-tutorial.html>
   构建 ExecuTorch iOS Demo App <https://pytorch.org/executorch/stable/demo-apps-ios.html>
   构建 ExecuTorch Android Demo App <https://pytorch.org/executorch/stable/demo-apps-android.html>
   Lowering a Model as a Delegate <https://pytorch.org/executorch/stable/examples-end-to-end-to-lower-model-to-delegate.html>   

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 推荐系统

   intermediate/torchrec_tutorial
   advanced/sharding

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 多模态

   beginner/flava_finetuning_tutorial
