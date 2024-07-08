TorchScript 部署
==========================

在本教程中,您将学习:

-  What TorchScript is
-  How to export your trained model in TorchScript format
-  How to load your TorchScript model in C++ and do inference
- TorchScript 是什么
- 如何将训练好的模型导出为 TorchScript 格式
- 如何在 C++ 中加载 TorchScript 模型并进行推理

环境要求
------------

-  PyTorch 1.5
-  TorchVision 0.6.0
-  libtorch 1.5
-  C++ compiler

安装这三个 PyTorch 组件的说明可在 `pytorch.org_` 上找到。C++ 编译器则取决于您的平台。



什么是 TorchScript?
--------------------

**TorchScript** 是 PyTorch 模型( ``nn.Module`` 的子类)的中间表示,可以在高性能环境(如 C++)中运行。
它是 Python 的一个高性能子集,旨在被 **PyTorch JIT 编译器** 使用,后者会对模型的计算进行运行时优化。
TorchScript 是使用 PyTorch 模型进行大规模推理的推荐模型格式。更多信息,
请参阅 `pytorch.org_` 上的 `PyTorch TorchScript 入门教程`、 `在 C++ 中加载 TorchScript 模型教程`
和 `完整的 TorchScript 文档_` 。

如何导出模型
------------------------

作为示例,让我们使用一个预训练的视觉模型。TorchVision 中的所有预训练模型都与 TorchScript 兼容。

运行以下 Python 3 代码,可以在脚本中或从 REPL 中运行:

.. code:: python3

   import torch
   import torch.nn.functional as F
   import torchvision.models as models

   r18 = models.resnet18(pretrained=True)       # 现在我们有一个预训练模型的实例
   r18_scripted = torch.jit.script(r18)         # *** 这是 TorchScript 导出
   dummy_input = torch.rand(1, 3, 224, 224)     # 快速测试一下

让我们快速检查一下两个模型的等价性:

::

   unscripted_output = r18(dummy_input)         # Get the unscripted model's prediction...
   scripted_output = r18_scripted(dummy_input)  # ...and do the same for the scripted version

   unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
   scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

   print('Python model top 5 results:\n  {}'.format(unscripted_top5))
   print('TorchScript model top 5 results:\n  {}'.format(scripted_top5))

会看到两个版本的模型给出相同的结果:

::

   Python model top 5 results:
     tensor([[463, 600, 731, 899, 898]])
   TorchScript model top 5 results:
     tensor([[463, 600, 731, 899, 898]])

确认检查通过后,继续保存模型:

::

   r18_scripted.save('r18_scripted.pt')

在 C++ 中加载 TorchScript 模型
---------------------------------

创建以下 C++ 文件,并将其命名为 ``ts-infer.cpp``:

.. code:: cpp

   #include <torch/script.h>
   #include <torch/nn/functional/activation.h>


   int main(int argc, const char* argv[]) {
       if (argc != 2) {
           std::cerr << "usage: ts-infer <path-to-exported-model>\n";
           return -1;
       }

       std::cout << "Loading model...\n";

       // 反序列化 ScriptModule
       torch::jit::script::Module module;
       try {
           module = torch::jit::load(argv[1]);
       } catch (const c10::Error& e) {
           std::cerr << "Error loading model\n";
           std::cerr << e.msg_without_backtrace();
           return -1;
       }

       std::cout << "Model loaded successfully\n";

       torch::NoGradGuard no_grad; // 确保自动梯度计算关闭
       module.eval(); // 关闭 dropout 和其他训练时层/函数

       // 创建一个输入"图像"
       std::vector<torch::jit::IValue> inputs;
       inputs.push_back(torch::rand({1, 3, 224, 224}));

       // 执行模型并将输出打包为张量
       at::Tensor output = module.forward(inputs).toTensor();

       namespace F = torch::nn::functional;
       at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
       std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
       at::Tensor top5 = std::get<1>(top5_tensor);

       std::cout << top5[0] << "\n";

       std::cout << "\nDONE\n";
       return 0;
   }

程序步骤:

- 加载您在命令行上指定的模型
- 创建一个虚拟的"图像"输入张量
- 对输入执行推理

另外,请注意这段代码中没有依赖 TorchVision。
保存的 TorchScript 模型包含您的学习权重和您的计算图。

构建和运行您的 C++ 推理引擎
----------------------------------------------

创建以下 ``CMakeLists.txt`` 文件:

::

   cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
   project(custom_ops)

   find_package(Torch REQUIRED)

   add_executable(ts-infer ts-infer.cpp)
   target_link_libraries(ts-infer "${TORCH_LIBRARIES}")
   set_property(TARGET ts-infer PROPERTY CXX_STANDARD 11)

构建程序:

::

   cmake -DCMAKE_PREFIX_PATH=<path to your libtorch installation>
   make

现在,我们可以在 C++ 中运行推理,并验证我们得到结果:

::

   $ ./ts-infer r18_scripted.pt
   Loading model...
   Model loaded successfully
    418
    845
    111
    892
    644
   [ CPULongType{5} ]

   DONE

其他资源
-------------------

-  `pytorch.org`_  查看安装说明和更多文档和教程。
-  `TorchScript 入门教程`_ 对 TorchScript 进一步了解
-  `TorchScript 文档`_ 查看完整的 TorchScript 语言和 API 参考

.. _pytorch.org: https://pytorch.org/
.. _TorchScript 入门教程: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
.. _TorchScript 文档: https://pytorch.org/docs/stable/jit.html