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

Let’s do a sanity check on the equivalence of the two models:

::

   unscripted_output = r18(dummy_input)         # Get the unscripted model's prediction...
   scripted_output = r18_scripted(dummy_input)  # ...and do the same for the scripted version

   unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
   scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

   print('Python model top 5 results:\n  {}'.format(unscripted_top5))
   print('TorchScript model top 5 results:\n  {}'.format(scripted_top5))

You should see that both versions of the model give the same results:

::

   Python model top 5 results:
     tensor([[463, 600, 731, 899, 898]])
   TorchScript model top 5 results:
     tensor([[463, 600, 731, 899, 898]])

With that check confirmed, go ahead and save the model:

::

   r18_scripted.save('r18_scripted.pt')

Loading TorchScript Models in C++
---------------------------------

Create the following C++ file and name it ``ts-infer.cpp``:

.. code:: cpp

   #include <torch/script.h>
   #include <torch/nn/functional/activation.h>


   int main(int argc, const char* argv[]) {
       if (argc != 2) {
           std::cerr << "usage: ts-infer <path-to-exported-model>\n";
           return -1;
       }

       std::cout << "Loading model...\n";

       // deserialize ScriptModule
       torch::jit::script::Module module;
       try {
           module = torch::jit::load(argv[1]);
       } catch (const c10::Error& e) {
           std::cerr << "Error loading model\n";
           std::cerr << e.msg_without_backtrace();
           return -1;
       }

       std::cout << "Model loaded successfully\n";

       torch::NoGradGuard no_grad; // ensures that autograd is off
       module.eval(); // turn off dropout and other training-time layers/functions

       // create an input "image"
       std::vector<torch::jit::IValue> inputs;
       inputs.push_back(torch::rand({1, 3, 224, 224}));

       // execute model and package output as tensor
       at::Tensor output = module.forward(inputs).toTensor();

       namespace F = torch::nn::functional;
       at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
       std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
       at::Tensor top5 = std::get<1>(top5_tensor);

       std::cout << top5[0] << "\n";

       std::cout << "\nDONE\n";
       return 0;
   }

This program:

-  Loads the model you specify on the command line
- Creates a dummy “image” input tensor
- Performs inference on the input

Also, notice that there is no dependency on TorchVision in this code.
The saved version of your TorchScript model has your learning weights
*and* your computation graph - nothing else is needed.

Building and Running Your C++ Inference Engine
----------------------------------------------

Create the following ``CMakeLists.txt`` file:

::

   cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
   project(custom_ops)

   find_package(Torch REQUIRED)

   add_executable(ts-infer ts-infer.cpp)
   target_link_libraries(ts-infer "${TORCH_LIBRARIES}")
   set_property(TARGET ts-infer PROPERTY CXX_STANDARD 11)

Make the program:

::

   cmake -DCMAKE_PREFIX_PATH=<path to your libtorch installation>
   make

Now, we can run inference in C++, and verify that we get a result:

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

Important Resources
-------------------

-  `pytorch.org`_ for installation instructions, and more documentation
   and tutorials.
-  `Introduction to TorchScript tutorial`_ for a deeper initial
   exposition of TorchScript
-  `Full TorchScript documentation`_ for complete TorchScript language
   and API reference

.. _pytorch.org: https://pytorch.org/
.. _Introduction to TorchScript tutorial: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
.. _Full TorchScript documentation: https://pytorch.org/docs/stable/jit.html
.. _在 C++ 中加载 TorchScript 模型教程: https://pytorch.org/tutorials/advanced/cpp_export.html
.. _full TorchScript documentation: https://pytorch.org/docs/stable/jit.html
