使用Intel® Neural Compressor实现PyTorch的简易量化
==================================================

概述
--------

大多数深度学习应用程序在推理时使用32位浮点精度。但是由于显著的性能提升，低精度数据类型（尤其是int8）正受到越来越多的关注。采用低精度时的一个主要问题是如何轻松地减轻可能的精度损失并达到预定的精度要求。

Intel® Neural Compressor 旨在通过扩展 PyTorch 的精度驱动自动调优策略来解决上述问题，帮助用户在Intel硬件上快速找到最佳量化模型，
包括Intel Deep Learning Boost（ `Intel DL Boost <https://www.intel.com/content/www/us/en/artificial-intelligence/deep-learning-boost.html>`_）
和Intel Advanced Matrix Extensions（ `Intel AMX <https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-amx-instructions/intrinsics-for-amx-tile-instructions.html>`_）。

Intel® Neural Compressor已作为开源项目发布在 `Github <https://github.com/intel/neural-compressor>`_上。

特性
--------

- **易用的Python API：** Intel® Neural Compressor 提供简单的前端Python API和实用工具，用户只需更改几行代码即可进行神经网络压缩。
通常只需要在原始代码中添加5到6个子句。

- **量化：** Intel® Neural Compressor 支持在 PyTorch fx 图模式和 eager 模式下进行精度驱动的自动调优过程，
包括训练后静态量化、训练后动态量化和量化感知训练。

*本教程主要关注量化部分。关于如何使用 Intel® Neural Compressor 进行剪枝和蒸馏，请参考 Intel® Neural Compressor github仓库中的相应文档。*

入门
---------------

安装
~~~~~~~~~~~~

.. code:: bash

    # 从pip安装稳定版本
    pip install neural-compressor

    # 从pip安装每日构建版本
    pip install -i https://test.pypi.org/simple/ neural-compressor

    # 从conda安装稳定版本
    conda install neural-compressor -c conda-forge -c intel

*支持的Python版本为3.6、3.7、3.8或3.9*

用法
~~~~~~

用户只需进行少量代码更改即可开始使用 Intel® Neural Compressor 量化API。支持 PyTorch fx 图模式和 eager 模式。

Intel® Neural Compressor 接受一个 FP32 模型和一个 yaml 配置文件作为输入。要构建量化过程，用户可以通过 yaml 配置文件
或 Python API 指定以下设置：

1. 校准数据加载器（静态量化需要）
2. 评估数据加载器
3. 评估指标

Intel® Neural Compressor 支持一些常用的数据加载器和评估指标。关于如何在 yaml 配置文件中配置它们，用户可以参考
`内置数据集 <https://github.com/intel/neural-compressor/blob/master/docs/dataset.md>`_。

如果用户想使用自定义的数据加载器或评估指标，Intel® Neural Compressor 支持通过 Python 代码注册自定义数据加载器/指标。

关于yaml配置文件格式，请参考 `yaml模板 <https://github.com/intel/neural-compressor/blob/master/neural_compressor/template/ptq.yaml>`_ 。

*Intel® Neural Compressor* 所需的代码更改在上面的注释中突出显示。

模型
^^^^^

在本教程中，我们使用LeNet模型来演示如何使用 *Intel® Neural Compressor* 。

.. code-block:: python3

    # main.py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # LeNet模型定义
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc1_drop = nn.Dropout()
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.reshape(-1, 320)
            x = F.relu(self.fc1(x))
            x = self.fc1_drop(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net()
    model.load_state_dict(torch.load('./lenet_mnist_model.pth'))

预训练模型权重 `lenet_mnist_model.pth` 来自
`这里 <https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h?usp=sharing>`_ 。

精度驱动量化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intel® Neural Compressor 支持精度驱动的自动调优，以生成满足预定精度目标的最佳 int8 模型。

以下是通过自动调优在PyTorch `FX图模式 <https://pytorch.org/docs/stable/fx.html>`_ 上量化简单网络的示例。

.. code-block:: yaml

    # conf.yaml
    model:
        name: LeNet
        framework: pytorch_fx

    evaluation:
        accuracy:
            metric:
                topk: 1

    tuning:
      accuracy_criterion:
        relative: 0.01

.. code-block:: python3

    # main.py
    model.eval()

    from torchvision import datasets, transforms
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=1)

    # Intel® Neural Compressor启动代码
    from neural_compressor.experimental import Quantization
    quantizer = Quantization("./conf.yaml")
    quantizer.model = model
    quantizer.calib_dataloader = test_loader
    quantizer.eval_dataloader = test_loader
    q_model = quantizer()
    q_model.save('./output')

在 `conf.yaml` 文件中，指定了 Intel® Neural Compressor 的内置指标 `top1` 作为评估方法，
并将 `1%` 的相对精度损失设置为自动调优的精度目标。Intel® Neural Compressor 将遍历每个操作级别上所有可能的量化配置组合，
以找出达到预定精度目标的最佳 int8 模型。

除了这些内置指标外，Intel® Neural Compressor 还支持通过 Python 代码自定义指标：

.. code-block:: yaml

    # conf.yaml
    model:
        name: LeNet
        framework: pytorch_fx

    tuning:
        accuracy_criterion:
            relative: 0.01

.. code-block:: python3

    # main.py
    model.eval()

    from torchvision import datasets, transforms
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=1)

    # 定义自定义指标
    class Top1Metric(object):
        def __init__(self):
            self.correct = 0
        def update(self, output, label):
            pred = output.argmax(dim=1, keepdim=True)
            self.correct += pred.eq(label.view_as(pred)).sum().item()
        def reset(self):
            self.correct = 0
        def result(self):
            return 100. * self.correct / len(test_loader.dataset)

    # Intel® Neural Compressor启动代码
    from neural_compressor.experimental import Quantization
    quantizer = Quantization("./conf.yaml")
    quantizer.model = model
    quantizer.calib_dataloader = test_loader
    quantizer.eval_dataloader = test_loader
    quantizer.metric = Top1Metric()
    q_model = quantizer()
    q_model.save('./output')

在上面的示例中，实现了一个包含 `update()` 和 `result()` 函数的 `class` ，用于记录每个小批量的结果并在最后计算最终精度。

量化感知训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^

除了训练后静态量化和训练后动态量化外，Intel® Neural Compressor 还支持具有精度驱动自动调优机制的量化感知训练。

以下是在PyTorch `FX图模式 <https://pytorch.org/docs/stable/fx.html>`_ 上对简单网络进行量化感知训练的示例。

.. code-block:: yaml

    # conf.yaml
    model:
        name: LeNet
        framework: pytorch_fx

    quantization:
        approach: quant_aware_training

    evaluation:
        accuracy:
            metric:
                topk: 1

    tuning:
        accuracy_criterion:
            relative: 0.01

.. code-block:: python3

    # main.py
    model.eval()

    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1)

    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.1)

    def training_func(model):
        model.train()
        for epoch in range(1, 3):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                print('训练轮次: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}'.format(
                      epoch, batch_idx * len(data), len(train_loader.dataset),
                      100. * batch_idx / len(train_loader), loss.item()))

    # Intel® Neural Compressor 启动代码
    from neural_compressor.experimental import Quantization
    quantizer = Quantization("./conf.yaml")
    quantizer.model = model
    quantizer.q_func = training_func
    quantizer.eval_dataloader = test_loader
    q_model = quantizer()
    q_model.save('./output')

仅性能量化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intel® Neural Compressor 支持使用虚拟数据集直接生成 int8 模型，用于性能基准测试目的。

以下是使用虚拟数据集在PyTorch `FX图模式 <https://pytorch.org/docs/stable/fx.html>`_ 上量化简单网络的示例。

.. code-block:: yaml

    # conf.yaml
    model:
        name: lenet
        framework: pytorch_fx

.. code-block:: python3

    # main.py
    model.eval()

    # Intel® Neural Compressor启动代码
    from neural_compressor.experimental import Quantization, common
    from neural_compressor.experimental.data.datasets.dummy_dataset import DummyDataset
    quantizer = Quantization("./conf.yaml")
    quantizer.model = model
    quantizer.calib_dataloader = common.DataLoader(DummyDataset([(1, 1, 28, 28)]))
    q_model = quantizer()
    q_model.save('./output')

量化输出
~~~~~~~~~~~~~~~~~~~~

用户可以从 Intel® Neural Compressor 打印的日志中了解有多少操作被量化，如下所示：

::

    2021-12-08 14:58:35 [INFO] |********Mixed Precision Statistics*******|
    2021-12-08 14:58:35 [INFO] +------------------------+--------+-------+
    2021-12-08 14:58:35 [INFO] |        Op Type         | Total  |  INT8 |
    2021-12-08 14:58:35 [INFO] +------------------------+--------+-------+
    2021-12-08 14:58:35 [INFO] |  quantize_per_tensor   |   2    |   2   |
    2021-12-08 14:58:35 [INFO] |         Conv2d         |   2    |   2   |
    2021-12-08 14:58:35 [INFO] |       max_pool2d       |   1    |   1   |
    2021-12-08 14:58:35 [INFO] |          relu          |   1    |   1   |
    2021-12-08 14:58:35 [INFO] |       dequantize       |   2    |   2   |
    2021-12-08 14:58:35 [INFO] |       LinearReLU       |   1    |   1   |
    2021-12-08 14:58:35 [INFO] |         Linear         |   1    |   1   |
    2021-12-08 14:58:35 [INFO] +------------------------+--------+-------+

量化模型将在 `./output` 目录下生成，其中包含两个文件：

1. best_configure.yaml
2. best_model_weights.pt

第一个文件包含每个操作的量化配置，第二个文件包含 int8 权重以及激活的零点和比例信息。

部署
~~~~

用户可以使用以下代码加载量化模型，然后进行推理或性能基准测试。

.. code-block:: python3

    from neural_compressor.utils.pytorch import load
    int8_model = load('./output', model)

教程
----

请访问 `Intel® Neural Compressor Github 仓库 <https://github.com/intel/neural-compressor>`_
获取更多教程。
