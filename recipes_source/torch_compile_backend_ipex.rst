Intel® PyTorch* 扩展后端
=====================================

为了更好地与 `torch.compile` 协作，Intel® PyTorch* 扩展实现了一个名为 `ipex` 的后端。
它旨在提高 Intel 平台上的硬件资源使用效率,从而获得更好的性能。
`ipex` 后端是通过 Intel® PyTorch* 扩展中进一步的定制设计来实现模型编译的。

使用示例
~~~~~~~~~~~~~

FP32 训练
----------

查看下面的示例,了解如何将 `ipex` 后端与 `torch.compile` 一起使用,进行 FP32 数据类型的模型训练。

.. code:: python

   import torch
   import torchvision

   LR = 0.001
   DOWNLOAD = True
   DATA = 'datasets/cifar10/'

   transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   train_dataset = torchvision.datasets.CIFAR10(
     root=DATA,
     train=True,
     transform=transform,
     download=DOWNLOAD,
   )
   train_loader = torch.utils.data.DataLoader(
     dataset=train_dataset,
     batch_size=128
   )

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
   model.train()

   #################### 代码修改 ####################
   import intel_extension_for_pytorch as ipex

   # 可选择调用以下 API,应用前端优化
   model, optimizer = ipex.optimize(model, optimizer=optimizer)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   for batch_idx, (data, target) in enumerate(train_loader):
       optimizer.zero_grad()
       output = compile_model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()


BF16 训练
----------

查看下面的示例,了解如何将 `ipex` 后端与 `torch.compile` 一起使用,进行 BFloat16 数据类型的模型训练。

.. code:: python

   import torch
   import torchvision

   LR = 0.001
   DOWNLOAD = True
   DATA = 'datasets/cifar10/'

   transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   train_dataset = torchvision.datasets.CIFAR10(
     root=DATA,
     train=True,
     transform=transform,
     download=DOWNLOAD,
   )
   train_loader = torch.utils.data.DataLoader(
     dataset=train_dataset,
     batch_size=128
   )

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
   model.train()

   #################### 代码修改 ####################
   import intel_extension_for_pytorch as ipex

   # 可选择调用以下 API,应用前端优化
   model, optimizer = ipex.optimize(model, dtype=torch.bfloat16, optimizer=optimizer)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   with torch.cpu.amp.autocast():
       for batch_idx, (data, target) in enumerate(train_loader):
           optimizer.zero_grad()
           output = compile_model(data)
           loss = criterion(output, target)
           loss.backward()
           optimizer.step()


FP32 推理
--------------

查看下面的示例,了解如何将 `ipex` 后端与 `torch.compile` 一起使用,进行 FP32 数据类型的模型推理。

.. code:: python

   import torch
   import torchvision.models as models

   model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   #################### 代码修改 ####################
   import intel_extension_for_pytorch as ipex

   # 可选择调用以下 API,应用前端优化
   model = ipex.optimize(model, weights_prepack=False)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   with torch.no_grad():
       compile_model(data)


BF16 推理
--------------

查看下面的示例,了解如何将 `ipex` 后端与 `torch.compile` 一起使用,进行 BFloat16 数据类型的模型推理。

.. code:: python

   import torch
   import torchvision.models as models

   model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   #################### 代码修改 ####################
   import intel_extension_for_pytorch as ipex

   # 可选择调用以下 API,应用前端优化
   model = ipex.optimize(model, dtype=torch.bfloat16, weights_prepack=False)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
       compile_model(data)
