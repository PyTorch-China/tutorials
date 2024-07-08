使用 Flask 进行部署
====================

在这个教程中,您将学习:

- 如何将训练好的 PyTorch 模型封装到 Flask 容器中,通过 Web API 暴露出去
- 如何将传入的 Web 请求转换为 PyTorch 张量,以供您的模型使用
- 如何为 HTTP 响应打包您模型的输出

环境设置
------------

您需要一个安装了以下软件包(及其依赖项)的 Python 3 环境:


-  PyTorch 1.5
-  TorchVision 0.6.0
-  Flask 1.1

另外,如果需要获取一些支持文件,您还需要 git。

安装 PyTorch 和 TorchVision 的说明在 `pytorch.org_` 上有介绍。安装 Flask 请查看 `Flask 官网_` 。


什么是 Flask?
--------------

Flask 是一个用 Python 编写的轻量级 Web 服务器。它为您提供了一种便捷的方式,快速建立一个 Web API,
用于您训练好的 PyTorch 模型的预测,可直接使用,或作为更大系统中的 Web 服务。

设置和支持文件
--------------------------

我们将创建一个 Web 服务,接收图像,并将其映射到 ImageNet 数据集的 1000 个类别之一。
为此,您需要一个用于测试的图像文件。另外,您还可以获取一个文件,将模型输出的类索引映射为可读的类名。

选项 1: 快速获取文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以通过检出 TorchServe 仓库并将文件复制到您的工作文件夹来快速获取这两个支持文件。
*(注意:本教程不依赖于 TorchServe - 这只是快速获取文件的一种方式。)* 
从您的 shell 提示符下发出以下命令:

::

   git clone https://github.com/pytorch/serve
   cp serve/examples/image_classifier/kitten.jpg .
   cp serve/examples/image_classifier/index_to_name.json .

And you've got them!

选项 2: 使用您自己的图像
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``index_to_name.json`` 文件在下面的 Flask 服务中是可选的。
您可以使用自己的图像测试您的服务 - 需确保是一个 3 色 JPEG 图像。


构建您的 Flask 服务
---------------------------

Flask 服务的完整 Python 脚本在本教程的最后展示;您可以复制并粘贴到您自己的 ``app.py`` 文件中。
下面我们将查看各个部分,以明确它们的功能。

导入
~~~~~~~

::

   import torchvision.models as models
   import torchvision.transforms as transforms
   from PIL import Image
   from flask import Flask, jsonify, request

按顺序:

- 将使用来自 ``torchvision.models`` 的预训练 DenseNet 模型
- ``torchvision.transforms`` 包含用于操作图像数据的工具
- Pillow (``PIL``) 是我们最初加载图像文件时将使用的库
- 当然我们还需要从 ``flask`` 导入一些类

预处理
~~~~~~~~~~~~~~

::

   def transform_image(infile):
       input_transforms = [transforms.Resize(255),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406],
               [0.229, 0.224, 0.225])]
       my_transforms = transforms.Compose(input_transforms)
       image = Image.open(infile)
       timg = my_transforms(image)
       timg.unsqueeze_(0)
       return timg

Web 请求给了我们一个图像文件,但我们的模型期望一个形状为 (N, 3, 224, 224) 的 PyTorch 张量,
其中 *N* 是输入批次的数量。(我们将只使用批量大小为 1。)我们首先要做的是组合一组 TorchVision 转换,
调整图像大小和裁剪图像,将其转换为张量,然后对张量中的值进行归一化。
(有关此归一化的更多信息,请参阅 ``torchvision.models_`` 的文档。)

之后,我们打开文件并应用转换。转换返回一个形状为 (3, 224, 224) 的张量 - 224x224 图像的 3 个颜色通道。
因为我们需要将这个单个图像变成一个批次,所以我们使用 ``unsqueeze_(0)`` 调用通过添加一个新的第一维来就地修改张量。
张量包含相同的数据,但现在形状为 (1, 3, 224, 224)。

一般来说,即使您不是在处理图像数据,您也需要将来自 HTTP 请求的输入转换为 PyTorch 可以使用的张量。

推理
~~~~~~~~~

::

   def get_prediction(input_tensor):
       outputs = model.forward(input_tensor)
       _, y_hat = outputs.max(1)
       prediction = y_hat.item()
       return prediction

推理本身是最简单的部分:当我们将输入张量传递给模型时,我们会得到一个张量值,代表模型估计图像属于特定类别的可能性。
``max()`` 调用找到具有最大可能性值的类别,并返回该值及其 ImageNet 类索引。
最后,我们使用 ``item()`` 调用从包含它的张量中提取该类索引,并返回它。


后处理
~~~~~~~~~~~~~~~

::

   def render_prediction(prediction_idx):
       stridx = str(prediction_idx)
       class_name = 'Unknown'
       if img_class_map is not None:
           if stridx in img_class_map is not None:
               class_name = img_class_map[stridx][1]

       return prediction_idx, class_name

The ``render_prediction()`` method maps the predicted class index to a
human-readable class label. It's typical, after getting the prediction
from your model, to perform post-processing to make the prediction ready
for either human consumption, or for another piece of software.

``render_prediction()`` 方法将预测的类索引映射为人类可读的类标签。在从您的模型获得预测之后,通常需要进行后处理,
使预测可供人类使用或供另一个软件使用。


运行完整的 Flask 应用
--------------------------

将以下内容粘贴到名为 ``app.py`` 的文件中:

::

   import io
   import json
   import os

   import torchvision.models as models
   import torchvision.transforms as transforms
   from PIL import Image
   from flask import Flask, jsonify, request


   app = Flask(__name__)
   model = models.densenet121(pretrained=True)               # 在 ImageNet 的 1000 个类别上训练
   model.eval()                                              # 关闭自动梯度计算 



   img_class_map = None
   mapping_file_path = 'index_to_name.json'                  # ImageNet 类别的可读名称
   if os.path.isfile(mapping_file_path):
       with open (mapping_file_path) as f:
           img_class_map = json.load(f)



   # 将输入转换为模型期望的形式
   def transform_image(infile):
       input_transforms = [transforms.Resize(255),           # 我们使用多个 TorchVision 转换来准备图像
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406],       # ImageNet 模型输入的标准归一化
               [0.229, 0.224, 0.225])]
       my_transforms = transforms.Compose(input_transforms)
       image = Image.open(infile)                            # 打开图像文件
       timg = my_transforms(image)                           # 将 PIL 图像转换为合适形状的 PyTorch 张量
       timg.unsqueeze_(0)                                    # PyTorch 模型期望批量输入;创建批量大小为 1
       return timg


   # 获取预测
   def get_prediction(input_tensor):
       outputs = model.forward(input_tensor)                 # 获取所有 ImageNet 类别的可能性
       _, y_hat = outputs.max(1)                             # 提取最可能的类别
       prediction = y_hat.item()                             # 从 PyTorch 张量中提取 int 值
       return prediction

   # 使预测结果可读
   def render_prediction(prediction_idx):
       stridx = str(prediction_idx)
       class_name = 'Unknown'
       if img_class_map is not None:
           if stridx in img_class_map is not None:
               class_name = img_class_map[stridx][1]

       return prediction_idx, class_name


   @app.route('/', methods=['GET'])
   def root():
       return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


   @app.route('/predict', methods=['POST'])
   def predict():
       if request.method == 'POST':
           file = request.files['file']
           if file is not None:
               input_tensor = transform_image(file)
               prediction_idx = get_prediction(input_tensor)
               class_id, class_name = render_prediction(prediction_idx)
               return jsonify({'class_id': class_id, 'class_name': class_name})


   if __name__ == '__main__':
       app.run()

从 shell 提示符启动服务器,请执行以下命令:

::

   FLASK_APP=app.py flask run

默认情况下,您的 Flask 服务器监听 5000 端口。服务器运行后,打开另一个终端窗口,测试您新的推理服务器:

::

   curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/predict -F "file=@kitten.jpg"

如果一切设置正确,您应该会收到类似如下的响应:

::

   {"class_id":285,"class_name":"Egyptian_cat"}

重要资源
-------------------

- `pytorch.org`_ 提供安装说明,以及更多文档和教程
- `Flask 官网`_ 有一个 `快速入门指南`_ ,对设置一个简单的 Flask 服务有更详细的介绍

.. _pytorch.org: https://pytorch.org
.. _Flask 官网: https://flask.palletsprojects.com/en/1.1.x/
.. _Quick Start guide: https://flask.palletsprojects.com/en/1.1.x/quickstart/
.. _torchvision.models: https://pytorch.org/vision/stable/models.html
.. _Flask 官网: https://flask.palletsprojects.com/en/1.1.x/installation/
