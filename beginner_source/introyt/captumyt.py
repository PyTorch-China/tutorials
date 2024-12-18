"""
`简介 <introyt1_tutorial.html>`_ ||
`张量 <tensors_deeper_tutorial.html>`_ ||
`自动微分 <autogradyt_tutorial.html>`_ ||
`构建模型 <modelsyt_tutorial.html>`_ ||
`TensorBoard支持 <tensorboardyt_tutorial.html>`_ ||
`训练模型 <trainingyt.html>`_ ||
**模型理解**

使用 Captum 进行模型理解
===============================

跟随下面的视频或在 `youtube <https://www.youtube.com/watch?v=Am2EF9CLu-g>`__ 上观看。从 `这里 <https://pytorch-tutorial-assets.s3.amazonaws.com/youtube-series/video7.zip>`__ 下载笔记本和相应文件。

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/Am2EF9CLu-g" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

`Captum <https://captum.ai/>`__ (拉丁语中的"理解")是一个开源的、可扩展的模型可解释性库,建立在PyTorch之上。

随着模型复杂性的增加和由此带来的透明度的缺乏,模型可解释性方法变得越来越重要。模型理解是一个活跃的研究领域,也是跨行业使用机器学习的实际应用的一个关注领域。Captum提供了最先进的算法,包括集成梯度,为研究人员和开发人员提供了一种简单的方式来理解哪些特征对模型的输出做出了贡献。

完整的文档、API参考和一套关于特定主题的教程可在 `captum.ai <https://captum.ai/>`__ 网站上找到。

介绍
------------

Captum对模型可解释性的方法是基于*归因*的。Captum中有三种类型的归因:

-  **特征归因**试图解释特定输出是由生成它的输入的哪些特征产生的。用某些词来解释一篇电影评论是正面还是负面的,就是特征归因的一个例子。
-  **层归因**检查模型的隐藏层在特定输入下的活动。检查卷积层对输入图像的空间映射输出就是层归因的一个例子。
-  **神经元归因**类似于层归因,但关注单个神经元的活动。

在这个交互式笔记本中,我们将看看特征归因和层归因。

每种归因类型都有多种**归因算法**与之相关联。许多归因算法可分为两大类:

-  **基于梯度的算法**计算模型输出、层输出或神经元激活相对于输入的反向梯度。**集成梯度**(用于特征)、**层梯度 \* 激活**和**神经元传导**都是基于梯度的算法。
-  **基于扰动的算法**检查模型、层或神经元的输出在输入发生变化时的变化情况。输入扰动可能是有针对性的或随机的。**遮挡**、**特征消融**和**特征置换**都是基于扰动的算法。

我们将在下面检查这两种类型的算法。

特别是对于大型模型,以与被检查的输入特征直接相关的方式可视化归因数据是很有价值的。虽然当然可以使用Matplotlib、Plotly或类似工具创建自己的可视化,但Captum提供了专门用于其归因的增强工具:

-  ``captum.attr.visualization``模块(下面导入为``viz``)提供了有助于可视化与图像相关的归因的函数。
-  **Captum Insights**是建立在Captum之上的一个易于使用的可解释性可视化小部件,提供了一个带有现成可视化工具的小部件,用于图像、文本和任意模型类型。

这两种可视化工具集都将在本笔记本中进行演示。前几个示例将集中在计算机视觉用例上,但最后的Captum Insights部分将演示视觉问答模型中的归因可视化。

安装
------------

在开始之前,你需要有一个Python环境,包括:

-  Python 3.6或更高版本
-  对于Captum Insights示例,需要Flask 1.1或更高版本和Flask-Compress(推荐使用最新版本)
-  PyTorch 1.2或更高版本(推荐使用最新版本)
-  TorchVision 0.6或更高版本(推荐使用最新版本)
-  Captum(推荐使用最新版本)
-  Matplotlib 3.3.4版本,因为Captum目前使用了一个在更高版本中参数已被重命名的Matplotlib函数

要在Anaconda或pip虚拟环境中安装Captum,请使用下面适用于您环境的命令:

使用``conda``:

.. code-block:: sh

    conda install pytorch torchvision captum flask-compress matplotlib=3.3.4 -c pytorch

使用``pip``:

.. code-block:: sh

    pip install torch torchvision captum matplotlib==3.3.4 Flask-Compress

在您设置的环境中重新启动此笔记本,您就可以开始了!


第一个示例
---------------
 
首先,让我们看一个简单的视觉示例。我们将从一个在ImageNet数据集上预训练的ResNet模型开始。我们将获取一个测试输入,并使用不同的**特征归因**算法来检查输入图像如何影响输出,并查看一些测试图像的输入归因映射的有用可视化。
 
首先,导入一些包: 

"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


#########################################################################
# 现在我们将使用TorchVision模型库下载一个预训练的ResNet。由于我们不进行训练,我们将暂时将其置于评估模式。
# 

model = models.resnet18(weights='IMAGENET1K_V1')
model = model.eval()


#######################################################################
# 你从中获取这个交互式笔记本的地方应该也有一个``img``文件夹,其中包含一个``cat.jpg``文件。
# 

test_img = Image.open('img/cat.jpg')
test_img_data = np.asarray(test_img)
plt.imshow(test_img_data)
plt.show()


##########################################################################
# 我们的ResNet模型是在ImageNet数据集上训练的,它期望图像具有一定的大小,并且通道数据被归一化到特定的值范围。我们还将获取模型识别的类别的人类可读标签列表 - 它应该也在``img``文件夹中。
# 

# 模型期望224x224 3色彩图像
transform = transforms.Compose([
 transforms.Resize(224),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])

# 标准ImageNet归一化
transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

transformed_img = transform(test_img)
input_img = transform_normalize(transformed_img)
input_img = input_img.unsqueeze(0) # 模型需要一个虚拟的批次维度

labels_path = 'img/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)


######################################################################
# 现在,我们可以问:这个模型认为这张图像代表什么?
# 

output = model(input_img)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx.squeeze_()
predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
print('预测:', predicted_label, '(', prediction_score.squeeze().item(), ')')


######################################################################
# 我们已经确认ResNet认为我们的猫的图像确实是一只猫。但是*为什么*模型认为这是一张猫的图像呢?
# 
# 要回答这个问题,我们就要求助于Captum。
# 


##########################################################################
# 使用集成梯度进行特征归因
# ---------------------------------------------
# 
# **特征归因**试图用生成特定输出的输入的特征来解释该输出。它使用特定的输入 - 在这里是我们的测试图像 - 来生成一个输入特征对特定输出特征的相对重要性的映射。
# 
# `集成梯度 <https://captum.ai/api/integrated_gradients.html>`__ 是Captum中可用的特征归因算法之一。集成梯度通过近似模型输出相对于输入的梯度的积分,为每个输入特征分配一个重要性分数。
# 
# 在我们的例子中,我们将使用输出向量的一个特定元素 - 也就是表示模型对所选类别的置信度的那个元素 - 并使用集成梯度来理解哪些输入图像部分对这个输出做出了贡献。
# 
# 一旦我们从集成梯度获得了重要性映射,我们将使用Captum中的可视化工具来提供与被检查的输入特征直接相关的重要性映射的有用表示。Captum的``visualize_image_attr()``函数提供了各种自定义显示归因数据的选项。在这里,我们传入一个自定义的Matplotlib颜色映射。
# 
# 运行带有``integrated_gradients.attribute()``调用的单元格通常需要一两分钟。
# 

# 用模型初始化归因算法
integrated_gradients = IntegratedGradients(model)

# 要求算法将我们的输出目标归因于
attributions_ig = integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)

# 显示原始图像以供比较
_ = viz.visualize_image_attr(None, np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)), 
                      method="original_image", title="Original Image")

default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#0000ff'),
                                                  (1, '#0000ff')], N=256)

_ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             title='集成梯度')


#######################################################################
# 在上面的图像中,你应该可以看到集成梯度在图像中猫的位置给出了最强的信号。
# 


##########################################################################
# 使用遮挡进行特征归因
# ----------------------------------
# 
# 基于梯度的归因方法有助于通过直接计算输出相对于输入的变化来理解模型。*基于扰动的归因*方法则更直接地解决这个问题,通过对输入进行变化来测量对输出的影响。
# `遮挡 <https://captum.ai/api/occlusion.html>`__ 就是这样一种方法。它涉及替换输入图像的部分区域,并检查对输出信号的影响。
# 
# 下面,我们设置遮挡归因。与配置卷积神经网络类似,你可以指定目标区域的大小,以及确定单个测量间距的步长长度。我们将使用``visualize_image_attr_multiple()``来可视化我们的遮挡归因输出,显示每个区域的正面和负面归因的热图,并用正面归因区域掩码原始图像。掩码可以给出一个非常有启发性的视图,显示模型发现哪些区域最"像猫"。
# 
occlusion = Occlusion(model)

attributions_occ = occlusion.attribute(input_img,
                                       target=pred_label_idx,
                                       strides=(3, 8, 8),
                                       sliding_window_shapes=(3,15, 15),
                                       baselines=0)


_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map", "heat_map", "masked_image"],
                                      ["all", "positive", "negative", "positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                      fig_size=(18, 6)
                                     )

######################################################################
# 同样,我们看到图像中包含猫的区域被赋予了更大的重要性。
#


#########################################################################
# 使用层梯度类激活映射(Layer GradCAM)进行层归因
# ------------------------------------
#
# **层归因**允许你将模型中隐藏层的活动归因于输入的特征。下面,我们将使用
# 层归因算法来检查模型中一个卷积层的活动。
#
# GradCAM计算目标输出相对于给定层的梯度,对每个输出通道(输出的第2维)进行平均,
# 并将每个通道的平均梯度乘以层激活。结果在所有通道上求和。GradCAM专为卷积网络
# 设计;由于卷积层的活动通常在空间上映射到输入,因此GradCAM归因通常会被上采样
# 并用于掩盖输入。
#
# 层归因的设置类似于输入归因,除了除了模型之外,你还必须指定模型中你希望检查的
# 隐藏层。与上面一样,当我们调用`attribute()`时,我们指定感兴趣的目标类。
#

layer_gradcam = LayerGradCam(model, model.layer3[1].conv2)
attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)

_ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                             sign="all",
                             title="Layer 3 Block 1 Conv 2")


##########################################################################
# 我们将使用`LayerAttribution <https://captum.ai/api/base_classes.html?highlight=layerattribution#captum.attr.LayerAttribution>`__
# 基类中的便利方法`interpolate()`来上采样这些归因数据,以便与输入图像进行比较。
#

upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])

print(attributions_lgc.shape)
print(upsamp_attr_lgc.shape)
print(input_img.shape)

_ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                      transformed_img.permute(1,2,0).numpy(),
                                      ["original_image","blended_heat_map","masked_image"],
                                      ["all","positive","positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Masked"],
                                      fig_size=(18, 6))


#######################################################################
# 这样的可视化可以让你深入了解隐藏层如何响应你的输入。
#


##########################################################################
# 使用Captum Insights进行可视化
# ----------------------------------
#
# Captum Insights是一个建立在Captum之上的可解释性可视化小部件,旨在促进模型理解。
# Captum Insights可用于图像、文本和其他特征,帮助用户理解特征归因。它允许你可视化
# 多个输入/输出对的归因,并提供用于图像、文本和任意数据的可视化工具。
#
# 在本笔记本的这一部分,我们将使用Captum Insights可视化多个图像分类推理。
#
# 首先,让我们收集一些图像,看看模型对它们的看法。为了增加多样性,我们将使用猫、
# 茶壶和三叶虫化石:
#

imgs = ['img/cat.jpg', 'img/teapot.jpg', 'img/trilobite.jpg']

for img in imgs:
    img = Image.open(img)
    transformed_img = transform(img)
    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0) # 模型需要一个虚拟的批次维度

    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted:', predicted_label, '/', pred_label_idx.item(), ' (', prediction_score.squeeze().item(), ')')


##########################################################################
# ...看起来我们的模型都正确识别了它们 - 但是,我们当然希望深入挖掘。为此,我们将
# 使用Captum Insights小部件,我们用下面导入的`AttributionVisualizer`对象对其进行配置。
# `AttributionVisualizer`期望批量数据,所以我们将引入Captum的`Batch`辅助类。
# 我们将查看图像,因此我们还将导入`ImageFeature`。
#
# 我们使用以下参数配置`AttributionVisualizer`:
#
# - 要检查的模型数组(在我们的例子中,只有一个)
# - 一个评分函数,允许Captum Insights从模型中提取前k个预测
# - 我们模型训练的类别的有序、人类可读列表
# - 要查找的特征列表 - 在我们的例子中,是一个`ImageFeature`
# - 一个数据集,它是一个可迭代对象,返回输入和标签的批次 - 就像你用于训练一样
#

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

# 基线是全零输入 - 这可能会因你的数据而有所不同
def baseline_func(input):
    return input * 0

# 合并上面的图像变换
def full_img_transform(input):
    i = Image.open(input)
    i = transform(i)
    i = transform_normalize(i)
    i = i.unsqueeze(0)
    return i


input_imgs = torch.cat(list(map(lambda i: full_img_transform(i), imgs)), 0)

visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: torch.nn.functional.softmax(o, 1),
    classes=list(map(lambda k: idx_to_labels[k][1], idx_to_labels.keys())),
    features=[
        ImageFeature(
            "照片",
            baseline_transforms=[baseline_func],
            input_transforms=[],
        )
    ],
    dataset=[Batch(input_imgs, labels=[282,849,69])]
)


#########################################################################
# 注意,运行上面的单元格并没有花费太多时间,不像我们之前的归因那样。这是因为
# Captum Insights允许你在可视化小部件中配置不同的归因算法,之后它将计算并显示
# 归因。*那个*过程将需要几分钟时间。
#
# 运行下面的单元格将渲染Captum Insights小部件。然后你可以选择归因方法及其参数、
# 根据预测的类或预测的正确性过滤模型响应、查看模型的预测及相关概率、查看归因与
# 原始图像的热力图。
#

visualizer.render()
