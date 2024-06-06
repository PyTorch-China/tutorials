"""
`基础知识 <intro.html>`_ ||
`快速入门 <quickstart_tutorial.html>`_ ||
`张量 <tensorqs_tutorial.html>`_ ||
`数据集与数据加载器 <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`构建神经网络 <buildmodel_tutorial.html>`_ ||
`自动微分 <autogradqs_tutorial.html>`_ ||
`优化模型参数 <optimization_tutorial.html>`_ ||
**保存和加载模型**

保存和加载模型
============================

在本节中，我们将学习如何通过保存、加载以及运行模型预测，来持久化模型。
"""

import torch
import torchvision.models as models


#######################################################################
# 保存和加载模型权重
# --------------------------------
# PyTorch模型将学习到的参数存储在一个内部状态字典中，称为``state_dict``。这些参数可以通过``torch.save``进行持久化。
# 方法:

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

##########################
# 要加载模型权重，您需要先创建一个相同模型的实例，然后使用``load_state_dict()``方法加载参数。

model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

###########################
# 。。 注意:: 在进行推理之前，请确保调用``model.eval()``方法以将 dropout 和 batch normalization layers设置为评估模式。如果不这样做，将导致不一致的推理结果。

#######################################################################
# 保存和加载带有结构的模型
# -------------------------------------
# 在加载模型权重时，我们需要先实例化模型类，因为类定义了网络的结构。我们可能希望将这个类的结构与模型一起保存，
# 在这种情况下，我们可以将``model``（而不是``model.state_dict()``）传递给 save 函数：

torch.save(model, 'model.pth')

########################
# 我们可以使用如下方式加载模型: 

model = torch.load('model.pth')

########################
# .. 注意:: 这种方法在序列化模型时使用 Python 的 `pickle <https://docs.python.org/3/library/pickle.html>`_模块，因此在加载模型时需要依赖实际的类定义。

#######################
# 相关教程
# -----------------
# - `PyTorch 中保存和加载通用Checkpoint <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>`_
# - `从 checkpoint 加载 nn.Module 的实用技巧 <https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html?highlight=loading%20nn%20module%20from%20checkpoint>`_
