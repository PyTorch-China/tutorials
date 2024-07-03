"""
如何在PyTorch中使用TensorBoard
===================================
TensorBoard是一个用于机器学习实验的可视化工具包。
TensorBoard允许跟踪和可视化指标,如损失和准确率,
可视化模型图,查看直方图,显示图像等。
在本教程中,我们将介绍TensorBoard的安装、
在PyTorch中的基本用法,以及如何在TensorBoard UI中可视化您记录的数据。

安装
----------------------
应安装PyTorch以将模型和指标记录到TensorBoard日志
目录。以下命令将通过Anaconda(推荐)安装PyTorch 1.4+:

.. code-block:: sh

   $ conda install pytorch torchvision -c pytorch
   

或者使用pip:

.. code-block:: sh

   $ pip install torch torchvision

"""

######################################################################
# 在PyTorch中使用TensorBoard
# -----------------------------
#
# 现在让我们尝试在PyTorch中使用TensorBoard!在记录任何内容之前,
# 我们需要创建一个 ``SummaryWriter`` 实例。
#

import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

######################################################################
# 写入器默认将输出到 ``./runs/`` 目录。
#


######################################################################
# 记录标量
# -----------
#
# 在机器学习中,了解关键指标(如损失)及其在训练期间的变化非常重要。
# 标量可用于保存每个训练步骤的损失值或每个epoch的准确率。
#
# 要记录标量值,请使用
# ``add_scalar(tag, scalar_value, global_step=None, walltime=None)``。
# 例如,让我们创建一个简单的线性回归训练,并
# 使用 ``add_scalar`` 记录损失值
#

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


train_model(10)
writer.flush()


######################################################################
# 调用 ``flush()`` 方法以确保所有待处理事件
# 已写入磁盘。
#
# 请参阅 `torch.utils.tensorboard 教程 <https://pytorch.org/docs/stable/tensorboard.html>`_
# 以了解您可以记录的更多TensorBoard可视化类型。
#
# 如果您不再需要摘要写入器,请调用 ``close()`` 方法。
#

writer.close()

######################################################################
# 运行TensorBoard
# ----------------
#
# 通过命令行安装TensorBoard以可视化您记录的数据
#
# .. code-block:: sh
#
#    pip install tensorboard
#
#
# 现在,启动TensorBoard,指定您之前使用的根日志目录。
# 参数 ``logdir`` 指向TensorBoard将查找可显示的事件文件的目录。
# TensorBoard将递归遍历 ``logdir`` 根目录下的目录结构,寻找 ``.*tfevents.*`` 文件。
#
# .. code-block:: sh
#
#    tensorboard --logdir=runs
#
# 转到它提供的URL或 `http://localhost:6006/ <http://localhost:6006/>`_
#
# .. image:: ../../_static/img/thumbnails/tensorboard_scalars.png
#    :scale: 40 %
#
# 此仪表板显示了损失和准确率如何随着每个epoch而变化。
# 您可以使用它来跟踪训练速度、学习率和其他标量值。
# 比较不同训练运行的这些指标有助于改进您的模型。
#


########################################################################
# 了解更多
# ----------------------------
#
# -  `torch.utils.tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ 文档
# -  `使用TensorBoard可视化模型、数据和训练 <https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html>`_ 教程
#
