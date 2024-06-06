跟着示例学习 PyTorch
==============================

**Author**: `Justin Johnson <https://github.com/jcjohnson/pytorch-examples>`_

.. 注意::
   这是我们较早的PyTorch教程之一。您可以在
   `学习基础知识 <https://pytorch.org/tutorials/beginner/basics/intro.html>`_ 中查看我们的最新
   初学者内容。

本教程通过自包含示例介绍了 `PyTorch <https://github.com/pytorch/pytorch>`__ 的基本概念。

在其核心，PyTorch提供了两个主要功能：

- 一个n维张量，类似于numpy，但可以在GPU上运行
- 用于构建和训练神经网络的自动微分

我们将使用拟合 :math:`y=\sin(x)` 的问题作为示例。网络将具有四个参数，并将使用梯度下降法训练，通过最小化网络输出与真实输出之间的欧几里得距离(Euclidean distance)来拟合随机数据。

.. note::
   可在
   :ref:`本文末尾处 <examples-download>` 查看示例。

.. contents:: Table of Contents
   :local:

张量
~~~~~~~

热身：numpy
--------------

在介绍 PyTorch 之前，我们将先使用 numpy 来实现网络。

Numpy提供了一个n维数组对象，并提供了许多用于操作这些数组的函数。Numpy是一个通用的科学计算框架；它不知道任何关于计算图、深度学习或梯度的信息。
然而，我们可以很容易地使用 numpy 提供的方法，手动实现前向和后向传播过程，来拟合一个三次多项式到正弦函数：

.. includenodoc:: /beginner/examples_tensor/polynomial_numpy.py


PyTorch：张量
----------------

Numpy是一个很棒的框架，但它不能利用GPU来加速其数值计算。对于现代深度神经网络，GPU通常提供 `50倍或更大的加速 <https://github.com/jcjohnson/cnn-benchmarks>`__，
因此，numpy对于现代深度学习来说还是不够的。

在这里，我们介绍了PyTorch最基本的概念： **张量**。
一个PyTorch张量在概念上与numpy数组相同：一个n维数组，
PyTorch提供了许多操作这些张量的函数，可以自动跟踪计算图和梯度，它们也作为科学计算的通用工具非常有用。

与 numpy 不同，PyTorch 张量可以利用GPU来加速它们的数值计算。要在GPU上运行 PyTorch 张量，您只需要指定正确的设备。

在这里，我们使用 PyTorch 张量来拟合一个三次多项式到正弦函数中。
与上面的numpy示例类似，我们需要手动实现网络的前向和后向传递：

.. includenodoc:: /beginner/examples_tensor/polynomial_tensor.py


自动求导
~~~~~~~~

PyTorch：张量和自动求导
-------------------------------

在上面的示例中，我们必须手动实现神经网络的前向和后向传递。对于一个小型的两层网络来说，手动实现后向传递并不是什么大问题，但对于大型复杂的网络来说，很快就会变得非常麻烦。

幸运的是，我们可以使用 `自动微分 <https://en.wikipedia.org/wiki/Automatic_differentiation>`__ 来自动计算神经网络中的后向传递。
PyTorch中的 **autograd** 包正是提供了这种功能。当使用自动求导时，网络的前向传递将定义一个 **计算图** ；图中的节点是张量，边是从输入张量生成输出张量的函数。通过这个图进行反向传播，然后可以轻松计算梯度。

这听起来很复杂，但在实际使用中非常简单。每个张量代表计算图中的一个节点。如果 ``x`` 是一个设置了 ``x.requires_grad=True`` 的张量，那么 ``x.grad`` 将是另一个张量，它包含了 ``x`` 相对于某个标量值的梯度。

在这里，我们使用PyTorch张量和自动求导来实现我们用三次多项式拟合正弦波的示例；现在我们不再需要手动实现网络的后向传递：

.. includenodoc:: /beginner/examples_autograd/polynomial_autograd.py


PyTorch：定义新的自动求导函数
----------------------------------------

在底层，每个原始的自动求导操作实际上是对张量进行操作的两个函数。**前向**函数从输入张量计算输出张量。**后向**函数接收输出张量相对于某个标量值的梯度，并计算输入张量相对于同一标量值的梯度。

在PyTorch中，我们可以通过定义一个``torch.autograd.Function``的子类并实现``forward``和``backward``函数，轻松定义自己的自动求导操作符。然后，我们可以通过构造实例并像函数一样调用它，传递包含输入数据的张量，来使用我们新的自动求导操作符。

在这个示例中，我们将模型定义为:math:`y=a+b P_3(c+dx)`而不是:math:`y=a+bx+cx^2+dx^3`，其中:math:`P_3(x)=\frac{1}{2}\left(5x^3-3x\right)`是三阶的`勒让德多项式`_。我们编写了自己的自定义自动求导函数来计算:math:`P_3`的前向和后向传递，并使用它来实现我们的模型：

.. _Legendre polynomial:
    https://en.wikipedia.org/wiki/Legendre_polynomials

.. includenodoc:: /beginner/examples_autograd/polynomial_custom_function.py

``nn`` module
~~~~~~~~~~~~~

PyTorch: ``nn``
---------------

计算图(Computational graphs) 和自动求导(autograd) 是定义复杂操作非常强大的功能；然而，原始的自动求导还是不足以支持实现大型神经网络。

在构建神经网络时，我们通常会考虑将计算安排成 **层(layers)**，其中一些层具有 **可学习的参数(learnable parameters)**，这些参数将在学习过程中进行优化。

在TensorFlow中，像 `Keras <https://github.com/fchollet/keras>`__、 `TensorFlow-Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`__
和 `TFLearn <http://tflearn.org/>`__ 提供了相较于原始计算图的更高层次的抽象，这些抽象对于构建神经网络非常有用。

在PyTorch中， ``nn``包起到了同样的作用。 ``nn``包定义了一组 **模块**，这些模块相当于神经网络层。一个模块接收输入张量并计算输出张量，
但也可以包含内部状态，例如包含可学习参数的张量。``nn``包还定义了一组常用于训练神经网络的有用的损失函数(loss functions)。

在这个示例中，我们使用 ``nn`` 包来实现我们的多项式模型网络：

.. includenodoc:: /beginner/examples_nn/polynomial_nn.py

PyTorch: optim
--------------

我们通过使用 ``torch.no_grad()`` 手动更改张量的可学习参数，来更新模型的权重。
对于像随机梯度下降这样的优化算法来说，这并不是一个很大的负担，但在实践中，我们经常使用更复杂的优化器来训练神经网络，
比如 ``AdaGrad``、 ``RMSProp``、 ``Adam`` 等。

PyTorch中的 ``optim`` 包抽象了优化算法的定义，并提供了常用优化算法的实现。

在这个示例中，我们将像以前一样使用 ``nn``包来定义我们的模型，但我们将使用 ``optim`` 包提供的 ``RMSprop`` 算法来优化模型：

.. includenodoc:: /beginner/examples_nn/polynomial_optim.py

PyTorch: 自定义 ``nn`` 模块
---------------------------

有时你可能会希望自定义比现有模块集更复杂的模型；在这些情况下，你可以通过继承 ``nn.Module`` 并定义一个 ``forward`` 方法来自定义模块，
该方法接收输入张量，并使用其他模块或在张量上自动求导等操作生成新的输出张量。

在这个示例中，我们将实现一个三次多项式作为自定义模块的子类：

.. includenodoc:: /beginner/examples_nn/polynomial_module.py

PyTorch: 控制流 + 权重共享
---------------------------

作为动态计算图和权重共享的一个示例，我们实现了一个非常奇特的模型：一个三至五阶的多项式，在每次前向传递时随机选择一个3到5之间的数字，
并使用该阶数多项式来计算，重复使用相同的权重多次以计算四阶和五阶多项式。

对于这个模型，我们可以使用Python流控制来实现循环，并且可以通过在定义前向传递时，多次重复使用相同的参数，来实现权重共享。

我们可以很容易地将这个模型实现为一个模块的子类：

.. includenodoc:: /beginner/examples_nn/dynamic_net.py


.. _examples-download:

示例
~~~~~~~~

具体示例如下

Tensors
-------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_tensor/polynomial_numpy
   /beginner/examples_tensor/polynomial_tensor

.. galleryitem:: /beginner/examples_tensor/polynomial_numpy.py

.. galleryitem:: /beginner/examples_tensor/polynomial_tensor.py

.. raw:: html

    <div style='clear:both'></div>

Autograd
--------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_autograd/polynomial_autograd
   /beginner/examples_autograd/polynomial_custom_function


.. galleryitem:: /beginner/examples_autograd/polynomial_autograd.py

.. galleryitem:: /beginner/examples_autograd/polynomial_custom_function.py

.. raw:: html

    <div style='clear:both'></div>

``nn`` module
--------------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_nn/polynomial_nn
   /beginner/examples_nn/polynomial_optim
   /beginner/examples_nn/polynomial_module
   /beginner/examples_nn/dynamic_net


.. galleryitem:: /beginner/examples_nn/polynomial_nn.py

.. galleryitem:: /beginner/examples_nn/polynomial_optim.py

.. galleryitem:: /beginner/examples_nn/polynomial_module.py

.. galleryitem:: /beginner/examples_nn/dynamic_net.py

.. raw:: html

    <div style='clear:both'></div>
