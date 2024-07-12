将 PyTorch Stable Diffusion 模型部署为 Vertex AI 端点
==================================================

部署大型模型(如 Stable Diffusion)可能具有挑战性且耗时。

在本教程中,我们将展示如何通过利用 Vertex AI 来简化 PyTorch Stable Diffusion 模型的部署过程。

PyTorch 是 Stability AI 在 Stable Diffusion v1.5 上使用的框架。Vertex AI 是一个全托管的机器学习平台,
提供工具和基础设施,旨在帮助 ML 从业者加速和扩展生产中的 ML,同时受益于 PyTorch 等开源框架。

您可以通过四个步骤部署 PyTorch Stable Diffusion 模型(v1.5)。

在 Vertex AI 端点上部署 Stable Diffusion 模型可以通过以下四个步骤完成:

* 创建自定义 TorchServe 处理程序。

* 将模型工件上传到 Google Cloud Storage (GCS)。

* 使用模型工件和预构建的 PyTorch 容器镜像创建 Vertex AI 模型。

* 将 Vertex AI 模型部署到端点。

让我们详细看看每个步骤。您可以使用 `Notebook 示例 <https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/vertex_endpoints/torchserve/dreambooth_stablediffusion.ipynb>`__ 来跟随并实施这些步骤。

注意:请记住,此教程需要一个计费的 Vertex AI,如 notebook 示例中更详细地解释的那样。

创建自定义 TorchServe 处理程序
---------------------------

TorchServe 是一个简单灵活的 PyTorch 模型服务工具。部署到 Vertex AI 的模型使用 TorchServe 来处理请求并从模型返回响应。
您必须创建一个自定义 TorchServe 处理程序,以包含在上传到 Vertex AI 的模型工件中。将处理程序文件包含在与其他模型工件相同的目录中,如下所示: `model_artifacts/handler.py`。

创建处理程序文件后,您必须将处理程序打包为模型归档(MAR)文件。
输出文件必须命名为 `model.mar`。


.. code:: shell

    !torch-model-archiver \
    -f \
    --model-name <your_model_name> \
    --version 1.0 \
     --handler model_artifacts/handler.py \
    --export-path model_artifacts

将模型工件上传到 Google Cloud Storage (GCS)
----------------------------------------

在这一步中,我们将 `模型工件 <https://github.com/pytorch/serve/tree/master/model-archiver#artifact-details>`__ 上传到 GCS,
例如模型文件或处理程序。将工件存储在 GCS 上的优势在于您可以在中央存储桶中跟踪工件。


.. code:: shell

    BUCKET_NAME = "your-bucket-name-unique"  # @param {type:"string"}
    BUCKET_URI = f"gs://{BUCKET_NAME}/"

    # 将工件复制到存储桶中
    !gsutil cp -r model_artifacts $BUCKET_URI

使用模型工件和预构建的 PyTorch 容器镜像创建 Vertex AI 模型
------------------------------------------------------

将模型工件上传到 GCS 存储桶后,您可以将 PyTorch 模型上传到 `Vertex AI 模型注册表 <https://cloud.google.com/vertex-ai/docs/model-registry/introduction>`__。
从 Vertex AI 模型注册表中,您可以概览您的模型,以便更好地组织、跟踪和训练新版本。为此,您可以使用
`Vertex AI SDK <https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk>`__
和这个 `预构建的 PyTorch 容器 <https://cloud.google.com/blog/products/ai-machine-learning/prebuilt-containers-with-pytorch-and-vertex-ai>`__。


.. code:: shell

    from google.cloud import aiplatform as vertexai
    PYTORCH_PREDICTION_IMAGE_URI = (
        "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest"
    )
    MODEL_DISPLAY_NAME = "stable_diffusion_1_5-unique"
    MODEL_DESCRIPTION = "stable_diffusion_1_5 container"

    vertexai.init(project='your_project', location='us-central1', staging_bucket=BUCKET_NAME)

    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        description=MODEL_DESCRIPTION,
        serving_container_image_uri=PYTORCH_PREDICTION_IMAGE_URI,
        artifact_uri=BUCKET_URI,
    )

将 Vertex AI 模型部署到端点
-------------------------

将模型上传到 Vertex AI 模型注册表后,您可以将其部署到 Vertex AI 端点。为此,您可以使用控制台或 Vertex AI SDK。在此
示例中,您将在 NVIDIA Tesla P100 GPU 和 n1-standard-8 机器上部署模型。您可以
指定您的机器类型。


.. code:: shell

    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=MODEL_DISPLAY_NAME,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_P100",
        accelerator_count=1,
        traffic_percentage=100,
        deploy_request_timeout=1200,
        sync=True,
    )

如果您按照这个 `notebook <https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/vertex_endpoints/torchserve/dreambooth_stablediffusion.ipynb>`__
操作,您还可以使用 Vertex AI SDK 获取在线预测,如下面的代码片段所示。


.. code:: shell

    instances = [{"prompt": "An examplePup dog with a baseball jersey."}]
    response = endpoint.predict(instances=instances)

    with open("img.jpg", "wb") as g:
        g.write(base64.b64decode(response.predictions[0]))

    display.Image("img.jpg")

使用模型工件和预构建的 PyTorch 容器镜像创建 Vertex AI 模型

更多资源
-------

本教程是使用供应商文档创建的。要参考供应商网站上的原始文档,请参阅
`torchserve 示例 <https://cloud.google.com/blog/products/ai-machine-learning/get-your-genai-model-going-in-four-easy-steps>`__。
