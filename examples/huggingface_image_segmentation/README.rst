Image Segmentation Demo (using Huggingface Model)
=================================================

This demo shows how to use an Image Semantic Segmentation model from
Huggingface with MERA software stack. The demo comprises installation of
the required library followed by compilation of the model and running
inference with MERA software stack.

Setup
-----

-  ``mera-models`` python library is required. To install use ``pip install mera-models``


Model export and compilation
----------------------------

The model can be compiled either by exporting it directly from
Huggingface or by using the pre-exported model.

Export model from Huggingface and compile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  To export the model directly from Huggingface and deploy it, use the
   below mentioned code.
-  This will also save the ONNX model in ``./source_model_files/``
-  This will deploy the model for the default target ``IP`` for running
   on the card.
-  Other target can be specified using *–target* argument, for example
   to deploy for the target ``Interpreter``, add following to the below
   mentioned code: ``--target interpreter``

.. code:: bash

   python deploy.py --model_id "nvidia/segformer-b0-finetuned-ade-512-512"

Deploy using pre-exported ONNX model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  If the ONNX model has already been exported, it can be used for the
   deployment:

.. code:: bash

   python deploy.py --model_id "source_model_files/nvidia__segformer-b0-finetuned-ade-512-512_onnx"

Inference
---------

Inferencing can be done using the compiled model as well as the ONNX
model exported from Huggingface.

Inference using compiled model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  To run inference using the compiled model, use the below mentioned
   code.
-  This will run inference for the default target ``IP`` for the card.
-  Other target can be specified using *–target* argument, for example
   to run inference for the target ``Interpreter``, add following to the
   below mentioned code: ``--target interpreter``

.. code:: bash

   python demo_model.py --model_path ./deploy_segformer-b0-512-512

Inference using exported ONNX model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  To run inference using the exported ONNX model, use the below
   mentioned code.

.. code:: bash

   python demo_model.py --model_path "source_model_files/nvidia__segformer-b0-finetuned-ade-512-512_onnx"

Appendix: Supported model
name: ``segformer-b0``	
model_id: ``nvidia/segformer-b0-finetuned-ade-512-512``	
params: 4M
fp32 onnx size: 15 MB

