.. meta::
  :description: Install the MONAI Model Zoo ROCm overlays and run inference on AMD Instinct GPUs
  :keywords: ROCm-LS, life sciences, MONAI Model Zoo installation, MONAI Model Zoo on ROCm

.. _installing-model-zoo:

********************************************
MONAI Model Zoo on ROCm installation
********************************************

MONAI Model Zoo on ROCm installation requires `MONAI on ROCm <https://rocm.docs.amd.com/projects/monai/en/latest/install/installation.html>`_.

1. Install `MONAI on ROCm <https://rocm.docs.amd.com/projects/monai/en/latest/install/installation.html>`_.

2. Clone the AMD model-zoo repository.

   The AMD ROCm overlay files are in the AMD model-zoo repository.

   .. code:: shell

      git clone https://github.com/AMD-Ecosystem/model-zoo
      cd model-zoo

3. Download a bundle from the MONAI model registry.

   Replace ``bundle_name`` with the bundle you want to download, for example ``vista3d``.

   .. code:: shell

      python -m monai.bundle download bundle_name \
          --bundle_dir models/

4. Run inference with the ROCm overlay.

   Pass the base inference configuration and the AMD ROCm overlay as a JSON list. The overlay merges into the base configuration at load time. 

   .. code:: shell

      python -m monai.bundle run \
          --config_file "['models/bundle_name/configs/inference.json', \
                          'models/bundle_name/configs/inference_rocm.json']" \
          --bundle_root models/bundle_name \
          --dataset_dir input_dir \
          --output_dir output_dir

   .. note:: 
      
      Configuration files can be in either JSON or YAML format, depending on the bundle. The input key also varies by bundle. For information about overlay keys, see :doc:`ROCm overlays <../reference/rocm-overlays>`. For information about overlay paths, configuration format, and input keys, see :doc:`Validated bundles <../reference/validated-bundles>` .
