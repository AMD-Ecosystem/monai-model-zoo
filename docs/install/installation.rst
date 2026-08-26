.. meta::
  :description: Install the MONAI Model Zoo ROCm overlays and run inference on AMD Instinct GPUs
  :keywords: ROCm-LS, life sciences, MONAI Model Zoo installation, MONAI Model Zoo on ROCm

.. _installing-model-zoo:

********************************************
MONAI Model Zoo on ROCm installation
********************************************

MONAI Model Zoo on ROCm can be installed using `AMD PyPI <https://pypi.amd.com/simple/>`_ in a Docker container or on bare metal.

System requirements:

+--------------+----------------+----------------+----------------------------------+
| ROCm version | Ubuntu version | Python version | AMD Instinct™ GPU (tested)       |
+==============+================+================+==================================+
| 10.0.0       | 24.04          | 3.12           | MI355X, MI325X, MI300X           |
+--------------+----------------+----------------+----------------------------------+


Setting up the environment
============================

Set up the environment for installing MONAI Model Zoo on ROCm as follows:

1. Optionally start an Ubuntu 24.04 Docker container.

   Use the ROCm PyTorch base image for a preconfigured environment.

   .. code:: shell

      docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true   \
      --shm-size=512GB --network=host --device=/dev/kfd        \
      --device=/dev/dri --group-add video -it                  \
      -v $HOME:$HOME  --name ${LOGNAME}_monai                  \
      rocm/dev-ubuntu-24.04:10.0.0-complete

2. Install ROCm and PyTorch.

   Install ROCm using the `ROCm Quick Start guide <https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html>`_. After ROCm is installed, install PyTorch for ROCm. See the `MONAI on ROCm installation guide <https://rocm.docs.amd.com/projects/monai/en/latest/install/installation.html>`_ for environment setup.

Installing MONAI using AMD PyPI
===============================

Install the `MONAI <https://rocm.docs.amd.com/projects/monai/en/latest/>`_ package, then clone the overlay repository.

1. Install MONAI.

   .. code:: shell

      export ROCM_VERSION=10.0.0
      pip install "amd-monai[fire]" \
          --extra-index-url=https://pypi.amd.com/rocm-${ROCM_VERSION}/simple/

2. Clone the AMD model-zoo repository.

   The AMD ROCm overlay files are in the AMD model-zoo repository.

   .. code:: shell

      git clone https://github.com/AMD-Ecosystem/model-zoo
      cd model-zoo

Using a ROCm overlay
====================

Download a bundle from the MONAI model registry, then run inference with the overlay.

1. Download a bundle.

   Replace ``bundle_name`` with the bundle you want to download.

   .. code:: shell

      python -m monai.bundle download bundle_name \
          --bundle_dir models/

2. Run inference with the ROCm overlay.

   Run inference using the base inference configuration and the AMD ROCm overlay. The overlay merges into the base configuration at load time. No upstream files are changed. See :doc:`ROCm overlays <../reference/rocm-overlays>` for information on overlay keys.

   .. code:: shell

      python -m monai.bundle run \
          --config_file "['models/bundle_name/configs/inference.json', \
                          'models/bundle_name/configs/inference_rocm.json']" \
          --bundle_root models/bundle_name \
          --dataset_dir input_dir \
          --output_dir output_dir

   .. note::

      Configuration files are JSON or yaml, depending on the bundle. For bundle configuration paths and input keys, see :doc:`Validated bundles <../reference/validated-bundles>`.
