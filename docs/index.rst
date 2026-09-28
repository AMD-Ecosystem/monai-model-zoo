.. SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. meta::
  :description: MONAI Model Zoo is a collection of medical imaging models in MONAI Bundle format with an AMD ROCm overlay port for inference.
  :keywords: ROCm-LS, life sciences, MONAI Model Zoo, MONAI Model Zoo on ROCm document, AMD MONAI Model Zoo, ROCm MONAI Model Zoo

.. _index:

*******************************************
MONAI Model Zoo on ROCm documentation
*******************************************

The MONAI Model Zoo is a collection of medical imaging models in the `MONAI Bundle <https://monai.readthedocs.io/en/stable/bundle_intro.html>`_ format maintained by
`Medical Open Network for Artificial Intelligence (MONAI) <https://project-monai.github.io/>`_.
Each bundle packages a model's weights, configuration, and inference scripts for download and inference.

MONAI Model Zoo on ROCm provides a MONAI Model Zoo port for inference on AMD GPUs. The MONAI Bundle override mechanism layers each overlay on the upstream bundle configuration without changing model weights.

The overlays configure these inference behaviors.

- Channels-last 3D memory format, which reorders the tensor memory layout on AMD CDNA architectures.
- BF16 automatic mixed precision for inference.
- ``torch.compile()`` graph compilation on the ROCm HIP backend.
- Device-aware checkpoint loading for bundles that don't already place weights on-device.

The code is open and hosted at `<https://github.com/AMD-Ecosystem/monai-model-zoo>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :ref:`installing-model-zoo`

  .. grid-item-card:: Reference

    * :ref:`validated-bundles`
    * :ref:`rocm-overlays`

To contribute to MONAI Model Zoo on ROCm, see
`Contributing to monai-model-zoo <https://github.com/AMD-Ecosystem/monai-model-zoo/blob/amd-develop/CONTRIBUTING.md>`_.

Licensing information is on the :doc:`Licensing <license>` page.
