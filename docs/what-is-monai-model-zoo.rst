.. meta::
  :description: What MONAI Model Zoo on ROCm is, including the ROCm overlay port and inference optimizations
  :keywords: MONAI Model Zoo, ROCm overlay, AMD Instinct, channels-last, BF16, torch.compile

.. _what-is-monai-model-zoo:

**************************************
What is MONAI Model Zoo on ROCm
**************************************

MONAI Model Zoo on ROCm provides an inference-optimized MONAI Model Zoo port for AMD GPUs. ROCm overlay configuration files cover five volumetric CT segmentation bundles. The overlays layer on top of the upstream bundle configurations through the MONAI Bundle override mechanism without changing model weights.

Features include:

- Channels-last 3D memory format, which reorders tensor memory layout for improved memory access on AMD CDNA architectures.
- BF16 automatic mixed precision, which reduces memory bandwidth pressure and improves compute throughput.
- ``torch.compile`` graph compilation for optimized kernel dispatch on the ROCm HIP backend.
- Device-aware checkpoint loading for bundles that don't already place weights on-device.
