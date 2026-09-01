.. meta::
  :description: What MONAI Model Zoo on ROCm is, including the ROCm overlay port and inference optimizations
  :keywords: MONAI Model Zoo, ROCm overlay, AMD Instinct, channels-last, BF16, torch.compile

.. _what-is-monai-model-zoo:

**************************************
What is MONAI Model Zoo on ROCm
**************************************

MONAI Model Zoo on ROCm provides a MONAI Model Zoo port for inference on AMD GPUs.
ROCm overlay configuration files cover five volumetric CT segmentation bundles.
The MONAI Bundle override mechanism layers each overlay on the upstream bundle configuration without changing model weights.

The overlays configure these inference behaviors.

- Channels-last 3D memory format, which reorders the tensor memory layout on AMD CDNA architectures.
- BF16 automatic mixed precision for inference.
- ``torch.compile()`` graph compilation on the ROCm HIP backend.
- Device-aware checkpoint loading for bundles that don't already place weights on-device.
