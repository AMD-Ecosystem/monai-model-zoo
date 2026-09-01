.. meta::
  :description: Architecture, overlay paths, and inference input keys for validated MONAI Model Zoo bundles
  :keywords: MONAI Model Zoo, validated bundles, vista3d, swin_unetr_btcv_segmentation, wholeBody_ct_segmentation, spleen_deepedit_annotation, pancreas_ct_dints_segmentation, ROCm

.. _validated-bundles:

***********************
Validated bundles
***********************

AMD has validated five inference-optimized bundles for AMD MI355X, MI325X, and MI300X GPUs with ROCm 10.0.0, Ubuntu 24.04, Python 3.12, and MONAI 1.6.0.

AMD provides a ROCm overlay configuration, ``inference_rocm.json`` or ``inference_rocm.yaml``, for each bundle.
See :doc:`ROCm overlays <rocm-overlays>` for merge behavior and shared keys.
The overlay applies these settings on top of the unmodified upstream inference configuration.

- Channels-last 3D memory format, ``torch.channels_last_3d``.
- BF16 automatic mixed precision, ``amp_kwargs={'dtype': torch.bfloat16}``.
- ``torch.compile()`` graph compilation.
- Device-aware checkpoint loading, ``map_location=@device``, where the upstream bundle doesn't already place weights on-device.

Each bundle targets a volumetric CT segmentation task.

.. list-table::
  :header-rows: 1
  :widths: 26 40 22 12

  * - Bundle
    - Task
    - Architecture
    - Modality
  * - ``vista3d``
    - Multi-organ segmentation, 130+ structures, with zero-shot label prompts
    - VISTA-3D
    - CT
  * - ``swin_unetr_btcv_segmentation``
    - 13-organ abdominal CT segmentation, BTCV Challenge
    - Swin UNETR
    - CT
  * - ``wholeBody_ct_segmentation``
    - 104-structure whole-body CT segmentation, TotalSegmentator
    - SegResNet
    - CT
  * - ``spleen_deepedit_annotation``
    - Interactive spleen segmentation with positive and negative click guidance
    - DynUNet
    - CT
  * - ``pancreas_ct_dints_segmentation``
    - Pancreas and pancreatic tumor segmentation, NAS-discovered architecture
    - DiNTS
    - CT

vista3d
=======

``vista3d`` is a class-prompted volumetric CT segmentation bundle.

.. list-table::
  :widths: 30 70

  * - Full name
    - VISTA-3D: Versatile Imaging SegmenTation and Annotation
  * - Upstream version
    - 0.5.11
  * - Task
    - Multi-organ segmentation in CT with zero-shot label prompts, 130+ structures
  * - Architecture
    - VISTA-3D, a custom encoder-decoder with class-prompt conditioning
  * - Modality
    - CT
  * - Patch size
    - 128 x 128 x 128
  * - Training data
    - Task09_Spleen, Medical Segmentation Decathlon, plus internal multi-organ data
  * - ROCm overlay
    - ``models/vista3d/configs/inference_rocm.json``
  * - Configuration format
    - JSON
  * - Input key
    - ``input_dict``, with ``image`` and ``label_prompt``

swin_unetr_btcv_segmentation
============================

``swin_unetr_btcv_segmentation`` is a 13-organ abdominal CT segmentation bundle.

.. list-table::
  :widths: 30 70

  * - Full name
    - Swin UNETR BTCV Multi-organ Segmentation
  * - Upstream version
    - 0.5.8
  * - Task
    - 13-organ abdominal CT segmentation, Beyond the Cranial Vault Challenge
  * - Architecture
    - Swin UNETR, a Swin Transformer encoder with a UNet decoder
  * - Modality
    - CT
  * - Patch size
    - 96 x 96 x 96
  * - Output channels
    - 14, background plus 13 organs
  * - Training data
    - BTCV Challenge dataset, Synapse
  * - ROCm overlay
    - ``models/swin_unetr_btcv_segmentation/configs/inference_rocm.json``
  * - Configuration format
    - JSON
  * - Input key
    - ``dataset_dir``

wholeBody_ct_segmentation
=========================

``wholeBody_ct_segmentation`` is a 104-structure whole-body CT segmentation bundle.

.. list-table::
  :widths: 30 70

  * - Full name
    - Whole Body CT Segmentation
  * - Upstream version
    - 0.2.7
  * - Task
    - 104-structure whole-body CT segmentation covering major organs, bones, muscles, and vasculature
  * - Architecture
    - SegResNet
  * - Modality
    - CT
  * - Patch size
    - 96 x 96 x 96
  * - Output channels
    - 105, background plus 104 structures
  * - Training data
    - TotalSegmentator dataset
  * - ROCm overlay
    - ``models/wholeBody_ct_segmentation/configs/inference_rocm.json``
  * - Configuration format
    - JSON
  * - Input key
    - ``dataset_dir``

spleen_deepedit_annotation
==========================

``spleen_deepedit_annotation`` is an interactive spleen segmentation bundle.

.. list-table::
  :widths: 30 70

  * - Full name
    - Spleen DeepEdit Interactive Segmentation
  * - Upstream version
    - 0.5.8
  * - Task
    - Interactive spleen segmentation with positive and negative click guidance, DeepEdit
  * - Architecture
    - DynUNet
  * - Modality
    - CT
  * - Patch size
    - 128 x 128 x 128
  * - Output channels
    - 2, background plus spleen
  * - Training data
    - Task09_Spleen, Medical Segmentation Decathlon
  * - ROCm overlay
    - ``models/spleen_deepedit_annotation/configs/inference_rocm.json``
  * - Configuration format
    - JSON
  * - Input key
    - ``dataset_dir``

pancreas_ct_dints_segmentation
==============================

``pancreas_ct_dints_segmentation`` is a NAS-discovered pancreas and tumor segmentation bundle.

.. list-table::
  :widths: 30 70

  * - Full name
    - Pancreas and Tumor DiNTS Segmentation
  * - Upstream version
    - 0.5.2
  * - Task
    - Pancreas and pancreatic tumor segmentation with a NAS-discovered architecture
  * - Architecture
    - DiNTS, Differentiable Neural Architecture Search
  * - Modality
    - CT
  * - Patch size
    - 96 x 96 x 96
  * - Output channels
    - 3, background plus pancreas plus tumor
  * - Training data
    - Task07_Pancreas, Medical Segmentation Decathlon
  * - ROCm overlay
    - ``models/pancreas_ct_dints_segmentation/configs/inference_rocm.yaml``
  * - Configuration format
    - YAML
  * - Input key
    - ``dataset_dir``

CI validation
==============

AMD provides unit tests for each bundle in ``ci/unit_tests/test_bundle_name.py``.
The tests run the ``ConfigWorkflow`` inference pipeline with a synthetic input.
The ROCm overlay applies on a ROCm build when ``torch.version.hip is not None``.
Bundles that guard weight loading with ``@load_pretrain`` run without pretrained weights.
