.. meta::
  :description: New features, bundle-specific changes, and known issues in MONAI Model Zoo on ROCm 26.08
  :keywords: MONAI Model Zoo, release notes, ROCm-LS 26.08, ROCm, AMD

.. _model-zoo-whats-new:

**********************************************
What's new in MONAI Model Zoo on ROCm 26.08
**********************************************

ROCm-LS 26.08 is the first AMD ROCm-optimized release of the MONAI Model Zoo. Five bundles are inference-validated and optimized for AMD Instinct™ GPUs using MONAI Bundle overlay configurations.

New features
============

ROCm inference overlays for five bundles
----------------------------------------

AMD provides ``inference_rocm.json`` or ``inference_rocm.yaml`` overlay files for five bundles. The overlays apply a consistent set of ROCm performance optimizations without changing the upstream bundle's model weights.

- ``vista3d``. VISTA-3D multi-organ segmentation with zero-shot label prompts
- ``swin_unetr_btcv_segmentation``. Swin UNETR 13-organ abdominal CT segmentation
- ``wholeBody_ct_segmentation``. SegResNet 104-structure whole-body CT segmentation
- ``spleen_deepedit_annotation``. DeepEdit interactive spleen segmentation
- ``pancreas_ct_dints_segmentation``. DiNTS pancreas and tumor segmentation

ROCm optimization wedge
-----------------------

All five overlays apply the following optimizations:

- Channels-last 3D memory format, which reorders tensor layout to ``torch.channels_last_3d`` for improved memory access efficiency on AMD CDNA architectures.
- BF16 automatic mixed precision, which reduces memory bandwidth pressure and speeds up matrix operations by setting ``amp_kwargs={'dtype': torch.bfloat16}`` on the evaluator.
- ``torch.compile``, which compiles the PyTorch graph for optimized kernel fusion and dispatch on the ROCm HIP backend.

Where the upstream bundle doesn't already place weights on-device, the overlay also applies device-aware checkpoint loading, ``map_location=@device``. This applies to ``vista3d`` and ``swin_unetr_btcv_segmentation``.

Bundle-specific changes
-----------------------

Some overlays add settings that apply to a single bundle.

.. list-table::
  :header-rows: 1
  :widths: 28 72

  * - Bundle
    - Additional change
  * - ``swin_unetr_btcv_segmentation``
    - Re-declares ``network_def``, identical to upstream with ``use_checkpoint: false``, so the ROCm network binding applies cleanly. Also adds ``map_location=@device`` to the checkpointloader.
  * - ``spleen_deepedit_annotation``
    - Calls ``network_def.enable_gemm_transpose(True)`` when the method is present. This toggles a GEMM-based ``ConvTranspose3d`` upsample path optimized for ROCm. The path is an exact decomposition of the same weights and is effective only on ROCm builds. Also sets ``evaluator.compile = True``. The network is compiled through the overlay's ``torch.compile(@network)`` step.
  * - ``pancreas_ct_dints_segmentation``
    - Overrides ``SlidingWindowInferer`` parameters, ``roi_size=[96,96,96]``, ``sw_batch_size=8``, and ``overlap=0.625``, for improved GPU utilization on AMD hardware.

CI unit tests
-------------

AMD provides unit tests for each validated bundle in ``ci/unit_tests/test_<bundle_name>.py``. Tests exercise the full ``ConfigWorkflow`` inference pipeline with synthetic inputs and the ROCm overlay applied, without requiring pretrained weights. Tests run in the ``aisw-ci-builder-tester`` GPU CI pipeline.

Known issues
============

These issues apply to ROCm-LS 26.08.

.. list-table::
  :header-rows: 1
  :widths: 40 20 40

  * - Issue
    - Affected bundles
    - Workaround
  * - ``torch.compile`` increases first-inference latency because of JIT compilation. Later inferences are faster.
    - All five bundles
    - Run a warm-up pass before timing. Remove ``torch.compile`` from the ``initialize`` block in the overlay if first-call latency matters.
  * - BF16 AMP might produce marginally different numerical outputs compared to FP32 or FP16 reference runs. Segmentation masks are equivalent for clinical use.
    - All five bundles
    - Override ``amp_kwargs`` to ``{'dtype': torch.float16}`` or remove the AMP override to run in FP32.

Validated support matrix
========================

Validation used these component versions.

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - Component
    - Version
  * - ROCm
    - 10.0.0
  * - MONAI
    - 1.6.0
  * - PyTorch
    - A ROCm 10.0.0-compatible build
  * - Python
    - 3.12
  * - Ubuntu
    - 24.04
  * - GPUs
    - AMD Instinct™ MI355X, ``gfx950``, MI325X, ``gfx942``, and MI300X, ``gfx942``
