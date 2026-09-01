.. meta::
  :description: New features, bundle-specific changes, and known issues in MONAI Model Zoo on ROCm 26.08
  :keywords: MONAI Model Zoo, release notes, ROCm-LS 26.08, ROCm, AMD

.. _model-zoo-whats-new:

*************************************************
Release notes for MONAI Model Zoo on ROCm 26.08
*************************************************

ROCm-LS 26.08 is the first MONAI Model Zoo release for AMD ROCm.
AMD validated five bundles for inference on AMD Instinct™ GPUs using MONAI Bundle overlay configurations.

New features
============

ROCm-LS 26.08 adds overlay files, shared overlay settings, bundle-specific overlay keys, and CI tests.

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

All five overlays apply the same settings.

- Channels-last 3D memory format, which reorders the tensor layout to ``torch.channels_last_3d`` on AMD CDNA architectures.
- BF16 automatic mixed precision, which sets ``amp_kwargs={'dtype': torch.bfloat16}`` on the evaluator.
- ``torch.compile()``, which compiles the PyTorch graph on the ROCm HIP backend.

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
    - | Re-declares ``network_def``, identical to upstream with ``use_checkpoint: false``, so the ROCm network binding applies.
      | Adds ``map_location=@device`` to the checkpointloader.
  * - ``spleen_deepedit_annotation``
    - | Calls ``network_def.enable_gemm_transpose(True)`` when the method is present.
      | This call toggles a GEMM-based ``ConvTranspose3d`` upsample path for ROCm.
      | The path is an exact decomposition of the same weights and is effective only on ROCm builds.
      | Sets ``evaluator.compile = True``.
      | The overlay's ``torch.compile(@network)`` step compiles the network.
  * - ``pancreas_ct_dints_segmentation``
    - Overrides ``SlidingWindowInferer`` parameters with ``roi_size=[96,96,96]``, ``sw_batch_size=8``, and ``overlap=0.625``.

CI unit tests
-------------

AMD provides unit tests for each validated bundle in ``ci/unit_tests/test_bundle_name.py``.
The tests run the ``ConfigWorkflow`` inference pipeline with synthetic inputs and the ROCm overlay without requiring pretrained weights.
The ``aisw-ci-builder-tester`` GPU CI pipeline runs the tests.

Known issues
============

These issues apply to ROCm-LS 26.08.

.. list-table::
  :header-rows: 1
  :widths: 40 20 40

  * - Issue
    - Affected bundles
    - Workaround
  * - ``torch.compile()`` increases first-inference latency because of JIT compilation.
    - All five bundles
    - Run a warm-up pass before timing. If first-call latency matters, remove ``torch.compile()`` from the ``initialize`` block in the overlay.
  * - BF16 AMP might produce different numerical outputs compared to FP32 or FP16 reference runs.
    - All five bundles
    - Override ``amp_kwargs`` to ``{'dtype': torch.float16}`` or remove the AMP override to run in FP32.
