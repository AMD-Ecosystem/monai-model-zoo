.. meta::
  :description: Reference for the MONAI Model Zoo ROCm overlay configuration files and the keys each overlay sets
  :keywords: MONAI Model Zoo, ROCm overlay, inference_rocm.json, inference_rocm.yaml, MONAI Bundle, AMD

.. _rocm-overlays:

********************
ROCm overlays
********************

A ROCm overlay is a MONAI Bundle configuration file that AMD ships next to an upstream inference config. The overlay adds or overrides selected keys at load time. Model weights aren't modified. Only the runtime execution path is adapted for ROCm.

Layered configuration
=====================

MONAI Bundle supports layered configuration. A base ``inference.json`` or ``inference.yaml`` file provides the complete model definition. An overlay file adds or overrides specific keys when you run inference.

Pass the base file and the overlay together in ``--config_file`` as a JSON list. The overlay merges into the base config at load time. Upstream files on disk aren't changed.

.. code:: shell

   python -m monai.bundle run \
       --config_file "['models/bundle_name/configs/inference.json', \
                       'models/bundle_name/configs/inference_rocm.json']" \
       --bundle_root models/bundle_name

Replace ``bundle_name`` with a validated bundle directory name. Use ``.yaml`` for both paths when the bundle's upstream inference config is YAML. Among the validated bundles, only ``pancreas_ct_dints_segmentation`` uses YAML.

See :doc:`Installation <../install/installation>` for setup steps. See :doc:`Validated bundles <validated-bundles>` for per-bundle overlay paths.

Overlay files
=============

AMD provides one overlay per validated bundle, named ``inference_rocm.json`` or ``inference_rocm.yaml``. Each file is under ``models/bundle_name/configs/``, next to the upstream inference config.

JSON overlays use ``+imports`` to add ``$import torch`` without replacing the upstream import list. The ``pancreas_ct_dints_segmentation`` overlay sets ``imports`` to ``$import torch``.

Shared overlay keys
===================

Every overlay rebinds ``network`` and replaces ``initialize``.

.. list-table::
  :header-rows: 1
  :widths: 28 72

  * - Key
    - Effect
  * - ``network``
    - Moves the module to ``@device``, then to ``torch.channels_last_3d``, through ``$@network_def.to(@device).to(memory_format=torch.channels_last_3d)``.
  * - ``initialize``
    - | Sets determinism with seed 123.
      | Sets ``torch.backends.cudnn.deterministic`` to ``False``.
      | Sets ``evaluator.amp_kwargs`` to ``{'dtype': torch.bfloat16}``.
      | Loads weights through ``checkpointloader`` when that step is present.
      | Sets ``evaluator.network`` to ``torch.compile(@network)``.

``wholeBody_ct_segmentation``, ``spleen_deepedit_annotation``, and ``pancreas_ct_dints_segmentation`` guard the checkpoint load with ``if @load_pretrain``.
``vista3d`` and ``swin_unetr_btcv_segmentation`` call ``checkpointloader`` unconditionally.

Bundle-specific keys
====================

Some overlays set keys beyond the shared set.

.. list-table::
  :header-rows: 1
  :widths: 32 68

  * - Bundle
    - Extra overlay keys
  * - ``vista3d``
    - Replaces ``checkpointloader`` with ``map_location`` set to ``@device`` and ``load_dict`` bound to ``@network_def``.
  * - ``swin_unetr_btcv_segmentation``
    - Re-declares ``network_def`` with ``use_checkpoint: false``. Replaces ``checkpointloader`` with ``map_location`` set to ``@device`` and ``load_path`` set to ``@checkpoint``.
  * - ``spleen_deepedit_annotation``
    - Calls ``network_def.enable_gemm_transpose(True)`` when that method exists. Sets ``evaluator.compile`` to ``True``.
  * - ``pancreas_ct_dints_segmentation``
    - Replaces ``inferer`` with ``SlidingWindowInferer``, ``roi_size`` ``[96, 96, 96]``, ``sw_batch_size`` 8, and ``overlap`` 0.625.
