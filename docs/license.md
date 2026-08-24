# License

## MONAI Model Zoo repository

The MONAI Model Zoo repository, including the AMD ROCm overlay configuration files and CI unit tests AMD contributes, is released under the Apache License, Version 2.0.

```{include} ../LICENSE
```

## Individual bundles

Each bundle in the MONAI Model Zoo carries its own license covering both the software, its configuration files and scripts, and the model weights. Consult the `docs/data_license.txt` and `LICENSE` files inside each bundle directory before use. The five bundles validated in ROCm-LS 26.08 all carry Apache 2.0 licenses for the bundle software. Model weight licenses might differ.

## AMD ROCm overlay files

The AMD-contributed overlay files, `inference_rocm.json` and `inference_rocm.yaml`, and CI unit tests are released under Apache 2.0 with the following notice:

```text
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

## Data licenses

The bundles validated in this release use publicly available benchmark datasets. Each dataset carries its own license and terms of use.

| Bundle | Dataset | License / terms |
| --- | --- | --- |
| `vista3d`, `spleen_deepedit_annotation` | Task09_Spleen, Medical Segmentation Decathlon | CC BY-SA 4.0 |
| `swin_unetr_btcv_segmentation` | BTCV Challenge dataset, Synapse | See the [Synapse terms](https://www.synapse.org/#!Synapse:syn3193805) |
| `wholeBody_ct_segmentation` | TotalSegmentator | CC BY 4.0 |
| `pancreas_ct_dints_segmentation` | Task07_Pancreas, Medical Segmentation Decathlon | CC BY-SA 4.0 |

AMD makes no statement of the suitability of any model for a particular clinical task, especially not for therapeutic or diagnostic use.
