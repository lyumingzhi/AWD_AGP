# Third-Party Notices

This repository is the official implementation of AWD-AGP. The original AWD-AGP
code is released under the Apache License 2.0 in `LICENSE`. Some directories,
wrappers, bundled files, and external checkpoints come from third-party projects
or are used to interface with third-party models. Those components remain under
their original licenses and terms.

## Vendored or Included Components

- `inpainting/AWD_AGP/superpixel_fcn/`: Superpixel FCN implementation. The
  directory includes its own MIT license at
  `inpainting/AWD_AGP/superpixel_fcn/LICENSE`.
- `inpainting/AWD_AGP/mask_RPN/`: Faster R-CNN based watermark-location proposal
  code adapted from a simplified Faster R-CNN implementation. Some files retain
  upstream Apache-2.0 notices. The authors-provided trained checkpoint
  `weights/mask_RPN/checkpoints/fasterrcnn_02191254_0` is released as part of
  the AWD-AGP weights package for research reproduction of the paper.
- `third_party/inpainting/WDNet/`: WDNet quickstart remover code and checkpoint
  used to demonstrate blind watermark-removal evaluation. Redistribution and use
  of this component follow the terms of the original WDNet project/authors.

## External Model Wrappers

The files in `inpainting/AWD_AGP/source_models/` adapt AWD-AGP to external
inpainting and watermark-removal models, including RFR, Generative Inpainting,
GMCNN, EdgeConnect, MAT, FcF, CR-Fill, WDNet, DBWE, and SLBR. The wrappers are
part of AWD-AGP, but the underlying model implementations and checkpoints remain
third-party materials and should be obtained from their original repositories or
from the released weights package when redistribution is permitted.

## Weights

The core weights archive linked in the README contains checkpoints/config files
needed for reproducibility. Third-party checkpoints retain their original
licenses and usage terms. If a released weight link becomes unavailable, please
refer to the corresponding original third-party repository whenever possible.
