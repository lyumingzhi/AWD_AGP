# Model Weights

This GitHub-ready copy does not include large model weights, third-party checkpoints, or generated experiment outputs.

Place or configure weights as follows:

| Component | CLI argument | Environment variable | Notes |
| --- | --- | --- | --- |
| RFR places2 | `--rfr_places2_checkpoint` | `AWD_AGP_RFR_PLACES2_CKPT` | Required when using `RFRnet` with `--dataset places2`. |
| WDNet | `--wdnet_checkpoint` | `AWD_AGP_WDNET_CKPT` | Falls back to `inpainting/WDNet/WDNet_G.pkl` for compatibility. |
| DBWE | `--dbwe_checkpoint` | `AWD_AGP_DBWE_CKPT` | Required when using `DBWEModel`. |
| SLBR | `--slbr_checkpoint` | `AWD_AGP_SLBR_CKPT` | Falls back to `inpainting/SLBR/model_best.pth (1).tar` for compatibility. |
| Mask RPN | n/a | n/a | Put proposal checkpoints under `mask_RPN/checkpoints/` or pass the result directory via `--RPNRefineMask`. |
| Superpixel FCN | function argument | n/a | Put `SpixelNet_bsd_ckpt.tar` under `superpixel_fcn/pretrain_ckpt/`, or pass a path to `get_superpixel_model(pretrained=...)`. |

Large weights should be released separately, for example through GitHub Releases, Google Drive, Hugging Face, or institutional storage.
