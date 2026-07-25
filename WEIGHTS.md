# Model Weights

This GitHub-ready copy does not include large model weights, third-party checkpoints, or generated experiment outputs.

Place or configure weights as follows. The canonical machine-readable list is `configs/weights_manifest.json`; fill its Google Drive fields after uploading your checkpoints.

| Component | CLI argument | Environment variable | Default local path | Notes |
| --- | --- | --- | --- | --- |
| Global weights root | `--weights_dir` | `AWD_AGP_WEIGHTS_DIR` | `weights/` | Used by all entries whose path starts with `weights/`. |
| RFR places2 | `--rfr_places2_checkpoint` | `AWD_AGP_RFR_PLACES2_CKPT` | `weights/RFR/checkpoint_places2.pth` | Required when using `RFRnet` with `--dataset places2`. |
| Generative Inpainting | `--generative_checkpoint`, `--generative_config` | `AWD_AGP_GENERATIVE_CKPT`, `AWD_AGP_GENERATIVE_CONFIG` | `weights/generative_inpainting/...` | Checkpoint and config are both represented in `configs/weights_manifest.json`. |
| GMCNN | `--gmcnn_checkpoint_dir`, `--gmcnn_config` | `AWD_AGP_GMCNN_CKPT_DIR`, `AWD_AGP_GMCNN_CONFIG` | `weights/inpainting_gmcnn/...` | Checkpoint argument is the directory containing `.pth` files. |
| EdgeConnect | `--edgeconnect_config` | `AWD_AGP_EDGECONNECT_CONFIG` | `weights/edge_connect/checkpoints/<dataset>/pipeline_config.yml` | The external EdgeConnect project still supplies model code/checkpoints referenced by the config. |
| MAT | `--mat_checkpoint` | `AWD_AGP_MAT_CKPT` | `weights/MAT/pretrained/Places_512_FullData_real.pkl` | Full-paper inpainting backend. |
| FcF | `--fcf_checkpoint` | `AWD_AGP_FCF_CKPT` | `weights/FcF_Inpainting/G.pt` | Full-paper inpainting backend. |
| CR-Fill | `--crfill_objrmv_config`, `--crfill_places_config` | `AWD_AGP_CRFILL_OBJRMV_CONFIG`, `AWD_AGP_CRFILL_PLACES_CONFIG` | `weights/crfill/checkpoints/.../*.yaml` | Config files point to CR-Fill checkpoints. |
| WDNet | `--wdnet_checkpoint` | `AWD_AGP_WDNET_CKPT` | `third_party/inpainting/WDNet/WDNet_G.pkl` | Bundled for quickstart; can be overridden. |
| DBWE | `--dbwe_checkpoint` | `AWD_AGP_DBWE_CKPT` | `weights/DBWEModel/27kpng_model_best.pth.tar` | Blind watermark remover. |
| SLBR | `--slbr_checkpoint` | `AWD_AGP_SLBR_CKPT` | `weights/SLBR/model_best.pth.tar` | Blind watermark remover. |
| Authors-provided Mask RPN | n/a | `AWD_AGP_MASK_RPN_CKPT` | `weights/mask_RPN/checkpoints/fasterrcnn_02191254_0` | Trained watermark-location proposal checkpoint used by the AWD-AGP demos/pipeline. |
| Optional Mask RPN snapshot directory | n/a | `AWD_AGP_MASK_RPN_CKPT_DIR` | `weights/mask_RPN/checkpoints/` | Optional full training snapshot directory; not needed when the authors-provided checkpoint above is available. |
| Superpixel FCN | function argument | `AWD_AGP_SUPERPIXEL_CKPT` | `weights/superpixel_fcn/pretrain_ckpt/SpixelNet_bsd_ckpt.tar` | Used by mask-location search. |

Large weights should be released separately, for example through GitHub Releases, Google Drive, Hugging Face, or institutional storage.

## Bundled Quickstart Checkpoint

The WDNet quickstart remover and checkpoint are bundled under
`third_party/inpainting/WDNet/`, including `WDNet_G.pkl`. This keeps the basic
protected-image generation and blind-remover test path runnable from a fresh
checkout.

## Full Paper Checkpoints

Several full paper backends require checkpoints larger than GitHub's normal
100 MB single-file limit, for example MAT/FcF/RFR and some partial-convolution
weights. Do not commit those large files directly to ordinary git history. Use
GitHub Releases, Git LFS, Hugging Face, Google Drive, or institutional storage,
then point the CLI arguments/environment variables above to the downloaded
files.
