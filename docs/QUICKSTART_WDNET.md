# Quickstart: Generate and Test a Protected Image with WDNet

This quickstart gives users a small end-to-end path before they reproduce the
full paper experiments. It uses WDNet as a test blind watermark remover:

1. Load one input image.
2. Attach a visible demo watermark.
3. Optimize a bounded adversarial perturbation.
4. Save the protected watermarked image.
5. Run WDNet on the normal watermarked image and the protected image.

## Install the Third-Party Test Remover

WDNet is used only as a lightweight quickstart remover. A verified copy of the
WDNet code and `WDNet_G.pkl` checkpoint is included in this repository:

```text
third_party/inpainting/WDNet
third_party/inpainting/WDNet/WDNet_G.pkl
```

The vendored copy comes from:

```text
https://github.com/MRUIL/WDNet.git
```

If you want to use an external copy instead, pass explicit paths:

```bash
export AWD_AGP_EXTERNAL_INPAINTING_ROOT=/path/to/parent-that-contains-inpainting
export AWD_AGP_WDNET_CKPT=/path/to/inpainting/WDNet/WDNet_G.pkl
```

## Run

From the repository root:

```bash
conda activate ensemble

python scripts/quickstart_wdnet_attack.py \
  --output-dir quickstart_outputs/wdnet_demo \
  --steps 30 \
  --budget 0.03
```

By default, the script generates a synthetic clean input image. To test your own
image, add `--input /path/to/clean_image.jpg`.

Expected outputs:

```text
quickstart_outputs/wdnet_demo/00_input.png
quickstart_outputs/wdnet_demo/01_watermarked.png
quickstart_outputs/wdnet_demo/02_protected.png
quickstart_outputs/wdnet_demo/03_wdnet_removed_watermarked.png
quickstart_outputs/wdnet_demo/04_wdnet_removed_protected.png
quickstart_outputs/wdnet_demo/05_perturbation_x10.png
quickstart_outputs/wdnet_demo/comparison.png
quickstart_outputs/wdnet_demo/metrics.json
```

`02_protected.png` is the protected adversarial example. The two WDNet outputs
let users visually inspect whether the perturbation makes blind watermark
removal less effective. `metrics.json` records the WDNet output error on the
watermark region and the actual perturbation magnitude.

## Notes

- This quickstart is a smoke-test path, not the full paper evaluation suite.
- The full paper reproduction still requires the other third-party remover and
  inpainting backends listed in `WEIGHTS.md`.
- The script uses CUDA because the current WDNet wrapper calls `.cuda()`.
