# AWD-AGP

Official implementation of **Adversarial Attack for Robust Watermark Protection Against Inpainting-based and Blind Watermark Removers**.

Paper: [ACM MM 2023 PDF](https://dl.acm.org/doi/pdf/10.1145/3581783.3612034)

This repository contains the core AWD-AGP attack code, model-adapter APIs, transfer-evaluation scripts, and the mask-location search components used in the paper. The third-party watermark-removal and inpainting models are referenced through lightweight wrappers in `source_models/`; users should install those projects and adapt checkpoints/paths locally.

## Repository Layout

- `inpainting/AWD_AGP/official_surrogate_generate_AE.py`: generate transferable adversarial examples with AWD-AGP.
- `inpainting/AWD_AGP/official_Attack.py`: attack objectives, perturbation optimization, integrated-gradient attribution, and AGP logic.
- `inpainting/AWD_AGP/official_transfer_attack.py`: transfer evaluation on inpainting-based removers.
- `inpainting/AWD_AGP/official_transfer_attack_for_wr.py`: transfer evaluation on blind watermark removers.
- `inpainting/AWD_AGP/official_evaluate.py`: metrics for inpainting-based removal results.
- `inpainting/AWD_AGP/official_evaluate_for_wr.py`: metrics for blind watermark-removal results.
- `inpainting/AWD_AGP/official_dataset.py`: dataset loading, watermark attachment, and mask handling.
- `inpainting/AWD_AGP/official_load_models.py`: central model-loader for inpainting and watermark-removal wrappers.
- `inpainting/AWD_AGP/official_utils.py`, `inpainting/AWD_AGP/official_noise_func.py`: shared utility and perturbation modules.
- `inpainting/AWD_AGP/source_models/`: adapter APIs for RFR, GMCNN, EdgeConnect, CR-Fill, Generative Inpainting, FcF, MAT, WDNet, DBWE, and SLBR.
- `inpainting/AWD_AGP/mask_RPN/`: Faster R-CNN based watermark-location proposal code; its training records are pseudo-labels produced by the superpixel-guided evolutionary search. See `docs/TRAIN_MASK_RPN.md` to train your own checkpoint.
- `inpainting/AWD_AGP/superpixel_fcn/`: imported superpixel segmentation dependency used by the mask-location search pipeline.

Compatibility wrappers `Attack.py`, `load_models.py`, and `utils.py` are kept for older scripts that imported the pre-extraction module names.

## Environment

The code was developed with PyTorch/CUDA and several image-processing packages. A minimal Python dependency list is provided in `requirements.txt`, but each remover/inpainting backend also has its own installation and checkpoint requirements.

Typical setup:

```bash
pip install -r requirements.txt
```

Then install or place the external model projects/checkpoints expected by the wrappers in `inpainting/AWD_AGP/source_models/`. Check `inpainting/AWD_AGP/official_load_models.py`, the corresponding file in `inpainting/AWD_AGP/source_models/`, and `WEIGHTS.md` when adapting paths.

## Quickstart Demo

Before running the full paper setup, users can generate one protected
watermarked image and test it with WDNet:

```bash
conda activate ensemble

python scripts/quickstart_wdnet_attack.py \
  --output-dir quickstart_outputs/wdnet_demo
```

By default the script generates a small synthetic clean image for the smoke test.
To use your own image, pass `--input /path/to/clean_image.jpg`. The protected
adversarial example is written to `quickstart_outputs/wdnet_demo/02_protected.png`,
and WDNet test outputs are saved in the same directory. The WDNet quickstart
remover is included under `third_party/inpainting/WDNet`. See
`docs/QUICKSTART_WDNET.md`.

## Google Drive Checkpoints

Full-paper checkpoints are expected to come from a separate weights package.
Download the prepared core archive from
[Google Drive](https://drive.google.com/open?id=1DQUgsaqpd3LnP_MLe9YYhswRv3DsPBoL).
The archive is named `AWD_AGP_weights_core_20260725.tar.gz` (about 1.7 GB) and
contains the expected `weights/` directory plus `README_DEPLOY_WEIGHTS.md`.
Optional checksum:
`57dd2f8aba072bc306468478d13b865b86c0a53aaa3fe937f125d204a0ceb7ac`. After
downloading it, extract it from the repository root:

```bash
tar -xzf AWD_AGP_weights_core_20260725.tar.gz -C /path/to/AWD_AGP
cd /path/to/AWD_AGP
python scripts/check_weights.py --profile core
```

The core archive includes Generative Inpainting, GMCNN, EdgeConnect, MAT, FcF,
CR-Fill, DBWE, SLBR, RFR-CelebA, Superpixel FCN, and the authors-provided
Mask RPN checkpoint `fasterrcnn_02191254_0`. WDNet is
already bundled for the quickstart under `third_party/inpainting/WDNet/`. The
RFR Places2 checkpoint is not included in the core archive. The full 23GB Mask
RPN snapshot directory is also not included; the archive keeps only the
checkpoint used by the AWD-AGP demos/pipeline. If any released weight link becomes unavailable, please obtain the
corresponding checkpoint from the original third-party repository whenever
possible, or replace it with an equivalent checkpoint supported by the wrapper.
The code also supports per-file Drive links through
`configs/weights_manifest.json`, `scripts/download_gdrive_weights.py`, and a
custom `--weights_dir` / `AWD_AGP_WEIGHTS_DIR`. See
`docs/GOOGLE_DRIVE_WEIGHTS.md` and `WEIGHTS.md`.

## Main Usage

Generate AWD-AGP adversarial examples:

```bash
python -m inpainting.AWD_AGP.official_surrogate_generate_AE \
  --dataset places2 \
  --target_models Matnet \
  --lossType perceptual_loss \
  --algorithm random_logo_alpha \
  --algorithm attribution_attack_for_multi_task3 \
  --InputImg_dir /path/to/input_images \
  --output_dir surrogate_mat_awdagp \
  --get_logo \
  --attach_logo \
  --budget 0.03 \
  --sign \
  --RPNRefineMask /path/to/rpn_masks
```

Evaluate transfer to an inpainting-based remover:

```bash
python -m inpainting.AWD_AGP.official_transfer_attack \
  --dataset places2 \
  --lossType perceptual_loss \
  --InputImg_dir /path/to/input_images \
  --experiment_dir inpainting/AWD_AGP/experiment_result/surrogate_mat_awdagp \
  --output_dir transfer_to_fcf \
  --target_model FcFnet \
  --algorithm optimal_mask_search \
  --get_logo \
  --attach_logo
```

Evaluate transfer to a blind watermark remover:

```bash
python -m inpainting.AWD_AGP.official_transfer_attack_for_wr \
  --dataset places2 \
  --lossType perceptual_loss \
  --InputImg_dir /path/to/input_images \
  --experiment_dir inpainting/AWD_AGP/experiment_result/surrogate_mat_awdagp \
  --output_dir transfer_to_wdnet \
  --target_model WDModel \
  --algorithm optimal_mask_search \
  --get_logo \
  --attach_logo
```

Run evaluation after transfer:

```bash
python -m inpainting.AWD_AGP.official_evaluate \
  --target_model FcFnet \
  --output_dir transfer_to_fcf \
  --InputImg_dir /path/to/watermarked_images \
  --source_dir surrogate_mat_awdagp \
  --get_logo
```

```bash
python -m inpainting.AWD_AGP.official_evaluate_for_wr \
  --target_model WDModel \
  --output_dir transfer_to_wdnet \
  --InputImg_dir /path/to/watermarked_images \
  --source_dir surrogate_mat_awdagp \
  --get_logo \
  --algorithm evaluate_rw
```

## Notes

- Outputs are written under `inpainting/AWD_AGP/experiment_result/<output_dir>/`.
- `--target_models` is used by the adversarial-example generation script; `--target_model` is used by transfer/evaluation scripts.
- If `random_logo_alpha.json` is absent, the dataset code falls back to `--watermark_alpha` when available, otherwise `0.1`.
- Some backend wrappers still contain local checkpoint paths from the research environment. Treat those paths as templates and adapt them to your installation.
- The repository intentionally does not vendor all third-party model code or checkpoints. See `WEIGHTS.md` for expected local checkpoint locations.

## License

The original AWD-AGP code is released under the Apache License 2.0. Third-party
code, models, and checkpoints retain their original licenses and terms; see
`THIRD_PARTY_NOTICES.md` for details.

## Citation

```bibtex
@inproceedings{lyu2023adversarial,
  title={Adversarial Attack for Robust Watermark Protection Against Inpainting-based and Blind Watermark Removers},
  author={Lyu, Mingzhi and Huang, Yi and Kong, Adams Wai-Kin},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={8396--8405},
  year={2023}
}
```
