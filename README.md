# AWD-AGP

Official implementation of **Adversarial Attack for Robust Watermark Protection Against Inpainting-based and Blind Watermark Removers**.

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
- `inpainting/AWD_AGP/mask_RPN/`: Faster R-CNN based watermark-location proposal code.
- `inpainting/AWD_AGP/superpixel_fcn/`: imported superpixel segmentation dependency used by the mask-location search pipeline.

Compatibility wrappers `Attack.py`, `load_models.py`, and `utils.py` are kept for older scripts that imported the pre-extraction module names.

## Environment

The code was developed with PyTorch/CUDA and several image-processing packages. A minimal Python dependency list is provided in `requirements.txt`, but each remover/inpainting backend also has its own installation and checkpoint requirements.

Typical setup:

```bash
pip install -r requirements.txt
```

Then install or place the external model projects/checkpoints expected by the wrappers in `inpainting/AWD_AGP/source_models/`. Check `inpainting/AWD_AGP/official_load_models.py`, the corresponding file in `inpainting/AWD_AGP/source_models/`, and `WEIGHTS.md` when adapting paths.

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
