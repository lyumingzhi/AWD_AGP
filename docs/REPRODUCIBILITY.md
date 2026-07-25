# Reproducibility Scope

This repository is intended to be the official implementation of AWD-AGP. It can
reproduce the main method pipeline from the paper when the required external
assets are installed locally.

## What Is Reproducible From This Repository

The released code supports these paper components:

| Paper component | Repository support | Entry points |
| --- | --- | --- |
| Generate protected visible-watermark images with AWD-AGP | Supported | `inpainting/AWD_AGP/official_surrogate_generate_AE.py` |
| Attack/optimization objectives, AGP logic, integrated-gradient attribution | Supported | `inpainting/AWD_AGP/official_Attack.py` |
| Transfer evaluation on inpainting-based removers | Supported | `inpainting/AWD_AGP/official_transfer_attack.py` |
| Transfer evaluation on blind watermark removers | Supported | `inpainting/AWD_AGP/official_transfer_attack_for_wr.py` |
| Quantitative evaluation scripts | Supported | `inpainting/AWD_AGP/official_evaluate.py`, `inpainting/AWD_AGP/official_evaluate_for_wr.py` |
| Superpixel-guided evolutionary mask search | Supported | `inpainting/AWD_AGP/official_surrogate_generate_AE_mask_DE_without_noise.py`, `inpainting/AWD_AGP/official_utils.py` |
| Train/use the authors' Mask RPN proposal model | Supported | `inpainting/AWD_AGP/mask_RPN/`, `docs/TRAIN_MASK_RPN.md` |
| WDNet smoke test that outputs a protected image and remover output | Supported from a fresh checkout | `scripts/quickstart_wdnet_attack.py` |

## What Is Needed For Full Paper-Level Reproduction

A fresh clone can run the WDNet quickstart, but full paper-level reproduction is
not a single-command, weight-free run. Users must provide:

- the paper evaluation image set, arranged as an `ImageFolder`-compatible input
  directory;
- the released core weights archive from Google Drive, extracted into the
  repository root;
- third-party model code/checkpoints required by the wrappers in
  `inpainting/AWD_AGP/source_models/`;
- CUDA/PyTorch versions compatible with the original research environment;
- optional RPN refined masks, or the score records/checkpoint needed to generate
  them.

The current core weights archive covers the configured core profile checked by:

```bash
python scripts/check_weights.py --profile core
```

The RFR Places2 checkpoint is intentionally listed separately and is not part of
the current core archive. Add it through `--rfr_places2_checkpoint` or
`AWD_AGP_RFR_PLACES2_CKPT` if you need that exact backend.

## Verified Smoke Tests

The repository has been checked with the `ensemble` conda environment for:

```bash
python -m py_compile \
  inpainting/AWD_AGP/official_surrogate_generate_AE.py \
  inpainting/AWD_AGP/official_transfer_attack.py \
  inpainting/AWD_AGP/official_transfer_attack_for_wr.py \
  inpainting/AWD_AGP/official_evaluate.py \
  inpainting/AWD_AGP/official_evaluate_for_wr.py \
  inpainting/AWD_AGP/official_surrogate_generate_AE_mask_DE_without_noise.py \
  scripts/quickstart_wdnet_attack.py \
  scripts/check_weights.py \
  scripts/download_gdrive_weights.py
```

The WDNet quickstart was also run with a short two-step optimization and produced
all expected outputs, including `02_protected.png`, WDNet removal results, a
perturbation image, metrics, and a comparison grid.

## Recommended Reproduction Order

1. Install the environment and third-party projects required by the selected
   backends.
2. Download and extract the core weights archive, then run
   `python scripts/check_weights.py --profile core`.
3. Run the WDNet quickstart to confirm the local CUDA/PyTorch/image stack works.
4. Generate or provide RPN refined masks. To retrain Mask RPN, first generate
   score records with the superpixel-guided evolutionary search described in
   `docs/TRAIN_MASK_RPN.md`.
5. Run `official_surrogate_generate_AE.py` to create protected images.
6. Run the transfer scripts for inpainting-based and blind watermark-removal
   models.
7. Run the evaluation scripts on the generated transfer outputs.

## Current Limitations

- Exact paper table reproduction still depends on the same datasets/splits,
  random seeds, backend revisions, and third-party checkpoint versions used in
  the experiments.
- Some wrappers keep compatibility paths from the original research workspace;
  users should prefer the documented CLI arguments or environment variables in
  `WEIGHTS.md`.
- The full historical Mask RPN snapshot directory is not released in the core
  archive; the archive includes the selected authors-provided checkpoint used by
  the pipeline.
