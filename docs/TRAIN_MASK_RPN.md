# Train Your Own Mask RPN

AWD-AGP can use a Faster R-CNN based Mask RPN to propose watermark/mask
locations. The released weights package includes the authors-provided checkpoint
used by the demos/pipeline:

```text
weights/mask_RPN/checkpoints/fasterrcnn_02191254_0
```

This document explains how to train a replacement checkpoint on your own images
and score records.

## What This Model Is

`inpainting/AWD_AGP/mask_RPN/` is adapted from a simplified Faster R-CNN codebase.
The trained checkpoint is an AWD-AGP authors-provided watermark-location proposal
model. It is not one of the third-party inpainting or watermark-removal models.

The RPN training code consumes:

- an image directory
- a matching JSON score-record directory
- the Superpixel FCN checkpoint, used by the semantic scoring loss

## Training Data Layout

Prepare two directories with matching file stems:

```text
/path/to/rpn_images/
  000001.jpg
  000002.jpg

/path/to/rpn_score_records/
  000001.json
  000002.json
```

Each JSON file must contain `box_list` and `score_list` with the same length:

```json
{
  "box_list": [
    [y, x, relative_h, relative_w]
  ],
  "score_list": [
    1
  ]
}
```

The current loader interprets each box as:

```text
y_min = y
x_min = x
y_max = y + image_height * relative_h
x_max = x + image_width * relative_w
```

The loader expects at least 10 unique boxes per image. The highest scoring boxes
are read from the end of the list after reversing `box_list` and `score_list`, so
keep the records sorted consistently with the scoring procedure you use.

## Train

The training script uses local imports, so run it from the `mask_RPN` directory.
Set the score-record directory through `AWD_AGP_RPN_SCORES_DIR`, and pass the
image directory through `--voc-data-dir`:

```bash
cd /path/to/AWD_AGP/inpainting/AWD_AGP/mask_RPN

export AWD_AGP_RPN_SCORES_DIR=/path/to/rpn_score_records
export AWD_AGP_SUPERPIXEL_CKPT=/path/to/AWD_AGP/weights/superpixel_fcn/pretrain_ckpt/SpixelNet_bsd_ckpt.tar

python train.py train \
  --voc-data-dir=/path/to/rpn_images \
  --epoch=14 \
  --num-workers=4 \
  --test-num-workers=4
```

Checkpoints are saved under:

```text
inpainting/AWD_AGP/mask_RPN/checkpoints/fasterrcnn_<timestamp>
```

To resume from an existing checkpoint:

```bash
python train.py train \
  --voc-data-dir=/path/to/rpn_images \
  --load-path=/path/to/AWD_AGP/weights/mask_RPN/checkpoints/fasterrcnn_02191254_0
```

## Deploy a Trained Checkpoint

After training, copy the checkpoint you want to release/use to:

```text
weights/mask_RPN/checkpoints/fasterrcnn_02191254_0
```

or set:

```bash
export AWD_AGP_MASK_RPN_CKPT=/path/to/your/fasterrcnn_checkpoint
```

Validate the core weights package:

```bash
python scripts/check_weights.py --profile core
```

## Use RPN Masks in AWD-AGP

The main AWD-AGP attack scripts consume generated/refined mask files through
`--RPNRefineMask`. That directory should contain the mask artifacts expected by
`official_dataset.py`, including `rect.json` and `attacked_mask/` when using the
corresponding workflow.

Example:

```bash
python -m inpainting.AWD_AGP.official_surrogate_generate_AE \
  --dataset places2 \
  --target_models Matnet \
  --InputImg_dir /path/to/input_images \
  --output_dir surrogate_mat_awdagp \
  --get_logo \
  --attach_logo \
  --RPNRefineMask /path/to/generated_rpn_masks
```

## Notes

- Training is CUDA-oriented and was developed in the research environment.
- The authors-provided checkpoint in the weights package is sufficient for the
  released demo/pipeline path; retraining is optional.
- The full local training snapshot directory may be very large. Release only the
  selected checkpoint needed for reproduction, not every intermediate snapshot.
