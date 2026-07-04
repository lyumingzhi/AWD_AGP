# Google Drive Checkpoints

AWD-AGP keeps the full paper checkpoints outside git. Upload each checkpoint or
checkpoint directory archive to Google Drive, then fill the corresponding
`gdrive_url` or `gdrive_file_id` field in `configs/weights_manifest.json`.

The code resolves checkpoints in this order:

1. CLI argument, for example `--mat_checkpoint /path/to/file.pkl`.
2. Environment variable, for example `AWD_AGP_MAT_CKPT=/path/to/file.pkl`.
3. Local path under `--weights_dir` or `AWD_AGP_WEIGHTS_DIR`.
4. Local path under the repository `weights/` directory.
5. Legacy research-environment path, when one exists.

## Third-Party Model Code

The Google Drive package covers checkpoints and config files. Full-paper model
code such as MAT, FcF, CR-Fill, EdgeConnect, GMCNN, RFR, DBWE, and SLBR must
still be importable in Python, for example by installing the original projects
or adding their parent directory to `PYTHONPATH`. The top-level `inpainting`
package in this repository is a namespace package so sibling third-party
projects can be discovered when their parent directory is on `PYTHONPATH`.

## Recommended Google Drive Layout

Download or extract the Drive package into the repository root so the tree looks
like this:

```text
weights/
  RFR/checkpoint_places2.pth
  generative_inpainting/chkpt/places2.pth
  generative_inpainting/configs/config.yaml
  inpainting_gmcnn/pytorch/chkpts/places2_rect/places2.pth
  inpainting_gmcnn/pytorch/options/config.yaml
  edge_connect/checkpoints/places2/pipeline_config.yml
  MAT/pretrained/Places_512_FullData_real.pkl
  FcF_Inpainting/G.pt
  crfill/checkpoints/objrmv/objrmv.yaml
  crfill/checkpoints/places/places.yaml
  DBWEModel/27kpng_model_best.pth.tar
  SLBR/model_best.pth.tar
  superpixel_fcn/pretrain_ckpt/SpixelNet_bsd_ckpt.tar
  mask_RPN/checkpoints/
```

`WDNet_G.pkl` is bundled for the quickstart under
`third_party/inpainting/WDNet/WDNet_G.pkl`; you can still override it with
`--wdnet_checkpoint` or `AWD_AGP_WDNET_CKPT`.

## Fill the Manifest

Edit `configs/weights_manifest.json` after uploading checkpoints to Google
Drive:

```json
"mat_places_checkpoint": {
  "local_path": "weights/MAT/pretrained/Places_512_FullData_real.pkl",
  "cli_arg": "--mat_checkpoint",
  "env": "AWD_AGP_MAT_CKPT",
  "gdrive_url": "https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing",
  "gdrive_file_id": ""
}
```

For a single Google Drive file, either `gdrive_url` or `gdrive_file_id` is
enough. For checkpoint directories, upload an archive or document manual
extraction; the manifest marks such entries with `"type": "directory"`.

## Download

Install `gdown` once:

```bash
pip install gdown
```

Then run:

```bash
python scripts/download_gdrive_weights.py
```

To download only selected entries:

```bash
python scripts/download_gdrive_weights.py --only mat_places_checkpoint --only fcf_checkpoint
```

## Validate

Check all non-bundled full-paper entries:

```bash
python scripts/check_weights.py
```

Check only entries needed for one experiment:

```bash
python scripts/check_weights.py --require mat_places_checkpoint --require fcf_checkpoint
```

Check the bundled WDNet quickstart checkpoint:

```bash
python scripts/check_weights.py --include-bundled --require wdnet_checkpoint
```

## Use a Different Weights Directory

```bash
export AWD_AGP_WEIGHTS_DIR=/path/to/AWD_AGP_weights
python scripts/check_weights.py
```

or pass it to AWD-AGP scripts:

```bash
python -m inpainting.AWD_AGP.official_surrogate_generate_AE \
  --weights_dir /path/to/AWD_AGP_weights \
  --dataset places2 \
  --target_models Matnet \
  ...
```
