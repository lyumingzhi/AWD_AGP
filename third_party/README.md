# Third-Party Components

This directory contains small third-party components needed by the quickstart.

## WDNet

`third_party/inpainting/WDNet` is a vendored copy of the WDNet test remover from:

https://github.com/MRUIL/WDNet.git

The quickstart uses `WDNet_G.pkl` to test whether a protected watermarked image
resists blind watermark removal. Larger model checkpoints used by the full paper
experiments are still documented in `WEIGHTS.md` and should be distributed via
GitHub Releases, Git LFS, Hugging Face, or institutional storage.

## License note

Before publishing this repository publicly, confirm that redistribution of the
vendored WDNet code, PDF, and checkpoint is permitted by the original authors or
replace this bundled copy with a download script/submodule plus external weight
release.
