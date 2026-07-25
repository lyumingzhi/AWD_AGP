#!/usr/bin/env python3
"""Minimal AWD-AGP-style quickstart using WDNet as a test remover.

This script is intentionally small: it demonstrates the repository workflow end to
end without requiring every model used in the paper. It creates a watermarked
image, optimizes a bounded adversarial perturbation against WDNet, and writes the
protected image plus WDNet removal outputs for visual inspection.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and test a protected watermarked image with WDNet.")
    parser.add_argument("--input", default="", help="optional clean input image path; if omitted, a synthetic demo image is generated")
    parser.add_argument("--output-dir", default="quickstart_outputs/wdnet_demo", help="directory for generated images")
    parser.add_argument("--external-inpainting-root", default=os.environ.get("AWD_AGP_EXTERNAL_INPAINTING_ROOT", ""), help="optional parent directory that contains third-party inpainting/WDNet; defaults to this repository third_party directory")
    parser.add_argument("--wdnet-checkpoint", default=os.environ.get("AWD_AGP_WDNET_CKPT", ""), help="path to WDNet_G.pkl")
    parser.add_argument("--size", type=int, default=256, help="square resize used by the quickstart")
    parser.add_argument("--steps", type=int, default=30, help="PGD/Adam optimization steps")
    parser.add_argument("--lr", type=float, default=0.01, help="optimizer learning rate")
    parser.add_argument("--budget", type=float, default=0.03, help="L-infinity perturbation budget in [0, 1]")
    parser.add_argument("--watermark-alpha", type=float, default=0.45, help="watermark opacity")
    parser.add_argument("--logo-text", default="AWD-AGP", help="text drawn as the demo watermark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="device to use")
    return parser.parse_args()


def configure_external_imports(args):
    candidate_roots = [REPO_ROOT / "third_party"]
    if args.external_inpainting_root:
        candidate_roots.append(Path(args.external_inpainting_root).expanduser().resolve())

    for root in candidate_roots:
        if root.exists() and str(root) not in sys.path:
            sys.path.append(str(root))

    checkpoint = Path(args.wdnet_checkpoint).expanduser() if args.wdnet_checkpoint else None
    if not checkpoint:
        candidates = [root / "inpainting" / "WDNet" / "WDNet_G.pkl" for root in candidate_roots]
        candidates.append(Path("inpainting/WDNet/WDNet_G.pkl"))
        for candidate in candidates:
            if candidate.exists():
                checkpoint = candidate
                break

    if not checkpoint or not checkpoint.exists():
        raise FileNotFoundError(
            "WDNet checkpoint not found. The default expected path is "
            f"{REPO_ROOT / 'third_party' / 'inpainting' / 'WDNet' / 'WDNet_G.pkl'}. "
            "Alternatively pass --wdnet-checkpoint /path/to/WDNet_G.pkl or set AWD_AGP_WDNET_CKPT."
        )

    try:
        from inpainting.AWD_AGP.source_models.WDnet import WDnet
    except Exception as exc:
        raise ImportError(
            "Could not import WDNet. Ensure this repository contains third_party/inpainting/WDNet "
            "or pass --external-inpainting-root /path/to/parent-that-contains-inpainting."
        ) from exc

    return WDnet, str(checkpoint)


def make_demo_image(size):
    x = np.linspace(0, 1, size, dtype="float32")
    y = np.linspace(0, 1, size, dtype="float32")
    xx, yy = np.meshgrid(x, y)
    image = np.stack([
        0.18 + 0.55 * xx,
        0.25 + 0.45 * yy,
        0.35 + 0.35 * (1 - xx) * (1 - yy),
    ], axis=2)

    canvas = Image.fromarray((np.clip(image, 0, 1) * 255).astype("uint8"), "RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, int(size * 0.66), size, size), fill=(48, 64, 74, 180))
    draw.ellipse((int(size * 0.08), int(size * 0.12), int(size * 0.38), int(size * 0.42)), fill=(238, 205, 109, 210))
    draw.polygon(
        [
            (int(size * 0.18), int(size * 0.72)),
            (int(size * 0.42), int(size * 0.44)),
            (int(size * 0.68), int(size * 0.72)),
        ],
        fill=(74, 122, 92, 230),
    )
    draw.polygon(
        [
            (int(size * 0.46), int(size * 0.72)),
            (int(size * 0.72), int(size * 0.38)),
            (int(size * 0.94), int(size * 0.72)),
        ],
        fill=(86, 105, 132, 230),
    )
    return canvas


def load_image(path, size, device):
    if path:
        image = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    else:
        image = make_demo_image(size)
    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def tensor_to_image(tensor):
    tensor = tensor.detach().clamp(0, 1).cpu()[0]
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def make_text_watermark(size, text, alpha, device):
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, size // 11))
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(8, size // 32)
    x = size - text_w - 2 * pad
    y = size - text_h - 2 * pad
    draw.rounded_rectangle((x - pad, y - pad, size - pad, size - pad), radius=4, fill=190)
    draw.text((x, y), text, fill=255, font=font)

    mask_np = np.asarray(mask).astype("float32") / 255.0
    mask_t = torch.from_numpy(mask_np).view(1, 1, size, size).to(device)
    logo = torch.ones(1, 3, size, size, device=device)
    alpha_mask = mask_t * alpha
    return logo, alpha_mask, mask_t


def run_wdnet(model, image, mask):
    output = model(image, mask)
    if isinstance(output, tuple):
        return output[0].clamp(0, 1)
    return output.clamp(0, 1)


def save_grid(paths, output_path):
    images = [Image.open(path).convert("RGB") for path in paths]
    w, h = images[0].size
    label_h = 26
    canvas = Image.new("RGB", (w * len(images), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(images):
        canvas.paste(image, (idx * w, label_h))
        draw.text((idx * w + 6, 6), Path(paths[idx]).stem, fill=(0, 0, 0))
    canvas.save(output_path)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu only with a CPU-compatible remover.")

    WDnet, checkpoint = configure_external_imports(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(args.input, args.size, args.device)
    logo, alpha_mask, logo_mask = make_text_watermark(args.size, args.logo_text, args.watermark_alpha, args.device)
    watermarked = (image * (1 - alpha_mask) + logo * alpha_mask).clamp(0, 1)

    model = WDnet(SimpleNamespace(wdnet_checkpoint=checkpoint)).to(args.device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        removed_watermarked = run_wdnet(model, watermarked, logo_mask)

    delta = torch.zeros_like(watermarked, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        protected = (watermarked + delta.clamp(-args.budget, args.budget)).clamp(0, 1)
        removed_protected = run_wdnet(model, protected, logo_mask)
        attack_loss = -F.mse_loss(removed_protected * logo_mask, image * logo_mask)
        visibility_loss = F.mse_loss(protected, watermarked)
        loss = attack_loss + 0.05 * visibility_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-args.budget, args.budget)
        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            print(f"step {step + 1:03d}/{args.steps}: loss={loss.item():.6f}")

    protected = (watermarked + delta.detach().clamp(-args.budget, args.budget)).clamp(0, 1)
    with torch.no_grad():
        removed_protected = run_wdnet(model, protected, logo_mask)

    masked_pixels = logo_mask.sum().clamp_min(1.0)
    metrics = {
        "wdnet_mse_removed_watermarked_vs_input_on_logo": float((((removed_watermarked - image) ** 2) * logo_mask).sum().detach().cpu() / masked_pixels),
        "wdnet_mse_removed_protected_vs_input_on_logo": float((((removed_protected - image) ** 2) * logo_mask).sum().detach().cpu() / masked_pixels),
        "mean_abs_perturbation": float((protected - watermarked).abs().mean().detach().cpu()),
        "max_abs_perturbation": float((protected - watermarked).abs().max().detach().cpu()),
        "budget": args.budget,
        "steps": args.steps,
    }

    files = {
        "input": out_dir / "00_input.png",
        "watermarked": out_dir / "01_watermarked.png",
        "protected": out_dir / "02_protected.png",
        "removed_watermarked": out_dir / "03_wdnet_removed_watermarked.png",
        "removed_protected": out_dir / "04_wdnet_removed_protected.png",
        "perturbation": out_dir / "05_perturbation_x10.png",
        "metrics": out_dir / "metrics.json",
    }
    tensor_to_image(image).save(files["input"])
    tensor_to_image(watermarked).save(files["watermarked"])
    tensor_to_image(protected).save(files["protected"])
    tensor_to_image(removed_watermarked).save(files["removed_watermarked"])
    tensor_to_image(removed_protected).save(files["removed_protected"])
    tensor_to_image((delta.detach() / (2 * args.budget) + 0.5).clamp(0, 1)).save(files["perturbation"])
    files["metrics"].write_text(json.dumps(metrics, indent=2) + "\n")
    save_grid([files["input"], files["watermarked"], files["protected"], files["removed_watermarked"], files["removed_protected"]], out_dir / "comparison.png")

    print("\nQuickstart metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nGenerated quickstart outputs:")
    for path in files.values():
        print(f"  {path}")
    print(f"  {out_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
