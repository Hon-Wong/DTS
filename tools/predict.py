"""Classify an image with exported DeiT-III DTS weights, without auxiliary heads."""

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recipes" / "imagenet_deit"))
from timm.models import create_model
import models_v2  # Registers the original architecture and checkpoint names.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Exported .safetensors file")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", required=True,
                        choices=["deit_tiny_patch16_LS", "deit_small_patch16_LS", "deit_base_patch16_LS"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-ratio", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 < args.crop_ratio <= 1 or args.input_size < 16 or args.topk < 1:
        parser.error("Require 0 < crop-ratio <= 1, input-size >= 16 and topk >= 1.")
    model = create_model(args.model, pretrained=False, img_size=args.input_size,
                         num_classes=args.num_classes, lfa=False)
    model.load_state_dict(load_file(str(args.weights)), strict=True)
    model.to(args.device).eval()
    transform = transforms.Compose([
        transforms.Resize(int(args.input_size / args.crop_ratio), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with Image.open(args.image) as image:
        batch = transform(image.convert("RGB")).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        values, indices = model(batch).softmax(-1)[0].topk(min(args.topk, args.num_classes))
    print(json.dumps([{"class_index": i, "probability": p}
                      for i, p in zip(indices.tolist(), values.tolist())], indent=2))


if __name__ == "__main__":
    main()
