# DTS research implementation; release maintenance by DTS contributors (2026).
# See docs/provenance.md for source and changes.
"""
Deep trajectory probing for pretrained ViTs and deit_ours checkpoints.
"""

import argparse
import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

matplotlib.use("Agg")


DATASET_CONFIGS = {
    "cifar10": {
        "classes": 10,
        "size": 32,
        "folder": "",
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "classes": 100,
        "size": 32,
        "folder": "",
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "tiny_imagenet": {
        "classes": 200,
        "size": 64,
        "folder": "tiny-imagenet-200",
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2302, 0.2265, 0.2262),
    },
    "food101": {
        "classes": 101,
        "size": 224,
        "folder": "food-101",
        "mean": (0.5493, 0.4450, 0.3435),
        "std": (0.2730, 0.2759, 0.2799),
    },
    "imagenet": {
        "classes": 1000,
        "size": 224,
        "folder": "imagenet",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "inat19": {
        "classes": 1010,
        "size": 224,
        "folder": "inat2019",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(0)
    return 0


def is_main():
    return dist.get_rank() == 0


def append_csv(path, row):
    if not is_main():
        return
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path, payload):
    if not is_main():
        return
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def get_loaders(args):
    cfg = DATASET_CONFIGS[args.dataset]
    size = args.input_size or cfg["size"]
    folder = cfg["folder"]
    dataset_root = os.path.join(args.data_dir, folder) if folder else args.data_dir

    train_transforms = [
        transforms.Resize(size),
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ]
    val_transforms = [
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ]

    t_train = transforms.Compose(train_transforms)
    t_val = transforms.Compose(val_transforms)

    if args.dataset in {"cifar10", "cifar100"}:
        ds_cls = torchvision.datasets.CIFAR10 if args.dataset == "cifar10" else torchvision.datasets.CIFAR100
        train_ds = ds_cls(dataset_root, train=True, download=True, transform=t_train)
        val_ds = ds_cls(dataset_root, train=False, download=True, transform=t_val)
    else:
        train_path = os.path.join(dataset_root, "train")
        val_path = os.path.join(dataset_root, "val")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        train_ds = torchvision.datasets.ImageFolder(train_path, transform=t_train)
        val_ds = torchvision.datasets.ImageFolder(val_path, transform=t_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_ds),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=DistributedSampler(val_ds, shuffle=False),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, cfg


def register_deit_ours_models(repo_root):
    repo_root = str(Path(repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    for module_name in ["models", "models_v2"]:
        importlib.import_module(module_name)


def load_checkpoint_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    clean = {}
    for key, value in state_dict.items():
        clean[key[7:] if key.startswith("module.") else key] = value
    return clean


def build_backbone(args, cfg):
    input_size = args.input_size or cfg["size"]

    if args.model in {"torchvision_resnet18", "torchvision_resnet50"}:
        if args.checkpoint:
            if args.model == "torchvision_resnet18":
                backbone = torchvision.models.resnet18(weights=None)
            else:
                backbone = torchvision.models.resnet50(weights=None)
            state_dict = load_checkpoint_state(args.checkpoint)
            msg = backbone.load_state_dict(state_dict, strict=False)
            if is_main():
                print("Loaded checkpoint:", args.checkpoint)
                print("Missing keys:", msg.missing_keys)
                print("Unexpected keys:", msg.unexpected_keys)
            return backbone

        if args.model == "torchvision_resnet18":
            return torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            )
        return torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )

    if args.checkpoint and args.checkpoint_format == "deit_ours":
        register_deit_ours_models(args.deit_ours_root)
        backbone = timm.create_model(
            args.model,
            pretrained=False,
            num_classes=cfg["classes"],
            img_size=input_size,
            lfa=False,
            head_type="shared",
        )
        state_dict = load_checkpoint_state(args.checkpoint)
        msg = backbone.load_state_dict(state_dict, strict=False)
        if is_main():
            print("Loaded checkpoint:", args.checkpoint)
            print("Missing keys:", msg.missing_keys)
            print("Unexpected keys:", msg.unexpected_keys)
        return backbone

    if args.checkpoint:
        backbone = timm.create_model(args.model, pretrained=False, num_classes=cfg["classes"])
        state_dict = load_checkpoint_state(args.checkpoint)
        msg = backbone.load_state_dict(state_dict, strict=False)
        if is_main():
            print("Loaded checkpoint:", args.checkpoint)
            print("Missing keys:", msg.missing_keys)
            print("Unexpected keys:", msg.unexpected_keys)
        return backbone

    return timm.create_model(args.model, pretrained=True)


class HookedViTProber(nn.Module):
    def __init__(self, backbone, num_classes, fix_head=False):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim
        self.blocks = self.backbone.blocks if hasattr(self.backbone, "blocks") else self.backbone.layers
        self.num_layers = len(self.blocks)
        self.norm_layer = getattr(self.backbone, "norm", nn.Identity())
        self.use_backbone_head = (
            hasattr(self.backbone, "head")
            and isinstance(self.backbone.head, nn.Module)
            and hasattr(self.backbone.head, "out_features")
            and self.backbone.head.out_features == num_classes
        )
        self.fix_head = fix_head

        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.use_backbone_head and not self.fix_head:
            for param in self.backbone.head.parameters():
                param.requires_grad = True

        probe_count = self.num_layers if self.use_backbone_head else self.num_layers + 1
        self.probes = nn.ModuleList(
            nn.Linear(self.embed_dim, num_classes) for _ in range(probe_count)
        )

    def _prepare_tokens(self, x):
        x = self.backbone.patch_embed(x)
        if hasattr(self.backbone, "_pos_embed"):
            x = self.backbone._pos_embed(x)
        else:
            batch_size = x.shape[0]
            prefix_tokens = []
            cls_token = getattr(self.backbone, "cls_token", None)
            dist_token = getattr(self.backbone, "dist_token", None)
            if cls_token is not None:
                prefix_tokens.append(cls_token.expand(batch_size, -1, -1))
            if dist_token is not None:
                prefix_tokens.append(dist_token.expand(batch_size, -1, -1))

            pos_embed = getattr(self.backbone, "pos_embed", None)
            if pos_embed is not None:
                num_prefix = len(prefix_tokens)
                if pos_embed.shape[1] == x.shape[1]:
                    x = x + pos_embed
                elif pos_embed.shape[1] == x.shape[1] + num_prefix:
                    if num_prefix:
                        prefix_pos = pos_embed[:, :num_prefix, :]
                        prefix_tokens = [
                            token + prefix_pos[:, idx : idx + 1, :]
                            for idx, token in enumerate(prefix_tokens)
                        ]
                    x = x + pos_embed[:, num_prefix:, :]
                else:
                    x = x + pos_embed[:, -x.shape[1] :, :]

            if prefix_tokens:
                x = torch.cat(prefix_tokens + [x], dim=1)
            if hasattr(self.backbone, "pos_drop"):
                x = self.backbone.pos_drop(x)

        if hasattr(self.backbone, "patch_drop"):
            x = self.backbone.patch_drop(x)
        if hasattr(self.backbone, "norm_pre"):
            x = self.backbone.norm_pre(x)
        return x

    def trainable_parameters(self):
        params = list(self.probes.parameters())
        if self.use_backbone_head and not self.fix_head:
            params.extend(self.backbone.head.parameters())
        return params

    def forward(self, x):
        x = self._prepare_tokens(x)

        states = [x[:, 0, :]]
        for block in self.blocks:
            x = block(x)
            states.append(x[:, 0, :])

        logits = []
        probe_states = states[:-1] if self.use_backbone_head else states
        for idx, state in enumerate(probe_states):
            logits.append(self.probes[idx](self.norm_layer(state)))

        final_state = self.norm_layer(states[-1])
        if self.use_backbone_head:
            logits.append(self.backbone.head(final_state))
        else:
            logits[-1] = self.probes[-1](final_state)

        return torch.stack(logits, dim=0)


class HookedResNetProber(nn.Module):
    def __init__(self, backbone, num_classes, fix_head=False):
        super().__init__()
        self.backbone = backbone
        self.num_layers = 4
        self.fix_head = fix_head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.use_backbone_head = (
            hasattr(self.backbone, "fc")
            and isinstance(self.backbone.fc, nn.Module)
            and hasattr(self.backbone.fc, "out_features")
            and self.backbone.fc.out_features == num_classes
        )

        stem_dim = self.backbone.conv1.out_channels
        stage_dims = [self._stage_out_dim(self.backbone.layer1)]
        stage_dims.append(self._stage_out_dim(self.backbone.layer2))
        stage_dims.append(self._stage_out_dim(self.backbone.layer3))
        stage_dims.append(self._stage_out_dim(self.backbone.layer4))
        probe_dims = [stem_dim] + stage_dims[:-1] if self.use_backbone_head else [stem_dim] + stage_dims

        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.use_backbone_head and not self.fix_head:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

        self.probes = nn.ModuleList(nn.Linear(dim, num_classes) for dim in probe_dims)

    @staticmethod
    def _stage_out_dim(stage):
        block = stage[-1]
        if hasattr(block, "conv3"):
            return block.conv3.out_channels
        return block.conv2.out_channels

    def _act(self, x):
        if hasattr(self.backbone, "act1"):
            return self.backbone.act1(x)
        return self.backbone.relu(x)

    def _pool(self, x):
        return self.avgpool(x).flatten(1)

    def trainable_parameters(self):
        params = list(self.probes.parameters())
        if self.use_backbone_head and not self.fix_head:
            params.extend(self.backbone.fc.parameters())
        return params

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self._act(x)
        if hasattr(self.backbone, "maxpool") and not isinstance(self.backbone.maxpool, nn.Identity):
            x = self.backbone.maxpool(x)

        stem = self._pool(x)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        states = [stem, self._pool(x1), self._pool(x2), self._pool(x3), self._pool(x4)]
        probe_states = states[:-1] if self.use_backbone_head else states

        logits = [probe(state) for probe, state in zip(self.probes, probe_states)]
        if self.use_backbone_head:
            logits.append(self.backbone.fc(states[-1]))
        else:
            logits[-1] = self.probes[-1](states[-1])
        return torch.stack(logits, dim=0)


def normalized_exponential_curve(t, beta):
    if abs(beta) < 1e-8:
        return t
    numerator = np.exp(beta * t) - 1.0
    denominator = np.exp(beta) - 1.0
    return numerator / denominator


def fit_normalized_exponential(evidence):
    evidence = np.asarray(evidence, dtype=np.float64)
    t = np.linspace(0.0, 1.0, len(evidence))
    y0, y1 = evidence[0], evidence[-1]
    best = {"beta": 0.0, "r2": -np.inf, "pred": np.linspace(y0, y1, len(evidence))}
    ss_tot = float(((evidence - evidence.mean()) ** 2).sum())

    for beta in np.linspace(0.1, 12.0, 600):
        basis = normalized_exponential_curve(t, beta)
        pred = y0 + (y1 - y0) * basis
        ss_res = float(((evidence - pred) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        if r2 > best["r2"]:
            best = {"beta": float(beta), "r2": float(r2), "pred": pred}
    return best


def save_evidence_fit(output_dir, epoch, gaps):
    if not is_main():
        return None

    fit = fit_normalized_exponential(gaps)
    layers = np.arange(len(gaps))
    pdf_path = Path(output_dir) / f"evidence_fit_ep{epoch:03d}.pdf"
    png_path = Path(output_dir) / f"evidence_fit_ep{epoch:03d}.png"
    json_path = Path(output_dir) / f"evidence_fit_ep{epoch:03d}.json"

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(layers, gaps, marker="o", linewidth=2.0, label="Measured evidence")
    ax.plot(layers, fit["pred"], linestyle="--", linewidth=2.0, label=f"Normalized exp fit, R^2={fit['r2']:.4f}")
    ax.set_xlabel("Depth index")
    ax.set_ylabel("Semantic evidence")
    ax.set_title(f"Semantic Evidence vs Depth (epoch {epoch})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    payload = {
        "epoch": epoch,
        "beta": fit["beta"],
        "r2": fit["r2"],
        "gaps": list(map(float, gaps)),
        "pred": list(map(float, fit["pred"])),
        "pdf_path": str(pdf_path),
        "png_path": str(png_path),
    }
    save_json(json_path, payload)
    return payload


@torch.no_grad()
def evaluate(model, loader, device, num_classes, epoch, output_dir):
    model.eval()
    L = model.module.num_layers
    acc_sum = torch.zeros(L + 1, dtype=torch.float64, device=device)
    gap_sum = torch.zeros(L + 1, dtype=torch.float64, device=device)
    pav_sum = torch.zeros(L, dtype=torch.float64, device=device)
    total_samples = torch.tensor(0.0, dtype=torch.float64, device=device)

    dump_logits = []
    dump_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)
        total_samples += batch_size

        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        p_gt = probs.gather(-1, targets.view(1, -1, 1).expand(L + 1, -1, -1)).squeeze(-1)

        acc_sum += logits.argmax(-1).eq(targets.view(1, -1)).float().sum(1)
        for idx in range(L + 1):
            layer_logits = logits[idx]
            gy = layer_logits.gather(1, targets.view(-1, 1)).squeeze()
            mask = torch.ones_like(layer_logits).scatter_(1, targets.view(-1, 1), 0.0)
            go = (layer_logits * mask).sum(1) / (num_classes - 1)
            gap_sum[idx] += (gy - go).sum()
            if idx < L:
                pav_sum[idx] += (p_gt[idx + 1] < p_gt[idx]).float().sum()

        dump_logits.append(logits.cpu())
        dump_targets.append(targets.cpu())

    for tensor in [acc_sum, gap_sum, pav_sum, total_samples]:
        dist.all_reduce(tensor)

    local_logits = torch.cat(dump_logits, dim=1)
    local_targets = torch.cat(dump_targets, dim=0)

    final_logits = None
    final_targets = None
    if dist.get_world_size() > 1:
        gather_logits = local_logits.to(device, non_blocking=True).contiguous()
        gather_targets = local_targets.to(device, non_blocking=True).contiguous()

        if is_main():
            gathered_logits = [torch.empty_like(gather_logits) for _ in range(dist.get_world_size())]
            gathered_targets = [torch.empty_like(gather_targets) for _ in range(dist.get_world_size())]
        else:
            gathered_logits = None
            gathered_targets = None

        dist.gather(gather_logits, gather_list=gathered_logits, dst=0)
        dist.gather(gather_targets, gather_list=gathered_targets, dst=0)

        if is_main():
            final_logits = torch.cat([tensor.cpu() for tensor in gathered_logits], dim=1)
            final_targets = torch.cat([tensor.cpu() for tensor in gathered_targets], dim=0)
    else:
        final_logits = local_logits
        final_targets = local_targets

    if is_main():
        num_samples = total_samples.item()
        acc = (acc_sum / num_samples).cpu().numpy()
        gaps = (gap_sum / num_samples).cpu().numpy()
        pav = (pav_sum / num_samples).cpu().numpy()
        fit_payload = save_evidence_fit(output_dir, epoch, gaps)

        row = {"epoch": epoch, "final_acc": acc[-1]}
        if fit_payload is not None:
            row["evidence_fit_r2"] = fit_payload["r2"]
            row["evidence_fit_beta"] = fit_payload["beta"]
        for idx in range(L + 1):
            row[f"l{idx}_acc"] = acc[idx]
            row[f"l{idx}_gap"] = gaps[idx]
            if idx < L:
                row[f"s{idx}_pav"] = pav[idx]
        append_csv(os.path.join(output_dir, "probe_stats.csv"), row)

        torch.save(
            {"logits": final_logits, "targets": final_targets, "accs": acc, "gaps": gaps},
            os.path.join(output_dir, f"trajectories_ep{epoch:03d}.pth"),
        )
        print(f"[Eval] epoch={epoch} final_acc={acc[-1]:.4f} fit_r2={row.get('evidence_fit_r2', float('nan')):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoint-format", type=str, default="timm", choices=["timm", "deit_ours"])
    parser.add_argument("--deit-ours-root", type=str, default=str(Path(__file__).resolve().parents[1] / "imagenet_deit"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--fix_head", action="store_true")
    parser.add_argument("--input_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./probing_results")
    parser.add_argument("--probe-output-dir", type=str, default="")
    args = parser.parse_args()

    rank = setup_ddp()
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    train_loader, val_loader, cfg = get_loaders(args)

    base_output_dir = Path(args.probe_output_dir or args.output_dir)
    if args.checkpoint:
        run_name = Path(args.checkpoint).stem
        output_dir = base_output_dir / run_name
    else:
        output_dir = base_output_dir
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)

    backbone = build_backbone(args, cfg).to(device)
    if hasattr(backbone, "layer1") and hasattr(backbone, "layer4") and hasattr(backbone, "conv1"):
        probe_model = HookedResNetProber(backbone, cfg["classes"], fix_head=args.fix_head).to(device)
    else:
        probe_model = HookedViTProber(backbone, cfg["classes"], fix_head=args.fix_head).to(device)
    train_head = probe_model.use_backbone_head and not args.fix_head
    optimizer = optim.AdamW(probe_model.trainable_parameters(), lr=args.lr, weight_decay=0.01)
    model = DDP(probe_model, device_ids=[rank], find_unused_parameters=train_head)
    scaler = torch.amp.GradScaler("cuda")

    evaluate(model, val_loader, device, cfg["classes"], 0, str(output_dir))

    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        iterator = tqdm(train_loader, desc=f"Probing {epoch}", disable=not is_main())
        optimizer.zero_grad()

        for step, (images, targets) in enumerate(iterator, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = sum(F.cross_entropy(logits[idx], targets) for idx in range(logits.shape[0])) / logits.shape[0]
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()
            if step % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        evaluate(model, val_loader, device, cfg["classes"], epoch + 1, str(output_dir))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
