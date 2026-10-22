# DTS research implementation; release maintenance by DTS contributors (2026).
# See docs/provenance.md for source and changes.
"""
Deep Trajectory Supervision (DTS) - Research Framework V11
=============================================================
Updates:
- Added support for Stanford Cars, iNat-19, iNat-18.
- Generalized ImageFolder loading logic.
- Retained all scientific metrics (Curl, PAV, SNR) and Strong Aug.
"""

import argparse
import csv
import datetime
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# =============================================================================
# 0. CONFIGURATIONS & UTILS
# =============================================================================

DATASET_CONFIGS = {
    # Standard CIFAR
    'cifar10': {'classes': 10, 'size': 32, 'patch': 4, 'folder': '', 'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
    'cifar100': {'classes': 100, 'size': 32, 'patch': 4, 'folder': '', 'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761)},

    # Small Scale / Fine-Grained
    'tiny_imagenet': {'classes': 200, 'size': 64, 'patch': 8, 'folder': 'tiny-imagenet-200', 'mean': (0.4802, 0.4481, 0.3975), 'std': (0.2302, 0.2265, 0.2262)},
    'cub200': {'classes': 200, 'size': 224, 'patch': 16, 'folder': 'CUB_200_2011', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
    'flowers102': {'classes': 102, 'size': 224, 'patch': 16, 'folder': 'flowers-102', 'mean': (0.436, 0.377, 0.287), 'std': (0.300, 0.245, 0.264)},
    'cars': {'classes': 196, 'size': 224, 'patch': 16, 'folder': 'stanford-cars', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
    'food101': {'classes': 101, 'size': 224, 'patch': 16, 'folder': 'food-101', 'mean': (0.5493, 0.4450, 0.3435), 'std': (0.2730, 0.2759, 0.2799)},

    # Large Scale / Long Tail
    'imagenet': {'classes': 1000, 'size': 224, 'patch': 16, 'folder': 'imagenet', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
    'inat19': {'classes': 1010, 'size': 224, 'patch': 16, 'folder': 'inat2019', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
    'inat18': {'classes': 8142, 'size': 224, 'patch': 16, 'folder': 'inat2018', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
}

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    else:
        os.environ["RANK"], os.environ["LOCAL_RANK"] = "0", "0"
        os.environ["WORLD_SIZE"], os.environ["MASTER_ADDR"] = "1", "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)
        return 0

def is_main(): return dist.get_rank() == 0

class StatefulLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.start_time = time.time()
        if is_main():
            os.makedirs(output_dir, exist_ok=True)
            self.log_file = open(os.path.join(output_dir, "log.txt"), "a")

    def info(self, msg):
        if is_main():
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted = f"[{ts}] {msg}"
            print(formatted)
            self.log_file.write(formatted + "\n")
            self.log_file.flush()

    def eta(self, step, total_steps):
        if step == 0: return "N/A"
        elapsed = time.time() - self.start_time
        rem = (elapsed / step) * (total_steps - step)
        return str(datetime.timedelta(seconds=int(rem)))

def append_csv(path, row):
    if not is_main(): return
    ex = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not ex: w.writeheader()
        w.writerow(row)


def read_last_layer_stats_epoch(output_dir):
    stats_path = Path(output_dir) / "layer_stats.csv"
    if not stats_path.exists():
        return None
    with stats_path.open() as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            last_row = row
    if last_row is None:
        return None
    return int(float(last_row["epoch"]))


def find_latest_model_checkpoint(output_dir):
    output_dir = Path(output_dir)
    candidates = []
    for path in output_dir.glob("ckpt_ep*.pth"):
        stem = path.stem
        try:
            epoch = int(stem.replace("ckpt_ep", ""))
        except ValueError:
            continue
        candidates.append((epoch, path))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])


def read_best_acc_from_stats(output_dir):
    stats_path = Path(output_dir) / "layer_stats.csv"
    if not stats_path.exists():
        return 0.0
    best = 0.0
    with stats_path.open() as f:
        for row in csv.DictReader(f):
            best = max(best, float(row["acc"]))
    return best


def build_scheduler(optimizer, total_epochs, start_epoch=0):
    if start_epoch > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        last_epoch=start_epoch - 1,
    )


def maybe_resume_training(model, optimizer, scaler, output_dir, total_epochs, steps_per_epoch, logger):
    output_dir = Path(output_dir)
    latest_state_path = output_dir / "latest_train_state.pth"
    if latest_state_path.exists():
        payload = torch.load(latest_state_path, map_location="cpu", weights_only=False)
        model.module.load_state_dict(payload["model"], strict=False)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler = build_scheduler(optimizer, total_epochs, start_epoch=0)
        scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload["epoch"])
        best_acc = float(payload.get("best_acc", 0.0))
        global_step = int(payload.get("global_step", start_epoch * steps_per_epoch))
        logger.info(f"Resumed full training state from {latest_state_path} at epoch {start_epoch}.")
        return start_epoch, best_acc, global_step, scheduler

    latest_epoch, latest_ckpt = find_latest_model_checkpoint(output_dir)
    if latest_ckpt is not None:
        payload = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.module.load_state_dict(state_dict, strict=False)
        start_epoch = int(latest_epoch)
        best_acc = read_best_acc_from_stats(output_dir)
        global_step = start_epoch * steps_per_epoch
        scheduler = build_scheduler(optimizer, total_epochs, start_epoch=start_epoch)
        logger.info(f"Resumed model weights from {latest_ckpt} at epoch {start_epoch}.")
        return start_epoch, best_acc, global_step, scheduler

    return 0, 0.0, 0, build_scheduler(optimizer, total_epochs, start_epoch=0)


def save_training_state(output_dir, model, optimizer, scheduler, scaler, epoch, best_acc, global_step):
    if not is_main():
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "global_step": global_step,
    }
    torch.save(payload, output_dir / "latest_train_state.pth")

# =============================================================================
# 1. MODEL
# =============================================================================

class TrajectoryTransformer(nn.Module):
    def __init__(self, strategy, **kwargs):
        super().__init__()
        self.strategy = strategy
        self.num_layers = kwargs['num_layers']
        self.embed_dim = kwargs['embed_dim']
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=kwargs['patch_size'], stride=kwargs['patch_size'])
        num_patches = (kwargs['img_size'] // kwargs['patch_size']) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, self.embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=kwargs['num_heads'],
                                       dim_feedforward=kwargs['mlp_dim'], dropout=0.1,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(self.num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)
        if self.strategy == 'aligned':
            self.aligned_classifier = nn.Linear(self.embed_dim, kwargs['num_classes'])
            self.classifiers = None
        else:
            self.aligned_classifier = None
            self.classifiers = nn.ModuleList([nn.Linear(self.embed_dim, kwargs['num_classes']) for _ in range(self.num_layers + 1)])
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02); nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) + self.pos_embed
        states = [x[:, 0]]
        for block in self.blocks:
            x = block(x); states.append(x[:, 0])
        all_logits = []
        if self.strategy == 'aligned':
            for s in states:
                all_logits.append(self.aligned_classifier(self.norm(s)))
        else:
            for i, s in enumerate(states):
                feat = s.detach() if (self.strategy == 'baseline' and i < self.num_layers) else s
                all_logits.append(self.classifiers[i](self.norm(feat)))
        return torch.stack(all_logits, dim=0), torch.stack(states, dim=0)


class TrajectoryResNet(nn.Module):
    def __init__(self, strategy, backbone, num_classes, image_size, granularity='stage'):
        super().__init__()
        self.strategy = strategy
        self.backbone = backbone
        self.granularity = granularity

        if backbone == 'resnet18':
            base = torchvision.models.resnet18(weights=None, num_classes=num_classes)
            stage_output_dims = [64, 128, 256, 512]
            self.state_dim = 512
        elif backbone == 'resnet50':
            base = torchvision.models.resnet50(weights=None, num_classes=num_classes)
            stage_output_dims = [256, 512, 1024, 2048]
            self.state_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Keep the standard ImageNet stem for 224x224 datasets and only switch
        # to the CIFAR-style stem for small-image regimes.
        if image_size <= 64:
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.stage_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        if self.granularity == 'stage':
            state_dims = [64, *stage_output_dims]
        elif self.granularity == 'block':
            state_dims = [64]
            for stage_dim, layer in zip(stage_output_dims, self.stage_layers):
                state_dims.extend([stage_dim] * len(layer))
        else:
            raise ValueError(f"Unsupported ResNet granularity: {self.granularity}")
        self.num_layers = len(state_dims)

        self.state_projs = nn.ModuleList([
            nn.Identity() if dim == self.state_dim else nn.Linear(dim, self.state_dim)
            for dim in state_dims
        ])
        self.final_state_proj = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.state_dim),
            nn.GELU(),
        )
        self.state_norm = nn.LayerNorm(self.state_dim)
        if self.strategy == 'aligned':
            self.aligned_classifier = nn.Linear(self.state_dim, num_classes)
            self.classifiers = None
        else:
            self.aligned_classifier = None
            self.classifiers = nn.ModuleList([
                nn.Linear(self.state_dim, num_classes) for _ in range(self.num_layers + 1)
            ])

    def _pool(self, x):
        return self.avgpool(x).flatten(1)

    def forward(self, x):
        x = self.stem(x)
        raw_states = [self._pool(x)]

        if self.granularity == 'stage':
            for layer in self.stage_layers:
                x = layer(x)
                raw_states.append(self._pool(x))
        else:
            for layer in self.stage_layers:
                for block in layer:
                    x = block(x)
                    raw_states.append(self._pool(x))

        states = [proj(state) for proj, state in zip(self.state_projs, raw_states)]
        states.append(self.final_state_proj(states[-1]))

        all_logits = []
        if self.strategy == 'aligned':
            for state in states:
                all_logits.append(self.aligned_classifier(self.state_norm(state)))
        else:
            for i, state in enumerate(states):
                feat = state.detach() if (self.strategy == 'baseline' and i < self.num_layers) else state
                all_logits.append(self.classifiers[i](self.state_norm(feat)))
        return torch.stack(all_logits, dim=0), torch.stack(states, dim=0)


class TrajectoryConvNeXt(nn.Module):
    def __init__(self, strategy, backbone, num_classes, image_size):
        super().__init__()
        self.strategy = strategy
        self.backbone = backbone
        self.num_layers = 5

        convnext_specs = {
            'convnext_tiny': {
                'factory': torchvision.models.convnext_tiny,
                'stem_dim': 96,
                'stage_dims': [96, 96, 192, 384, 768],
                'state_dim': 768,
            },
            'convnext_base': {
                'factory': torchvision.models.convnext_base,
                'stem_dim': 128,
                'stage_dims': [128, 128, 256, 512, 1024],
                'state_dim': 1024,
            },
        }
        if backbone not in convnext_specs:
            raise ValueError(f"Unsupported backbone: {backbone}")

        spec = convnext_specs[backbone]
        base = spec['factory'](weights=None, num_classes=num_classes)

        # Keep more spatial support for small-image regimes such as CIFAR-100.
        if image_size <= 64:
            base.features[0][0] = nn.Conv2d(
                3, spec['stem_dim'], kernel_size=3, stride=1, padding=1, bias=True
            )

        self.stem = base.features[0]
        self.stage1 = base.features[1]
        self.downsample2 = base.features[2]
        self.stage2 = base.features[3]
        self.downsample3 = base.features[4]
        self.stage3 = base.features[5]
        self.downsample4 = base.features[6]
        self.stage4 = base.features[7]
        self.head_norm = base.classifier[0]
        self.avgpool = base.avgpool

        stage_dims = spec['stage_dims']
        self.state_dim = spec['state_dim']
        self.state_projs = nn.ModuleList([
            nn.Identity() if dim == self.state_dim else nn.Linear(dim, self.state_dim)
            for dim in stage_dims
        ])
        self.final_state_proj = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.state_dim),
            nn.GELU(),
        )
        self.state_norm = nn.LayerNorm(self.state_dim)
        if self.strategy == 'aligned':
            self.aligned_classifier = nn.Linear(self.state_dim, num_classes)
            self.classifiers = None
        else:
            self.aligned_classifier = None
            self.classifiers = nn.ModuleList([
                nn.Linear(self.state_dim, num_classes) for _ in range(self.num_layers + 1)
            ])

    def _pool(self, x):
        return self.avgpool(x).flatten(1)

    def forward(self, x):
        x = self.stem(x)
        stem_state = self._pool(x)

        x = self.stage1(x)
        x1 = self._pool(x)

        x = self.downsample2(x)
        x = self.stage2(x)
        x2 = self._pool(x)

        x = self.downsample3(x)
        x = self.stage3(x)
        x3 = self._pool(x)

        x = self.downsample4(x)
        x = self.stage4(x)
        x4 = self._pool(self.head_norm(x))

        raw_states = [stem_state, x1, x2, x3, x4]
        states = [proj(state) for proj, state in zip(self.state_projs, raw_states)]
        states.append(self.final_state_proj(states[-1]))

        all_logits = []
        if self.strategy == 'aligned':
            for state in states:
                all_logits.append(self.aligned_classifier(self.state_norm(state)))
        else:
            for i, state in enumerate(states):
                feat = state.detach() if (self.strategy == 'baseline' and i < self.num_layers) else state
                all_logits.append(self.classifiers[i](self.state_norm(feat)))
        return torch.stack(all_logits, dim=0), torch.stack(states, dim=0)


def build_model(args, cfg):
    if args.backbone == 'vit':
        s_map = {'tiny': (192, 3, 768), 'small': (384, 6, 1536), 'base': (768, 12, 3072)}
        emb, n_h, mlp = s_map[args.model_scale]
        return TrajectoryTransformer(
            args.strategy,
            img_size=cfg['size'],
            patch_size=cfg['patch'],
            num_classes=cfg['classes'],
            embed_dim=emb,
            num_layers=12,
            num_heads=n_h,
            mlp_dim=mlp,
        )
    if args.backbone in ['resnet18', 'resnet50']:
        return TrajectoryResNet(
            args.strategy,
            args.backbone,
            cfg['classes'],
            cfg['size'],
            granularity=args.resnet_granularity,
        )
    return TrajectoryConvNeXt(args.strategy, args.backbone, cfg['classes'], cfg['size'])

def get_loaders(args):
    conf = DATASET_CONFIGS[args.dataset]
    folder = conf['folder']
    dataset_root = os.path.join(args.data_dir, folder) if folder else args.data_dir

    # --- Data Augmentation ---
    aug_list = [
        transforms.Resize(conf['size']),
        transforms.RandomResizedCrop(conf['size'], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip()
    ]
    if args.strong_aug:
        aug_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    aug_list.append(transforms.ToTensor())
    aug_list.append(transforms.Normalize(conf['mean'], conf['std']))

    t_tr = transforms.Compose(aug_list)
    t_va = transforms.Compose([
        transforms.Resize(conf['size']),
        transforms.CenterCrop(conf['size']),
        transforms.ToTensor(),
        transforms.Normalize(conf['mean'], conf['std'])
    ])

    # Generalized Loading Logic
    if args.dataset in ['cifar10', 'cifar100']:
        ds_cls = torchvision.datasets.CIFAR10 if args.dataset == 'cifar10' else torchvision.datasets.CIFAR100
        # Use root=dataset_root directly because folder is handled in DATASET_CONFIGS for these
        # But wait, torchvision manages folder structure internally for CIFAR.
        # If folder is '', then dataset_root is args.data_dir.
        train_ds = ds_cls(dataset_root, train=True, download=True, transform=t_tr)
        val_ds = ds_cls(dataset_root, train=False, download=True, transform=t_va)
    else:
        # All other datasets use ImageFolder structure (tiny, cub, cars, flowers, food, inat)
        train_path = os.path.join(dataset_root, 'train')
        val_path = os.path.join(dataset_root, 'val')
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}. Please format your dataset first.")

        train_ds = torchvision.datasets.ImageFolder(train_path, t_tr)
        val_ds = torchvision.datasets.ImageFolder(val_path, t_va)

    train_ldr = DataLoader(train_ds, args.batch_size, sampler=DistributedSampler(train_ds), num_workers=4)
    val_ldr = DataLoader(val_ds, args.batch_size, sampler=DistributedSampler(val_ds, shuffle=False), num_workers=4)
    return train_ldr, val_ldr, conf

# =============================================================================
# 2. EVALUATOR
# =============================================================================

def evaluate(model, loader, device, num_classes, epoch, logger, output_dir):
    model.eval()
    L = model.module.num_layers
    acc_sum = torch.zeros(L + 1, dtype=torch.float64).to(device)
    acc_sq_sum = torch.zeros(L + 1, dtype=torch.float64).to(device)
    pos_logit_sum = torch.zeros(L + 1, dtype=torch.float64).to(device)
    neg_logit_sum = torch.zeros(L + 1, dtype=torch.float64).to(device)
    gap_sum = torch.zeros(L + 1, dtype=torch.float64).to(device)
    gap_sq_sum = torch.zeros(L + 1, dtype=torch.float64).to(device)
    curl_l_sum = torch.zeros(L - 1, dtype=torch.float64).to(device)
    curl_s_sum = torch.zeros(L - 1, dtype=torch.float64).to(device)
    turb_count = torch.tensor(0.0, dtype=torch.float64).to(device)
    pav_sum = torch.zeros(L, dtype=torch.float64).to(device)
    total_samples = torch.tensor(0.0, dtype=torch.float64).to(device)
    batch_count = torch.tensor(0.0, dtype=torch.float64).to(device)

    all_l, all_s, all_t = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0); total_samples += B; batch_count += 1
            logits, states = model(x)
            probs = F.softmax(logits, -1)
            p_gt = probs.gather(-1, y.view(1, -1, 1).expand(L + 1, -1, -1)).squeeze(-1)

            correct = logits.argmax(-1).eq(y.view(1, -1))
            acc_sum += correct.float().sum(1)
            acc_sq_sum += (correct.float().mean(1).double() ** 2)

            for i in range(L + 1):
                py = logits[i].gather(1, y.view(-1, 1)).squeeze()
                mask = torch.ones_like(logits[i]).scatter_(1, y.view(-1, 1), 0.)
                pn = (logits[i] * mask).sum(1) / (num_classes - 1)
                pos_logit_sum[i] += py.sum(); neg_logit_sum[i] += pn.sum()
                gap_sum[i] += (py - pn).sum(); gap_sq_sum[i] += ((py - pn).double() ** 2).sum()
                if i < L:
                    turb_count += (correct[i].int() > correct[i+1].int()).float().sum()
                    pav_sum[i] += (p_gt[i+1] < p_gt[i]).float().sum()

            vz = states[1:] - states[:-1]; vp = probs[1:] - probs[:-1]
            vzn, vpn = F.normalize(vz, p=2, dim=-1), F.normalize(vp, p=2, dim=-1)
            for i in range(L - 1):
                curl_l_sum[i] += (1.0 - (vzn[i] * vzn[i+1]).sum(-1)).sum()
                curl_s_sum[i] += (1.0 - (vpn[i] * vpn[i+1]).sum(-1)).sum()

            all_l.append(logits); all_s.append(states); all_t.append(y)

    sync = [acc_sum, acc_sq_sum, pos_logit_sum, neg_logit_sum, gap_sum, gap_sq_sum,
            curl_l_sum, curl_s_sum, turb_count, pav_sum, total_samples, batch_count]
    for s in sync: dist.all_reduce(s)

    local_l, local_s, local_t = torch.cat(all_l, 1), torch.cat(all_s, 1), torch.cat(all_t, 0)
    if dist.is_initialized() and dist.get_world_size() > 1:
        gl, gs, gt = [[torch.zeros_like(t) for _ in range(dist.get_world_size())] for t in [local_l, local_s, local_t]]
        dist.all_gather(gl, local_l); dist.all_gather(gs, local_s); dist.all_gather(gt, local_t)
        if is_main(): fl, fs, ft = torch.cat(gl, 1).cpu(), torch.cat(gs, 1).cpu(), torch.cat(gt, 0).cpu()
    else: fl, fs, ft = local_l.cpu(), local_s.cpu(), local_t.cpu()

    if is_main():
        N, Bc = total_samples.item(), batch_count.item()
        am, gm = (acc_sum / N).cpu().numpy(), (gap_sum / N).cpu().numpy()
        as_ = np.sqrt(np.maximum((acc_sq_sum / Bc).cpu().numpy() - (am**2), 0))
        gs_ = np.sqrt(np.maximum((gap_sq_sum / N).cpu().numpy() - (gm**2), 0))
        lp, ln = (pos_logit_sum / N).cpu().numpy(), (neg_logit_sum / N).cpu().numpy()
        cl, cs, pv = (curl_l_sum / N).cpu().numpy(), (curl_s_sum / N).cpu().numpy(), (pav_sum / N).cpu().numpy()

        row = {"epoch": epoch, "acc": am[-1], "turb_freq": turb_count.item() / (N * (L-1) + 1e-8)}
        for i in range(L + 1):
            row.update({f"l{i}_acc": am[i], f"l{i}_acc_std": as_[i], f"l{i}_gap": gm[i], f"l{i}_gap_std": gs_[i], f"l{i}_pos": lp[i], f"l{i}_neg": ln[i]})
            if i < L: row[f"s{i}_pav"] = pv[i]
            if i < L - 1: row.update({f"s{i}_curl_l": cl[i], f"s{i}_curl_s": cs[i]})
        append_csv(os.path.join(output_dir, "layer_stats.csv"), row)
        torch.save({"logits": fl, "states": fs, "targets": ft}, os.path.join(output_dir, f"trajectory_ep{epoch:03d}.pth"))
        torch.save(model.module.state_dict(), os.path.join(output_dir, f"ckpt_ep{epoch:03d}.pth"))
        logger.info(f"Epoch {epoch} | Acc: {am[-1]*100:.2f}% | LatentCurl: {cl.mean():.4f} | Data Dumped.")
        return am[-1]

# =============================================================================
# 3. CRITERION & MAIN
# =============================================================================

class TrajectoryCriterion(nn.Module):
    def __init__(self, strategy, num_classes, g_max=12.0, beta=3.0):
        super().__init__()
        self.strategy, self.K, self.g_max, self.beta = strategy, num_classes, g_max, beta

    def _normalized_gap_fraction(self, t):
        if self.strategy == 'linear':
            return t
        if self.strategy == 'exponential':
            return t if abs(self.beta) < 1e-12 else math.expm1(self.beta * t) / math.expm1(self.beta)
        if self.strategy == 'cosine':
            return 0.5 * (1.0 - math.cos(math.pi * t))
        if self.strategy == 'tanh':
            if abs(self.beta) < 1e-12:
                return t
            denom = 2.0 * math.tanh(self.beta)
            return (math.tanh(self.beta * (2.0 * t - 1.0)) + math.tanh(self.beta)) / denom
        return 0.0

    def get_target(self, t, targets):
        if self.strategy == 'greedy': return F.one_hot(targets, self.K).float()
        if self.strategy == 'fisher_rao':
            u_sqrt = math.sqrt(1.0/self.K); r_y, r_o = (1-t)*u_sqrt + t, (1-t)*u_sqrt
            p_y, p_o = r_y**2, r_o**2; norm = p_y + (self.K-1)*p_o
            q = torch.full((targets.shape[0], self.K), p_o/norm, device=targets.device)
            q.scatter_(1, targets.view(-1, 1), p_y/norm); return q
        gap = self.g_max * self._normalized_gap_fraction(t)
        denom = math.exp(gap) + self.K - 1
        q = torch.full((targets.shape[0], self.K), 1.0/denom, device=targets.device)
        q.scatter_(1, targets.view(-1, 1), math.exp(gap)/denom); return q

    def forward(self, all_logits, targets, lambd, use_anneal=True):
        L = all_logits.shape[0] - 1
        if self.strategy == 'aligned':
            weights = torch.arange(1, L + 1, device=all_logits.device, dtype=all_logits.dtype)
            weights = 2.0 * weights / (L * (L + 1))
            return sum(
                weights[i - 1] * F.cross_entropy(all_logits[i], targets)
                for i in range(1, L + 1)
            )
        if self.strategy == 'baseline':
            return F.cross_entropy(all_logits[-1], targets) + sum(F.cross_entropy(all_logits[i], targets) for i in range(1, L)) / (L - 1)
        if use_anneal:
            l_traj = sum(-(self.get_target(i/L, targets) * F.log_softmax(all_logits[i], -1)).sum(-1).mean() for i in range(1, L + 1))
            return F.cross_entropy(all_logits[-1], targets) + lambd * (l_traj / L)
        return sum(-(self.get_target(i/L, targets) * F.log_softmax(all_logits[i], -1)).sum(-1).mean() for i in range(1, L + 1)) / L

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--strategy', type=str, default='exponential')
    parser.add_argument('--backbone', type=str, default='vit', choices=['vit', 'resnet18', 'resnet50', 'convnext_tiny', 'convnext_base'])
    parser.add_argument('--resnet_granularity', type=str, default='stage', choices=['stage', 'block'])
    parser.add_argument('--model_scale', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--epochs', type=int, default=200); parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--accum_steps', type=int, default=1); parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--anneal', action='store_true'); parser.add_argument('--strong_aug', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Archived default; pass the desired recipe value explicitly.')
    parser.add_argument('--g-max', type=float, default=12.0, help='Terminal target margin; archived default is 12.')
    parser.add_argument('--beta', type=float, default=3.0, help='Schedule curvature; archived default is 3.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lambda_max', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rank = setup_ddp(); device = torch.device(f"cuda:{rank}")
    logger = StatefulLogger(args.output_dir)
    if is_main():
        import json
        (Path(args.output_dir) / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    train_ldr, val_ldr, cfg = get_loaders(args)
    model = build_model(args, cfg).to(device)
    find_unused = args.backbone != 'vit' and args.strategy == 'baseline'
    model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = TrajectoryCriterion(args.strategy, cfg['classes'], g_max=args.g_max, beta=args.beta); scaler = torch.amp.GradScaler('cuda')

    steps_per_epoch = len(train_ldr)
    total_steps = args.epochs * steps_per_epoch
    start_epoch, best_acc, global_step, scheduler = maybe_resume_training(
        model,
        optimizer,
        scaler,
        args.output_dir,
        args.epochs,
        steps_per_epoch,
        logger,
    )
    if start_epoch == 0:
        evaluate(model, val_ldr, device, cfg['classes'], 0, logger, args.output_dir)

    for epoch in range(start_epoch, args.epochs):
        model.train(); train_ldr.sampler.set_epoch(epoch); optimizer.zero_grad()
        for i, (x, y) in enumerate(train_ldr):
            global_step += 1; x, y = x.to(device), y.to(device)
            lambd = args.lambda_max * (1.0 - global_step / total_steps) if args.anneal else args.lambda_max
            with torch.amp.autocast('cuda'):
                loss = criterion(model(x)[0], y, lambd, use_anneal=args.anneal) / args.accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            acc = evaluate(model, val_ldr, device, cfg['classes'], epoch + 1, logger, args.output_dir)
            if is_main() and acc > best_acc:
                best_acc = acc; torch.save(model.module.state_dict(), os.path.join(args.output_dir, "best.pth"))
            save_training_state(args.output_dir, model.module, optimizer, scheduler, scaler, epoch + 1, best_acc, global_step)
            if is_main(): logger.info(f"ETA: {logger.eta(global_step, total_steps)}")
    dist.destroy_process_group()

if __name__ == "__main__": main()
