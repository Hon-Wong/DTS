# DTS research implementation; release maintenance by DTS contributors (2026).
# See docs/provenance.md for source and changes.
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss, LFALoss
from samplers import RASampler
from augment import new_data_aug_generator

import models
import models_v2

import utils


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)

    parser.add_argument('--ThreeAugment', action='store_true') #3augment

    parser.add_argument('--src', action='store_true') #simple random crop

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Cosub params
    parser.add_argument('--cosub', action='store_true')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true')

    # Dataset parameters
    parser.add_argument('--data-path', required=True, type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # 1. get_args_parser 增加参数
    parser.add_argument('--lfa', action='store_true', help='Enable Laminar Flow Alignment')
    parser.add_argument('--lfa-head-type', default='shared', choices=['shared', 'independent'])
    parser.add_argument('--lfa-initial-k', default=10.0, type=float)
    parser.add_argument('--lfa-initial-b', default=1.0, type=float)
    parser.add_argument('--lfa-learn-k', action='store_true')
    parser.add_argument('--lfa-reg-weight', default=0.5, type=float)
    parser.add_argument('--lfa-loss-type', default='incremental')
    parser.add_argument('--init-lfa-scale', default=1.0, type=float)
    parser.add_argument('--decay-lfa-scale', default=False, type=lambda s: s.lower() in ('true', '1', 'yes'))
    parser.add_argument('--no-decay-last-layer', action='store_false', dest='decay_last_layer')
    parser.set_defaults(decay_last_layer=True)
    parser.add_argument('--final-loss', default='bce', choices=['ce', 'bce'], type=str, help='Loss type for final layer')
    parser.add_argument('--save-interval', default=10, type=int,
                    help='Save checkpoint every X epochs (default: 50)')

    return parser


def check_keywords_in_name(name, keywords=()):
    return any(keyword in name for keyword in keywords)


def param_groups_weight_decay_with_keywords(model, weight_decay, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith('.bias') or name in skip_list or check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [
        {'params': has_decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def main(args):
    if args.lfa_learn_k:
        raise ValueError("The released fixed-margin recipe does not learn G_max; omit --lfa-learn-k.")
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
        lfa=args.lfa,
        head_type=args.lfa_head_type
    )

    print("args.finetune: ", args.finetune)
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        checkpoint_model = checkpoint['model']

        new_state_dict = {}
        for k, v in checkpoint_model.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 去掉 'module.'
            else:
                new_state_dict[k] = v
        checkpoint_model = new_state_dict
        state_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("Missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)

    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu', weights_only=False)
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # ------------------------- Modified --------------------
    # 3. 创建 Criterion 和设置优化参数
    # if args.lfa:
    #     from losses import LFALoss
    #     criterion = LFALoss(num_layers=model.get_num_layers(),
    #                         num_classes=args.nb_classes,
    #                         initial_k=args.lfa_initial_k,
    #                         reg_weight=args.lfa_reg_weight,
    #                         learn_k=args.lfa_learn_k,
    #                         loss_type=args.lfa_loss_type,
    #                         use_bce=args.bce_loss).to(device)

    #     if args.lfa_learn_k:
    #         # 【关键点】在此处显式构建参数组
    #         parameters = [
    #             {'params': model.parameters()},
    #             {'params': [criterion.k], 'lr': args.lr * 10, 'weight_decay': 0.0}
    #         ]
    #     else:
    #         parameters = model.parameters()
    # else:
    #     # 原版逻辑
    #     # criterion = LabelSmoothingCrossEntropy() if not mixup_active else SoftTargetCrossEntropy()
    #     criterion = LabelSmoothingCrossEntropy()

    #     if mixup_active:
    #         # smoothing is handled with mixup label transform
    #         criterion = SoftTargetCrossEntropy()
    #     elif args.smoothing:
    #         criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    #     else:
    #         criterion = torch.nn.CrossEntropyLoss()

    #     if args.bce_loss:
    #         criterion = torch.nn.BCEWithLogitsLoss()
    #     criterion = DistillationLoss(
    #         criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    #     )
    #     parameters = model.parameters()
    # # ------------------------- Modified --------------------

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)
    # if not args.unscale_lr:
    #     linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    #     args.lr = linear_scaled_lr
    # optimizer = create_optimizer(args, parameters)

# ------------------------- 优化后的模型与优化器设置 -------------------------

    # 1. 确定基础模型对象 (DDP 之前或 DDP 的 .module)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 2. 学习率缩放
    if not args.unscale_lr:
        # 基于 512 batch size 的线性缩放
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    # 3. 损失函数与参数配置
    if args.lfa:
        from losses import LFALoss, AdaptiveLogitFlowLoss, AdaptiveBCELogitFlowLoss
        # 初始化 LFA 损失函数
        # criterion = LFALoss(
        #     num_layers=model_without_ddp.get_num_layers(),
        #     num_classes=args.nb_classes,
        #     initial_k=args.lfa_initial_k,
        #     reg_weight=args.lfa_reg_weight,
        #     learn_k=args.lfa_learn_k,
        #     loss_type=args.lfa_loss_type,
        #     use_bce=args.bce_loss
        # ).to(device)
        criterion = AdaptiveLogitFlowLoss(
            num_classes=args.nb_classes,
            num_layers=model_without_ddp.get_num_layers(),
            fixed_scale=args.lfa_initial_k,
            fixed_bias=args.lfa_initial_b,
            schedule_type=args.lfa_loss_type,
            init_lfa_scale=args.init_lfa_scale,
            decay_lfa_scale=args.decay_lfa_scale,
            decay_last_layer=args.decay_last_layer,
            final_loss=args.final_loss,
        )

        if args.bce_loss:
            criterion = AdaptiveBCELogitFlowLoss(
                num_classes=args.nb_classes,
                num_layers=model_without_ddp.get_num_layers(),
                fixed_scale=args.lfa_initial_k,
                fixed_bias=None,
                schedule_type=args.lfa_loss_type,
                init_lfa_scale=args.init_lfa_scale,
                decay_lfa_scale=args.decay_lfa_scale,
            )

        # --- 关键：使用 model_without_ddp 提取标准的 ViT 参数组 ---
        # 这样可以自动处理哪些参数需要 Weight Decay（如权重矩阵），哪些不需要（如 bias, LayerNorm）
        skip = {}
        skip_keywords = {}
        if hasattr(model_without_ddp, 'no_weight_decay'):
            skip = model_without_ddp.no_weight_decay()
        if hasattr(model_without_ddp, 'no_weight_decay_keywords'):
            skip_keywords = model_without_ddp.no_weight_decay_keywords()

        parameters = param_groups_weight_decay_with_keywords(
            model_without_ddp, args.weight_decay, skip, skip_keywords
        )

        # --- 关键：将外部的 criterion.k 注入到参数组中 ---
        if args.lfa_learn_k:
            parameters.append({
                'params': [criterion.k],
                'lr': args.lr * 10,  # 这里的 lr 覆盖掉全局 lr，实现 $k$ 的快速收敛
                'weight_decay': 0.0  # k 作为一个尺度标量，严禁加 weight decay
            })
            # 如果你有 b (intercept) 也可以同理添加
            if hasattr(criterion, 'b'):
                parameters.append({
                    'params': [criterion.b],
                    'lr': args.lr * 10,
                    'weight_decay': 0.0
                })

        # 此时的 parameters 是一个列表，create_optimizer 能够识别这种格式并正确初始化
        optimizer = create_optimizer(args, parameters)

    else:
        # --- 原版非 LFA 逻辑 ---
        if mixup_active:
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        if args.bce_loss:
            criterion = torch.nn.BCEWithLogitsLoss()

        criterion = DistillationLoss(
            criterion, teacher_model, args.distillation_type,
            args.distillation_alpha, args.distillation_tau
        )

        skip = {}
        skip_keywords = {}
        if hasattr(model_without_ddp, 'no_weight_decay'):
            skip = model_without_ddp.no_weight_decay()
        if hasattr(model_without_ddp, 'no_weight_decay_keywords'):
            skip_keywords = model_without_ddp.no_weight_decay_keywords()
        parameters = param_groups_weight_decay_with_keywords(
            model_without_ddp, args.weight_decay, skip, skip_keywords
        )
        optimizer = create_optimizer(args, parameters)

    # 4. 统计与打印
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # ------------------------------------------------------------------------------------

    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    # criterion = LabelSmoothingCrossEntropy()

    # if mixup_active:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif args.smoothing:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    # if args.bce_loss:
    #     criterion = torch.nn.BCEWithLogitsLoss()



    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    # criterion = DistillationLoss(
    #     criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    # )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04d}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)


        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}




        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
