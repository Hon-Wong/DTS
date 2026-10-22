# Training and evaluation

Run commands from the repository root. Data and output directories are supplied explicitly; no cluster account, filesystem mount, or scheduler is required by the launcher.

## DeiT-III on ImageNet-1K

| Recipe | Model | Resolution | Epochs | Per-GPU batch | Reference GPUs | Terminal loss | Initial auxiliary scale |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `deit_small_dts` | Small | 224 | 400 | 256 | 8 | CE | 0.5 |
| `deit_base_baseline` | Base | 192 | 400 | 256 | 8 | BCE | — |
| `deit_base_dts` | Base | 192 | 400 | 256 | 8 | BCE | 0.001 |
| `deit_base_dts_finetune` | Base | 224 | 20 | 64 | 8 | CE | 0.00005 |

These values come from the archived launch scripts. The historical Small script's filename contains `bce`, but its actual terminal loss is **CE**. The JSON configuration records the executed flags, not an interpretation of the filename.

All DTS recipes above use independent auxiliary heads, $G_{\max}=10$, and $\beta=0.5$. The trajectory term is averaged over layers; its weight decays linearly with training progress, while the final task term remains weighted by one.

The original `--lfa-*` flag names and parameter names are retained for checkpoint compatibility:

| Flag | Meaning |
| --- | --- |
| `--lfa` | Enable DTS heads and trajectory loss |
| `--lfa-initial-k` | Terminal margin $G_{\max}$ |
| `--lfa-initial-b` | Schedule curvature $\beta$ |
| `--lfa-loss-type` | `exponential`, `linear`, `constant`, `cosine`, `tanh`, `fisher_rao`, or `deep_supervision` |
| `--init-lfa-scale` | Coefficient of the mean auxiliary loss |
| `--decay-lfa-scale True` | Decay the auxiliary coefficient through training |
| `--no-decay-last-layer` | Keep the terminal task-loss coefficient equal to one |
| `--final-loss ce` / `bce` | Select the terminal objective |

For Mixup/CutMix targets $y$, the archived DeiT loss sets target logits to $G(t)+\log(y_i+10^{-8})$ on positive classes and zero elsewhere, then applies softmax. It does **not** use a linear interpolation between a uniform distribution and a one-hot target. With a BCE terminal objective, the terminal labels are binarized with `y > 0`.

Train the base model and then fine-tune:

```bash
python tools/train.py --recipe deit_base_dts \
  --data-path /path/to/imagenet --output-dir outputs/deit_base_dts \
  --nproc-per-node 8

python tools/train.py --recipe deit_base_dts_finetune \
  --data-path /path/to/imagenet --output-dir outputs/deit_base_dts_ft \
  --finetune outputs/deit_base_dts/best_checkpoint.pth --nproc-per-node 8
```

Resume an interrupted run by appending `--resume /path/to/checkpoint.pth` to the same recipe command. Resume restores optimizer, scheduler and scaler state. Fine-tuning instead uses `--finetune`, interpolates positional embeddings when needed, and starts a new optimizer.

Evaluate a trusted Small checkpoint:

```bash
python recipes/imagenet_deit/main.py \
  --model deit_small_patch16_LS --data-path /path/to/imagenet \
  --input-size 224 --eval-crop-ratio 1.0 \
  --lfa --lfa-head-type independent --lfa-loss-type exponential \
  --lfa-initial-k 10 --lfa-initial-b 0.5 \
  --resume outputs/deit_small_dts/best_checkpoint.pth --eval
```

This evaluation entrypoint builds both data splits, so both `train` and `val` directories must be present. Evaluation reads final-layer logits. For inference without computing auxiliary heads, use `tools/export_weights.py --deit-inference-only` and `tools/predict.py`.

The recipe launcher targets a single node. For multiple nodes, use its `--dry-run` output to recover the training arguments, then launch the entrypoint with your site's `torchrun --nnodes ... --node_rank ... --master_addr ... --master_port ...` configuration. Keep the reference **global** batch size when comparing results.

## Small-scale experiments

The CIFAR presets use ViT-Small (12 blocks, width 384, 6 heads), 200 epochs, AdamW, learning rate $5\times10^{-4}$, weight decay 0.05, and global batch size 1024 with eight GPUs. Presets select `baseline`, `greedy`, or `exponential` supervision. Add `--strong_aug` for RandAugment.

The exponential preset uses $G_{\max}=10$, $\beta=0.5$, and the mean trajectory cross-entropy without an additional terminal hard-label loss. This corresponds to the paper's small-scale objective. `--anneal` switches to the archived alternative with an additional terminal loss and a decaying auxiliary coefficient. These are different objectives; do not compare runs without recording this flag.

The archived Python defaults were $G_{\max}=12$, $\beta=3$, and weight decay 0.1. They remain available for historical compatibility, but the checked-in presets explicitly override them. These presets are **paper-configured examples**, not recovered commands for individual checkpoints.

CIFAR is downloaded through torchvision. Other datasets must be organized as ImageFolder trees:

```text
DATA_ROOT/
  tiny-imagenet-200/{train,val}/CLASS_NAME/images...
  food-101/{train,val}/CLASS_NAME/images...
  inat2019/{train,val}/CLASS_NAME/images...
```

For Tiny-ImageNet, reorganize the original flat validation directory using its annotations first. Ensure training and validation class-directory names agree. No dataset files are redistributed here.

The small-scale script saves `args.json`, `layer_stats.csv`, periodic model checkpoints and trajectory tensors. Trajectory dumps can be large; reserve sufficient disk and GPU memory. It automatically resumes from its output directory if checkpoints already exist, so use a fresh output directory for a new experiment. The archived validation sampler pads a split to equal rank lengths; for exact validation counts, use a world size that divides the split size or evaluate on one GPU.

## Frozen-backbone probing

```bash
torchrun --standalone --nproc_per_node=1 recipes/probing/probe_pretrained.py \
  --dataset imagenet --data_dir /path/to/data_root \
  --model deit_small_patch16_LS --checkpoint-format deit_ours \
  --checkpoint outputs/deit_small_dts/best_checkpoint.pth \
  --input_size 224 --epochs 10 --batch_size 128 \
  --output_dir outputs/probe_deit_small
```

Here `data_root` contains an `imagenet/` directory. `--checkpoint-format timm` supports the pretrained-model branch in the archived probing script. External pretrained models are downloaded under their own terms. The probe fits per-layer affine maps on frozen features and saves layer statistics; it does not train the backbone.

## ConvNeXt extension

`recipes/imagenet_convnext/` preserves an experimental extension from the same archive. Its `dts_exponential` branch constructs $q_t=\alpha(t)y+(1-\alpha(t))u$, with $u$ uniform. **This differs from the paper's logit-margin construction.** It is kept as a separate recipe, and its results should not be labeled as reproductions of the DeiT experiments.

```bash
torchrun --standalone --nproc_per_node=8 recipes/imagenet_convnext/main.py \
  --model convnext_tiny_dts --data_path /path/to/imagenet \
  --output_dir outputs/convnext_extension \
  --batch_size 64 --update_freq 8 --epochs 300 --lr 0.004 \
  --weight_decay 0.05 --drop_path 0.1 --warmup_epochs 20 \
  --model_ema true --model_ema_eval true --use_amp true \
  --traj_strategy dts_exponential --traj_granularity block \
  --traj_scale 0.001 --traj_beta 0.5
```

The example preserves the archived global batch size of 4096 (the original launcher used 16 GPUs, batch 64, accumulation 4). This extension is not accompanied by a validated released checkpoint.
