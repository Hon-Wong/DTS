# Checkpoints

Pretrained weights are not included in this initial source release. Original training logs and checkpoint locations have been recovered; the weight files themselves have not yet been retrieved and verified. There are no active pretrained download links at this time.

The source includes everything needed to export a locally trained DeiT checkpoint:

```bash
python tools/export_weights.py /path/to/best_checkpoint.pth \
  outputs/model.safetensors --trust-checkpoint --deit-inference-only
```

Only pass `--trust-checkpoint` for a checkpoint you trust: legacy PyTorch research checkpoints can contain pickle objects. The distributed training scripts likewise load trusted research checkpoints. The prediction tool accepts tensor-only safetensors files.

An export contains model tensors only, excludes optimizer/scheduler/scaler state, and removes DeiT auxiliary classifiers when requested. Its JSON manifest records file size, tensor count, and SHA-256. It preserves the source tensor precision. A separate model description must identify the architecture, class order, input resolution, normalization, crop ratio, training configuration, and evaluation result before a pretrained artifact is listed here.
