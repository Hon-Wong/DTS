# Source provenance and release changes

The primary source is `ICML2026_DTS_Imagenet_Handoff_20260327.zip`, an original research handoff archive. It contains training code, launchers and logs, but deliberately excludes model weights. [source_manifest.json](source_manifest.json) records the archive-relative source name and original SHA-256 for every imported Python file. Private filesystem paths, account identifiers, job logs, reviewer discussions, datasets, and manuscript copies are not part of this release.

The DeiT repeated-augmentation sampler was missing from the archive and was restored from the [official upstream implementation](https://github.com/facebookresearch/deit/blob/main/samplers.py). Original upstream copyright and license notices are retained.

## Release changes

- Removed imports for unrelated, unavailable experimental architectures.
- Replaced site-specific data paths and probe-repository defaults with explicit arguments or repository-relative paths.
- Added portable recipe JSON files and a launcher. DeiT parameters were extracted from the archived shell launchers; automatic resume paths were omitted so a new run starts from scratch unless requested.
- Exposed small-scale margin, curvature, weight decay and seed arguments, retaining the archived defaults in Python. CIFAR presets explicitly select the paper's settings, including its trajectory-only objective. They are not certified reproductions of individual historical runs.
- Added the continuous linear limit of the exponential schedule at zero curvature.
- Fixed a missing functional import in DeiT dropout and an undefined target variable in the optional Fisher–Rao loss branch.
- Made the DeiT decay flag parse `False` correctly; rejected the unsupported learnable-margin option instead of accessing a nonexistent parameter.
- Updated local PyTorch checkpoint loads for trusted research payloads and guarded DeiT CUDA-only operations for CPU smoke testing.
- Made checkpoint saving work when model EMA is disabled, as required by current timm.
- Added safetensors export, strict final-head inference and regression tests. Export keeps all backbone and terminal-head parameters unchanged.

The main exponential target construction, Mixup target rule, auxiliary loss reduction and checkpoint tensor names are preserved. The ConvNeXt extension is explicitly separated because its probability-space interpolation differs from the logit-margin method.

## Validation

Release checks cover loss values against their equations, gradients, target normalization, the zero-curvature limit, entrypoint imports, portable configuration expansion, and identical DeiT final logits before and after removing auxiliary heads and serializing weights. Synthetic CPU training is an execution check, not an accuracy experiment.

The local validation environment uses Python 3.9, PyTorch 2.8.0, torchvision 0.23.0 and timm 1.0.25. CI uses Python 3.11 with the same PyTorch/torchvision/timm versions. Full multi-GPU training and the paper's benchmark scores have not been rerun for this packaging release.

The recovered environment note records a separate historical inspection using Python 3.11.3, a PyTorch 2.11 development build and timm 1.0.25. That inspection is not an independently verified environment lockfile from every original training job.
